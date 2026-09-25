package com.idealab.grapholap.queries;

import com.idealab.grapholap.model.OlapCuboid;
import com.idealab.grapholap.model.QueryResult;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;
import org.graphframes.GraphFrame;
import scala.collection.JavaConverters;

import java.util.ArrayList;
import java.util.List;

/**
 * QUERY CLASS 2 — Path &amp; Distance-based Aggregations.
 *
 * Required by the exercise:
 *   2a. shortest-path distance metrics
 *   2b. multi-hop reachability counts
 *   2c. path-length distributions between node subsets
 *
 * Implementation note: GraphX's shortestPaths (landmark BFS) is used via the
 * GraphFrames wrapper, then the *aggregation* over those distances is done in
 * Spark SQL. Memgraph never evaluates a path query.
 */
public class PathQueries {

    private static final String CLASS_NAME = "path";

    /**
     * 2a. Shortest-path distance metrics.
     *
     * In one line: breadth-first search out from the biggest hubs, then the
     * histogram of how far everything else sits from them.
     *
     * Example: stand on the most-reviewed coffee and ask how far away everything
     * else is. Distance 1 is the users who reviewed it. Distance 2 is the other
     * products those users bought. Distance 3 is the users of those products. The
     * answer is the count at each distance.
     *
     * Runs a landmark BFS from a set of chosen landmark vertices, then
     * aggregates the resulting distance vectors:
     *   - distance histogram (how many vertices sit at distance d)
     *   - mean / max distance per landmark (the landmark's eccentricity proxy)
     *
     * On the bipartite graph, odd distances land on the opposite partition and
     * even distances on the same one — visible in the histogram.
     */
    public static QueryResult shortestPathMetrics(OlapCuboid cuboid, int nLandmarks) {
        long t0 = System.currentTimeMillis();
        GraphFrame g = undirected(cuboid.graph());

        // landmarks = highest-degree vertices, so BFS actually covers the cuboid
        List<Object> landmarks = new ArrayList<>();
        List<Row> hubs = g.edges().groupBy("src")
                .agg(functions.count("*").alias("deg"))
                .orderBy(functions.desc("deg"))
                .limit(nLandmarks)
                .collectAsList();
        for (Row r : hubs) landmarks.add(r.getString(0));

        if (landmarks.isEmpty()) {
            return new QueryResult(CLASS_NAME, "shortestPathMetrics_L" + nLandmarks,
                    cuboid.vertexCount(), cuboid.edgeCount(),
                    System.currentTimeMillis() - t0, 0,
                    "landmarks=0;" + cuboid.description());
        }

        Dataset<Row> sp = g.shortestPaths()
                .landmarks(JavaConverters.asScalaBuffer(landmarks).toSeq())
                .run();

        // distances arrives as a map<landmark,int>; explode it to aggregate
        Dataset<Row> exploded = sp.select(
                functions.col("id"),
                functions.explode(functions.col("distances")).as(new String[]{"landmark", "dist"}));

        Dataset<Row> histogram = exploded.groupBy("dist")
                .agg(functions.count("*").alias("vertex_count"))
                .orderBy("dist");

        Dataset<Row> perLandmark = exploded.groupBy("landmark")
                .agg(functions.count("*").alias("reached"),
                     functions.round(functions.avg("dist"), 3).alias("avg_dist"),
                     functions.max("dist").alias("max_dist"));

        long rows = histogram.count() + perLandmark.count();
        long ms = System.currentTimeMillis() - t0;

        System.out.printf("%n--- 2a. shortest-path distance histogram, %d landmarks  [%s] ---%n",
                landmarks.size(), cuboid.description());
        histogram.show(15, false);
        System.out.println("    per-landmark distance aggregation:");
        perLandmark.show(10, false);

        return new QueryResult(CLASS_NAME, "shortestPathMetrics_L" + nLandmarks,
                cuboid.vertexCount(), cuboid.edgeCount(), ms, rows,
                "landmarks=" + landmarks.size() + ";" + cuboid.description());
    }

    /**
     * 2b. Multi-hop reachability counts.
     *
     * In one line: starting from a seed, how much of the graph opens up after one
     * hop, two hops, three hops.
     *
     * Example: start from one heavy reviewer. After 1 hop I see the 500 products
     * he reviewed. After 2 hops, the thousands of users who reviewed those. After
     * 3 hops, most of the graph. The answer is that curve - how fast it saturates.
     *
     * |{v : dist(seed, v) <= k}| for k = 1..maxHops, computed by iterative
     * frontier expansion from a seed set. The aggregation is the reachability
     * growth curve — how fast the neighbourhood saturates.
     */
    public static QueryResult reachabilityCounts(OlapCuboid cuboid, int nSeeds, int maxHops) {
        long t0 = System.currentTimeMillis();
        GraphFrame g = undirected(cuboid.graph());

        Dataset<Row> adj = g.edges().selectExpr("src", "dst").distinct().cache();

        Dataset<Row> seeds = adj.groupBy("src")
                .agg(functions.count("*").alias("deg"))
                .orderBy(functions.desc("deg"))
                .limit(nSeeds)
                .selectExpr("src as seed");

        // reached: (seed, vertex) pairs discovered so far
        Dataset<Row> reached = seeds.selectExpr("seed", "seed as v");
        Dataset<Row> frontier = reached;
        List<String> curve = new ArrayList<>();

        for (int h = 1; h <= maxHops; h++) {
            // localCheckpoint() truncates the plan each hop. A plain cache() keeps
            // the full lineage, so hop h carries every join from hops 1..h-1 and
            // the driver eventually OOMs building the plan string.
            frontier = frontier.as("f")
                    .join(adj.as("a"), functions.col("f.v").equalTo(functions.col("a.src")))
                    .selectExpr("f.seed as seed", "a.dst as v")
                    .except(reached)          // only genuinely new vertices
                    .localCheckpoint();
            Dataset<Row> prevReached = reached;
            reached = reached.union(frontier).distinct().localCheckpoint();
            prevReached.unpersist();

            Dataset<Row> perSeed = reached.groupBy("seed")
                    .agg(functions.count("*").alias("reachable_within_" + h));
            long totalReached = reached.count();
            curve.add("h" + h + "=" + totalReached);

            System.out.printf("%n--- 2b. reachability within %d hop(s): %d (seed,vertex) pairs ---%n",
                    h, totalReached);
            perSeed.show(10, false);

            if (frontier.isEmpty()) {
                System.out.printf("    frontier exhausted at hop %d%n", h);
                break;
            }
        }

        long reachedRows = reached.count();
        adj.unpersist();
        reached.unpersist();

        long ms = System.currentTimeMillis() - t0;
        return new QueryResult(CLASS_NAME, "reachabilityCounts_h" + maxHops,
                cuboid.vertexCount(), cuboid.edgeCount(), ms, reachedRows,
                "seeds=" + nSeeds + ";maxHops=" + maxHops + ";" + String.join("|", curve)
                        + ";" + cuboid.description());
    }

    /**
     * 2c. Path-length distribution between two node subsets.
     *
     * In one line: how far apart the busiest users and the busiest products are,
     * as a distribution.
     *
     * Example: the 20 busiest reviewers against the 20 most-reviewed coffees.
     * Distance 1 means that reviewer reviewed that coffee himself. Distance 3
     * means he is linked to it through somebody else. Only odd distances ever
     * appear - you cannot walk from a user to a product in an even number of steps.
     *
     * Subsets are chosen along OLAP dimensions: A = high-degree users,
     * B = high-degree products. BFS from A gives dist(a, b) for every b in B;
     * the aggregation is the distribution of those path lengths, which
     * characterises how tightly the two subsets are coupled.
     */
    public static QueryResult pathLengthDistribution(OlapCuboid cuboid, int subsetSize) {
        long t0 = System.currentTimeMillis();
        GraphFrame g = undirected(cuboid.graph());

        Dataset<Row> deg = g.edges().groupBy("src")
                .agg(functions.count("*").alias("deg"))
                .cache();

        // subset A: top users, subset B: top products (bipartite partitions)
        List<Object> subsetA = new ArrayList<>();
        for (Row r : deg.filter(functions.col("src").startsWith("u_"))
                        .orderBy(functions.desc("deg")).limit(subsetSize)
                        .collectAsList()) {
            subsetA.add(r.getString(0));
        }
        List<String> subsetB = new ArrayList<>();
        for (Row r : deg.filter(functions.col("src").startsWith("p_"))
                        .orderBy(functions.desc("deg")).limit(subsetSize)
                        .collectAsList()) {
            subsetB.add(r.getString(0));
        }

        if (subsetA.isEmpty() || subsetB.isEmpty()) {
            return new QueryResult(CLASS_NAME, "pathLengthDistribution_n" + subsetSize,
                    cuboid.vertexCount(), cuboid.edgeCount(),
                    System.currentTimeMillis() - t0, 0,
                    "emptySubset;" + cuboid.description());
        }

        Dataset<Row> sp = g.shortestPaths()
                .landmarks(JavaConverters.asScalaBuffer(subsetA).toSeq())
                .run();

        Dataset<Row> aToB = sp
                .filter(functions.col("id").isin(subsetB.toArray()))
                .select(functions.col("id"),
                        functions.explode(functions.col("distances")).as(new String[]{"from_a", "path_len"}));

        Dataset<Row> dist = aToB.groupBy("path_len")
                .agg(functions.count("*").alias("pair_count"))
                .orderBy("path_len");

        Dataset<Row> summary = aToB.agg(
                functions.count("*").alias("pairs_connected"),
                functions.round(functions.avg("path_len"), 3).alias("avg_path_len"),
                functions.min("path_len").alias("min_path_len"),
                functions.max("path_len").alias("max_path_len"));

        long rows = dist.count();
        long ms = System.currentTimeMillis() - t0;

        System.out.printf("%n--- 2c. path-length distribution, |A|=%d users -> |B|=%d products  [%s] ---%n",
                subsetA.size(), subsetB.size(), cuboid.description());
        dist.show(15, false);
        summary.show(false);

        return new QueryResult(CLASS_NAME, "pathLengthDistribution_n" + subsetSize,
                cuboid.vertexCount(), cuboid.edgeCount(), ms, rows,
                "|A|=" + subsetA.size() + ";|B|=" + subsetB.size() + ";" + cuboid.description());
    }

    /** Bidirectional view: path metrics are meaningless on a one-way bipartite graph. */
    private static GraphFrame undirected(GraphFrame g) {
        Dataset<Row> both = g.edges().selectExpr("src", "dst")
                .union(g.edges().selectExpr("dst as src", "src as dst"))
                .distinct().cache();
        return GraphFrame.apply(g.vertices(), both);
    }
}
