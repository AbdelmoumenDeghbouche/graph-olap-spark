package com.idealab.grapholap.queries;

import com.idealab.grapholap.model.OlapCuboid;
import com.idealab.grapholap.model.QueryResult;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;
import org.graphframes.GraphFrame;

/**
 * QUERY CLASS 3 — Centrality &amp; Density Metrics.
 *
 * Required by the exercise:
 *   3a. local clustering coefficients
 *   3b. k-core subgraph decompositions
 *   3c. PageRank-style iterative rank aggregations over filtered subgraphs
 *
 * All three run in Spark. k-core is implemented as an explicit iterative
 * peeling loop rather than a library call, so the "iterative aggregation"
 * behaviour is visible in the code and in the timings.
 */
public class CentralityQueries {

    private static final String CLASS_NAME = "centrality";

    /**
     * 3a. Local clustering coefficient.
     *
     * In one line: are my neighbours also connected to each other? On a bipartite
     * graph the honest answer is zero, and that is a fact about the data.
     *
     * Example: a triangle here would mean a user reviewed a coffee, the coffee
     * reviewed a second user, and that user reviewed the first one back. That
     * cannot happen: users only ever point at products. So there are no triangles
     * and the coefficient is exactly zero - it proves the graph really is
     * bipartite, it is not a bug in the query.
     *
     *   C(v) = 2 * triangles(v) / (deg(v) * (deg(v) - 1))
     *
     * Computed on the undirected projection. On a pure bipartite graph the
     * triangle count is 0 by construction (no odd cycles) — so this query is
     * most informative on the co-review projection, and the framework reports
     * the aggregate honestly either way. That zero is a real structural fact
     * about the dataset, not a failure.
     */
    public static QueryResult localClusteringCoefficient(OlapCuboid cuboid) {
        long t0 = System.currentTimeMillis();
        GraphFrame g = undirected(cuboid.graph());

        Dataset<Row> tri = g.triangleCount().run()
                .selectExpr("id", "count as triangles");
        Dataset<Row> deg = g.degrees();

        Dataset<Row> cc = tri.join(deg, "id")
                .withColumn("lcc",
                        functions.when(functions.col("degree").gt(1),
                                functions.round(
                                        functions.col("triangles").multiply(2.0)
                                                .divide(functions.col("degree")
                                                        .multiply(functions.col("degree").minus(1))), 6))
                         .otherwise(functions.lit(0.0)));

        Dataset<Row> summary = cc.agg(
                functions.count("*").alias("vertices"),
                functions.round(functions.avg("lcc"), 6).alias("avg_lcc"),
                functions.max("lcc").alias("max_lcc"),
                functions.sum("triangles").alias("total_triangle_ends"));

        long rows = cc.count();
        long ms = System.currentTimeMillis() - t0;

        System.out.printf("%n--- 3a. local clustering coefficient  [%s] ---%n", cuboid.description());
        summary.show(false);
        System.out.println("    highest-LCC vertices:");
        cc.orderBy(functions.desc("lcc")).show(10, false);

        return new QueryResult(CLASS_NAME, "localClusteringCoefficient",
                cuboid.vertexCount(), cuboid.edgeCount(), ms, rows, cuboid.description());
    }

    /**
     * 3b. k-core decomposition.
     *
     * In one line: keep deleting every vertex with fewer than k connections, and
     * report what survives for each k - the density profile of the slice.
     *
     * Example: for k=5, throw away every user with fewer than 5 reviews and every
     * coffee with fewer than 5 reviews. Those deletions push other vertices below
     * 5, so repeat until nothing moves. What is left is the serious core: the
     * committed reviewers and the genuinely popular products.
     *
     * Iterative peeling: repeatedly delete every vertex whose degree in the
     * surviving subgraph is &lt; k, until nothing more can be removed. What
     * remains is the k-core. Reports the core size for each k in 1..maxK, i.e.
     * the density profile of the cuboid.
     */
    public static QueryResult kCoreDecomposition(OlapCuboid cuboid, int maxK) {
        long t0 = System.currentTimeMillis();
        GraphFrame g0 = undirected(cuboid.graph());

        // materialised once and reused for every k — each k restarts the peel from
        // the same symmetric edge set, so re-deriving it per k would recompute the
        // whole union and re-cache an identical plan maxK times over
        Dataset<Row> base = g0.edges().selectExpr("src", "dst").localCheckpoint();

        StringBuilder profile = new StringBuilder();
        long rows = 0;

        System.out.printf("%n--- 3b. k-core decomposition, k=1..%d  [%s] ---%n", maxK, cuboid.description());
        System.out.println("      k | core_vertices | core_edges | iterations");
        System.out.println("    ----+---------------+------------+-----------");

        for (int k = 1; k <= maxK; k++) {
            Dataset<Row> edges = base;
            int iterations = 0;

            while (true) {
                iterations++;
                // degree in the CURRENT surviving edge set
                Dataset<Row> deg = edges.groupBy("src")
                        .agg(functions.count("*").alias("d"));
                Dataset<Row> survivors = deg.filter(functions.col("d").geq(k))
                        .selectExpr("src as keep");

                // an edge survives only if BOTH endpoints survive
                // localCheckpoint() truncates the logical plan. Without it every
                // peeling round stacks two more joins onto the previous plan, and
                // the plan string Spark builds on each action grows until the
                // driver dies of OutOfMemoryError. It is eager, so `next` no
                // longer depends on `edges` once it returns.
                Dataset<Row> next = edges.as("e")
                        .join(survivors.as("s1"),
                              functions.col("e.src").equalTo(functions.col("s1.keep")))
                        .join(survivors.as("s2"),
                              functions.col("e.dst").equalTo(functions.col("s2.keep")))
                        .selectExpr("e.src as src", "e.dst as dst")
                        .localCheckpoint();

                long before = edges.count();
                long after = next.count();
                if (edges != base) edges.unpersist();       // previous round is dead
                edges = next;
                if (after == before || after == 0) break;   // fixed point
                if (iterations > 50) break;                 // safety ceiling
            }

            long coreEdges = edges.count();
            long coreVertices = coreEdges == 0 ? 0
                    : edges.selectExpr("src as id").union(edges.selectExpr("dst as id"))
                           .distinct().count();
            if (edges != base) edges.unpersist();           // do not carry k into k+1
            rows++;
            profile.append("k").append(k).append("=").append(coreVertices).append("|");
            System.out.printf("    %3d | %13d | %10d | %10d%n", k, coreVertices, coreEdges / 2, iterations);

            if (coreEdges == 0) {
                System.out.printf("    (empty at k=%d, higher k also empty)%n", k);
                break;
            }
        }

        base.unpersist();

        long ms = System.currentTimeMillis() - t0;
        return new QueryResult(CLASS_NAME, "kCoreDecomposition_k" + maxK,
                cuboid.vertexCount(), cuboid.edgeCount(), ms, rows,
                "maxK=" + maxK + ";" + profile + cuboid.description());
    }

    /**
     * 3c. PageRank over a filtered subgraph.
     *
     * In one line: PageRank, but only inside the slice - then split the importance
     * between the users and the products.
     *
     * Example: importance flows along the reviews. A coffee reviewed by many
     * people is important, and a user who reviewed many important coffees is
     * important. Then we add it up on each side: how much of the total importance
     * sits on the users, how much on the products.
     *
     * The "filtered subgraph" is exactly the OLAP cuboid handed in — so this is
     * rank aggregation restricted to a dice of the graph, which is the
     * behaviour the exercise asks for. Aggregates the resulting rank
     * distribution and splits the mass across the two partitions.
     */
    public static QueryResult pageRankAggregation(OlapCuboid cuboid, int iterations) {
        long t0 = System.currentTimeMillis();
        GraphFrame g = cuboid.graph();

        Dataset<Row> pr = g.pageRank()
                .resetProbability(0.15)
                .maxIter(iterations)
                .run()
                .vertices()
                .select("id", "label", "pagerank");

        // aggregation 1: rank mass per partition (Users vs Products)
        Dataset<Row> byLabel = pr.groupBy("label")
                .agg(functions.count("*").alias("vertices"),
                     functions.round(functions.sum("pagerank"), 4).alias("total_rank"),
                     functions.round(functions.avg("pagerank"), 6).alias("avg_rank"),
                     functions.round(functions.max("pagerank"), 4).alias("max_rank"));

        // aggregation 2: the top-ranked vertices in this cuboid
        Dataset<Row> top = pr.orderBy(functions.desc("pagerank")).limit(10);

        long rows = pr.count();
        long ms = System.currentTimeMillis() - t0;

        System.out.printf("%n--- 3c. PageRank, %d iterations  [%s] ---%n", iterations, cuboid.description());
        byLabel.show(false);
        System.out.println("    top-ranked vertices:");
        top.show(10, false);

        return new QueryResult(CLASS_NAME, "pageRankAggregation_i" + iterations,
                cuboid.vertexCount(), cuboid.edgeCount(), ms, rows,
                "iterations=" + iterations + ";" + cuboid.description());
    }

    // a two-way view of the graph: triangles and cores need edges that go both ways.
    private static GraphFrame undirected(GraphFrame g) {
        Dataset<Row> both = g.edges().selectExpr("src", "dst")
                .union(g.edges().selectExpr("dst as src", "src as dst"))
                .distinct().cache();
        return GraphFrame.apply(g.vertices(), both);
    }
}
