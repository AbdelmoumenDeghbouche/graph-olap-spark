package com.idealab.grapholap.queries;

import com.idealab.grapholap.model.OlapCuboid;
import com.idealab.grapholap.model.QueryResult;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;
import org.graphframes.GraphFrame;

/**
 * QUERY CLASS 1 — Neighborhood &amp; Degree Aggregations.
 *
 * Required by the exercise:
 *   1a. multi-hop degree distributions
 *   1b. localized ego-network aggregations
 *   1c. edge-weight / rating summaries
 *
 * All three are computed with Spark DataFrame/GraphFrames operators. No Cypher
 * aggregation is involved: the graph arrives from Memgraph as raw rows.
 */
public class NeighborhoodQueries {

    private static final String CLASS_NAME = "neighborhood";

    /**
     * 1a. Multi-hop degree distribution.
     *
     * In one line: how many products a user reviewed, then how many other users
     * reviewed those same products - a roll-up along the distance dimension.
     *
     * Example: I reviewed 3 coffees. Hop 1 is those 3 coffees. Hop 2 is everyone
     * else who reviewed the same 3 coffees - maybe 4000 people. The output is not
     * my number, it is the histogram: how many users have a 2-hop degree of 10,
     * of 100, of 4000.
     *
     * hop 1 = |direct neighbours|            (a user's reviewed products)
     * hop 2 = |neighbours of neighbours|     (other users who reviewed the same products)
     * hop k = iterated frontier expansion, duplicates removed each round.
     *
     * The aggregation is the histogram: how many vertices have k-hop degree d.
     * This is the graph analogue of a roll-up along the "distance" dimension.
     */
    public static QueryResult multiHopDegreeDistribution(OlapCuboid cuboid, int hops) {
        long t0 = System.currentTimeMillis();
        GraphFrame g = cuboid.graph();

        // symmetric adjacency so the frontier can traverse User<->Product
        Dataset<Row> adj = g.edges().selectExpr("src", "dst")
                .union(g.edges().selectExpr("dst as src", "src as dst"))
                .distinct()
                .cache();

        // frontier: (origin, reached) — starts as the 1-hop neighbourhood
        Dataset<Row> frontier = adj.selectExpr("src as origin", "dst as reached");
        Dataset<Row> reached = frontier;

        for (int h = 2; h <= hops; h++) {
            frontier = frontier.as("f")
                    .join(adj.as("a"),
                          functions.col("f.reached").equalTo(functions.col("a.src")))
                    .selectExpr("f.origin as origin", "a.dst as reached")
                    .distinct();
            reached = reached.union(frontier).distinct();
        }

        // exclude the origin itself from its own k-hop neighbourhood
        Dataset<Row> degrees = reached
                .filter(functions.col("origin").notEqual(functions.col("reached")))
                .groupBy("origin")
                .agg(functions.count("*").alias("khop_degree"));

        // the aggregation proper: distribution over degree values
        Dataset<Row> dist = degrees.groupBy("khop_degree")
                .agg(functions.count("*").alias("vertex_count"))
                .orderBy("khop_degree");

        long rows = dist.count();
        long ms = System.currentTimeMillis() - t0;

        System.out.printf("%n--- 1a. %d-hop degree distribution  [%s] ---%n", hops, cuboid.description());
        dist.show(15, false);

        return new QueryResult(CLASS_NAME, "multiHopDegreeDistribution_h" + hops,
                cuboid.vertexCount(), cuboid.edgeCount(), ms, rows,
                "hops=" + hops + ";" + cuboid.description());
    }

    /**
     * 1b. Localized ego-network aggregation.
     *
     * In one line: take the biggest hubs, and aggregate the rating and the density
     * inside the small neighbourhood around each one.
     *
     * Example: take the single most-reviewed coffee. Its ego-network is that
     * coffee plus every user who reviewed it - one small circle cut out of the
     * graph. We report how big that circle is, what it rates the coffee on
     * average, and how densely connected it is inside.
     *
     * For each of the top-N hub vertices, aggregate over its ego-network
     * (the vertex + its 1-hop neighbours + the edges among them):
     *   - ego size, internal edge count
     *   - mean / min / max rating inside the ego-net
     *   - ego density = |E| / (|V| * (|V|-1)), the local connectedness measure
     */
    public static QueryResult egoNetworkAggregation(OlapCuboid cuboid, int topN) {
        long t0 = System.currentTimeMillis();
        GraphFrame g = cuboid.graph();

        Dataset<Row> adj = g.edges().selectExpr("src", "dst", "score")
                .union(g.edges().selectExpr("dst as src", "src as dst", "score"))
                .distinct()
                .cache();

        // pick the ego centres: the highest-degree vertices in this cuboid
        Dataset<Row> hubs = adj.groupBy("src")
                .agg(functions.count("*").alias("deg"))
                .orderBy(functions.desc("deg"))
                .limit(topN)
                .selectExpr("src as ego");

        // every (ego, member) pair: the ego plus its 1-hop neighbourhood
        Dataset<Row> members = hubs.as("h")
                .join(adj.as("a"), functions.col("h.ego").equalTo(functions.col("a.src")))
                .selectExpr("h.ego as ego", "a.dst as member")
                .union(hubs.selectExpr("ego", "ego as member"))
                .distinct()
                .cache();

        // edges internal to each ego-net: both endpoints must be members
        Dataset<Row> internal = members.as("m1")
                .join(adj.as("a"), functions.col("m1.member").equalTo(functions.col("a.src")))
                .join(members.as("m2"),
                      functions.col("a.dst").equalTo(functions.col("m2.member"))
                               .and(functions.col("m1.ego").equalTo(functions.col("m2.ego"))))
                .selectExpr("m1.ego as ego", "a.src as src", "a.dst as dst", "a.score as score");

        Dataset<Row> sizes = members.groupBy("ego")
                .agg(functions.count("*").alias("ego_size"));

        Dataset<Row> stats = internal.groupBy("ego")
                .agg(functions.count("*").alias("internal_edges"),
                     functions.round(functions.avg("score"), 3).alias("avg_score"),
                     functions.min("score").alias("min_score"),
                     functions.max("score").alias("max_score"));

        Dataset<Row> agg = sizes.join(stats, "ego")
                .withColumn("ego_density",
                        functions.round(
                                functions.col("internal_edges")
                                        .divide(functions.col("ego_size")
                                                .multiply(functions.col("ego_size").minus(1))), 5))
                .orderBy(functions.desc("ego_size"));

        long rows = agg.count();
        long ms = System.currentTimeMillis() - t0;

        System.out.printf("%n--- 1b. ego-network aggregation, top %d hubs  [%s] ---%n", topN, cuboid.description());
        agg.show(15, false);

        return new QueryResult(CLASS_NAME, "egoNetworkAggregation_top" + topN,
                cuboid.vertexCount(), cuboid.edgeCount(), ms, rows,
                "topN=" + topN + ";" + cuboid.description());
    }

    /**
     * 1c. Edge-weight / rating summaries.
     *
     * In one line: per product - how many reviews, the average rating, the spread.
     * Classic OLAP aggregation, with the graph supplying the grouping.
     *
     * Example: one row per coffee - 900 reviews, average 4.6, min 1, max 5,
     * stddev 0.8. Two coffees can both average 4.6: one where everybody agrees,
     * one where half gave 5 and half gave 1. The stddev is what tells them apart.
     *
     * Per-product roll-up over the rating measure: count, mean, stddev, spread,
     * plus the mean helpfulness of the reviews. This is classic OLAP
     * aggregation, with the graph supplying the grouping structure.
     */
    public static QueryResult ratingSummary(OlapCuboid cuboid, long minReviews) {
        long t0 = System.currentTimeMillis();

        Dataset<Row> agg = cuboid.graph().edges()
                .groupBy("dst")
                .agg(functions.count("*").alias("n_reviews"),
                     functions.round(functions.avg("score"), 3).alias("avg_score"),
                     functions.round(functions.stddev("score"), 3).alias("stddev_score"),
                     functions.min("score").alias("min_score"),
                     functions.max("score").alias("max_score"),
                     functions.round(functions.avg("helpfulness"), 3).alias("avg_helpfulness"))
                .filter(functions.col("n_reviews").geq(minReviews))
                .orderBy(functions.desc("n_reviews"));

        long rows = agg.count();
        long ms = System.currentTimeMillis() - t0;

        System.out.printf("%n--- 1c. rating summary, products with >= %d reviews  [%s] ---%n",
                minReviews, cuboid.description());
        agg.show(15, false);

        return new QueryResult(CLASS_NAME, "ratingSummary_min" + minReviews,
                cuboid.vertexCount(), cuboid.edgeCount(), ms, rows,
                "minReviews=" + minReviews + ";" + cuboid.description());
    }
}
