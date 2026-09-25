package com.idealab.grapholap.model;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;
import org.graphframes.GraphFrame;

/**
 * The OLAP selection step that precedes every aggregation.
 *
 * In graph OLAP terms this is the *dicing* operation: it cuts a cuboid out of
 * the full bipartite graph along the available dimensions before any measure is
 * computed. Every query class in this framework operates on a cuboid, never on
 * the raw graph — that is what makes the vertex/edge count on the X axis of the
 * experiment vary in a controlled way.
 *
 * Dimensions available on the Fine Foods bipartite graph:
 *   - score        (edge measure, 1.0 .. 5.0)   -> rating dice
 *   - helpfulness  (edge measure, 0.0 .. 1.0)   -> quality dice
 *   - timestamp    (edge measure, epoch secs)   -> temporal roll-up
 *   - degree       (derived vertex measure)     -> activity dice
 */
public class OlapCuboid {

    private final GraphFrame graph;
    private final String description;

    private OlapCuboid(GraphFrame graph, String description) {
        this.graph = graph;
        this.description = description;
    }

    // the two accessors: the sliced graph itself, and the name of the slice.
    public GraphFrame graph() { return graph; }
    public String description() { return description; }

    // these two numbers are the x-axis of every plot in the experiment.
    public long vertexCount() { return graph.vertices().count(); }
    public long edgeCount() { return graph.edges().count(); }

    /**
     * Drop this cuboid's cached vertex and edge sets.
     *
     * In one line: every slice keeps its vertices and edges in memory so the
     * queries stay fast. This throws that memory away after each measurement, so
     * the next query is never timed on a cache the previous one left warm.
     *
     * induce() caches both, so a long random sweep would otherwise pile up two
     * cached DataFrames per query and never release them — the driver's storage
     * memory fills, eviction starts thrashing, and the later timings measure
     * memory pressure instead of query cost. The experiment calls this after
     * every measurement so each query is timed on a clean cache.
     */
    public void release() {
        try {
            graph.edges().unpersist();
            graph.vertices().unpersist();
        } catch (Exception e) {
            System.err.println("[cuboid] release failed: " + e.getMessage());
        }
    }

    /** The whole graph: the apex cuboid, no dicing. */
    public static OlapCuboid all(GraphFrame g) {
        return new OlapCuboid(g, "all");
    }

    /**
     * Dice on the rating dimension: keep only edges whose review score falls in
     * [minScore, maxScore], then drop vertices left with no incident edge.
     */
    public static OlapCuboid byScore(GraphFrame g, double minScore, double maxScore) {
        Dataset<Row> e = g.edges()
                .filter(functions.col("score").geq(minScore)
                        .and(functions.col("score").leq(maxScore)));
        return new OlapCuboid(induce(g, e),
                String.format("score=[%.1f,%.1f]", minScore, maxScore));
    }

    /** Dice on the temporal dimension: reviews inside an epoch-second window. */
    public static OlapCuboid byTimeWindow(GraphFrame g, long fromTs, long toTs) {
        Dataset<Row> e = g.edges()
                .filter(functions.col("timestamp").geq(fromTs)
                        .and(functions.col("timestamp").leq(toTs)));
        return new OlapCuboid(induce(g, e),
                String.format("time=[%d,%d]", fromTs, toTs));
    }

    /** Dice on review quality: helpfulness ratio at or above a threshold. */
    public static OlapCuboid byHelpfulness(GraphFrame g, double minHelpfulness) {
        Dataset<Row> e = g.edges()
                .filter(functions.col("helpfulness").geq(minHelpfulness));
        return new OlapCuboid(induce(g, e),
                String.format("helpfulness>=%.2f", minHelpfulness));
    }

    /**
     * Dice on the activity dimension: keep only products reviewed at least
     * minReviews times. This is the knob the experiment sweeps, because it
     * scales the resulting subgraph smoothly across three orders of magnitude.
     */
    public static OlapCuboid byProductPopularity(GraphFrame g, long minReviews) {
        Dataset<Row> popular = g.edges()
                .groupBy("dst")
                .agg(functions.count("*").alias("n"))
                .filter(functions.col("n").geq(minReviews))
                .select(functions.col("dst").alias("keep"));

        Dataset<Row> e = g.edges()
                .join(popular, g.edges().col("dst").equalTo(popular.col("keep")))
                .drop("keep");

        return new OlapCuboid(induce(g, e),
                String.format("productReviews>=%d", minReviews));
    }

    /** Random sample of the edge set — used for the random query sweep. */
    public static OlapCuboid sample(GraphFrame g, double fraction, long seed) {
        Dataset<Row> e = g.edges().sample(false, fraction, seed);
        return new OlapCuboid(induce(g, e),
                String.format("sample=%.4f", fraction));
    }

    /**
     * In one line: this is the helper every dice method calls. It is what makes
     * a slice a real graph: after the edges are filtered, it rebuilds the vertex
     * set from the endpoints that survived, so no vertex is left behind with all
     * of its edges gone.
     *
     * Vertex-induced subgraph: after filtering edges, retain exactly the
     * vertices that still appear as an endpoint. Without this the vertex count
     * would stay at the full 330k and the X axis would be meaningless.
     */
    private static GraphFrame induce(GraphFrame g, Dataset<Row> edges) {
        Dataset<Row> e = edges.cache();
        Dataset<Row> ids = e.selectExpr("src as id")
                .union(e.selectExpr("dst as id"))
                .distinct();
        Dataset<Row> v = g.vertices().join(ids, "id").cache();
        return GraphFrame.apply(v, e);
    }
}
