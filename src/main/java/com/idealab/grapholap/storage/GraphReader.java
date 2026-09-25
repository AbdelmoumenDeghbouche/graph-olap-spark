package com.idealab.grapholap.storage;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.graphframes.GraphFrame;

import java.util.List;

/**
 * The storage -> compute boundary.
 *
 * Pulls raw vertices and edges out of Memgraph and materialises a Spark
 * GraphFrame. Everything downstream of this class is pure Spark: this is what
 * makes the exercise's NOTE hold — Memgraph persists, Spark aggregates.
 *
 * The bipartite graph is  (User) -[REVIEWED {score}]-> (Product).
 */
public class GraphReader {

    /** Directed bipartite view straight from storage. */
    public static GraphFrame fromMemgraph(SparkSession spark, String boltUri, int limit) {
        List<Object[]> raw;
        try (MemgraphStorage store = new MemgraphStorage(boltUri)) {
            store.verifyConnectivity();
            raw = store.readAllEdges(limit);
        }
        System.out.printf("[reader] pulled %d edges out of Memgraph%n", raw.size());
        return build(spark, raw);
    }

    /**
     * Fast path used by the experiment harness: read the same CSV that was
     * persisted to Memgraph, so a 60-query sweep is not 60 full Bolt scans.
     * The graph content is identical either way.
     */
    public static GraphFrame fromCsv(SparkSession spark, String edgesCsv) {
        Dataset<Row> e = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(edgesCsv)
                .selectExpr("src", "dst",
                            "cast(score as double) as score",
                            "cast(helpfulness as double) as helpfulness",
                            "cast(timestamp as long) as timestamp");

        Dataset<Row> v = e.selectExpr("src as id").distinct()
                .union(e.selectExpr("dst as id").distinct())
                .distinct();
        // label is derived from the id prefix written by the parser (u_ / p_)
        v = v.withColumn("label",
                org.apache.spark.sql.functions.when(
                        org.apache.spark.sql.functions.col("id").startsWith("u_"),
                        org.apache.spark.sql.functions.lit("User"))
                 .otherwise(org.apache.spark.sql.functions.lit("Product")));

        return GraphFrame.apply(v.cache(), e.cache());
    }

    private static GraphFrame build(SparkSession spark, List<Object[]> raw) {
        JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());

        StructType edgeSchema = new StructType()
                .add("src", DataTypes.StringType, false)
                .add("dst", DataTypes.StringType, false)
                .add("score", DataTypes.DoubleType, false)
                .add("helpfulness", DataTypes.DoubleType, false)
                .add("timestamp", DataTypes.LongType, false);

        JavaRDD<Row> edgeRows = jsc.parallelize(raw)
                .map(a -> RowFactory.create(a[0], a[1], a[2], a[3], a[4]));
        Dataset<Row> e = spark.createDataFrame(edgeRows, edgeSchema);

        Dataset<Row> v = e.selectExpr("src as id").distinct()
                .union(e.selectExpr("dst as id").distinct())
                .distinct()
                .withColumn("label",
                        org.apache.spark.sql.functions.when(
                                org.apache.spark.sql.functions.col("id").startsWith("u_"),
                                org.apache.spark.sql.functions.lit("User"))
                         .otherwise(org.apache.spark.sql.functions.lit("Product")));

        return GraphFrame.apply(v.cache(), e.cache());
    }

    /**
     * Undirected projection of the bipartite graph.
     *
     * Needed by the metrics that are only meaningful on an undirected view:
     * local clustering coefficient, k-core, and multi-hop reachability. A
     * bipartite User->Product graph has no triangles among Users unless the
     * co-review paths are traversable in both directions.
     */
    public static GraphFrame undirected(SparkSession spark, GraphFrame g) {
        Dataset<Row> both = g.edges().selectExpr("src", "dst", "score")
                .union(g.edges().selectExpr("dst as src", "src as dst", "score"))
                .distinct();
        return GraphFrame.apply(g.vertices(), both.cache());
    }
}
