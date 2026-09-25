package com.idealab.grapholap.queries;

import com.idealab.grapholap.model.OlapCuboid;
import com.idealab.grapholap.model.QueryResult;
import com.idealab.grapholap.storage.GraphReader;
import org.apache.spark.sql.SparkSession;
import org.graphframes.GraphFrame;

import java.util.ArrayList;
import java.util.List;

/**
 * INPUT INSTANCES demonstrating the framework's functionality.
 *
 * Runs every one of the nine required queries across the three classes, each on
 * an explicitly chosen OLAP cuboid, so the output shows both what the query
 * computes and which slice of the graph it ran on.
 *
 * Usage:
 *   spark-submit --class com.idealab.grapholap.queries.DemoQueries graph-olap.jar \
 *       <edgesCsvOrMemgraph> [boltUri]
 *
 *   arg0 = "memgraph"          -> read the graph out of Memgraph over Bolt
 *          <path to edges.csv> -> read the same graph from HDFS/local CSV
 */
public class DemoQueries {

    // the demonstration: four chosen cuboids, and on each one the queries that show
    // what the dicing changed. Nine distinct queries, eighteen evaluations.
    //
    // The four cuboids are hand-picked, not random. INSTANCE 1 proves completeness
    // (all nine queries run there). INSTANCES 2, 3 and 4 each change exactly one
    // thing against that baseline, so the difference in the output is attributable
    // to the dice and to nothing else.
    //
    //   #   SLICE                                          QUERY
    //   --  ---------------------------------------------  --------------------------------------
    //   INSTANCE 1 - products with >= 300 reviews  (small dense cuboid, all 9 queries)
    //    1  productReviews>=300                            1a multiHopDegreeDistribution  hops=2
    //    2  productReviews>=300                            1b egoNetworkAggregation       top 5
    //    3  productReviews>=300                            1c ratingSummary               min 300
    //    4  productReviews>=300                            2a shortestPathMetrics         3 landmarks
    //    5  productReviews>=300                            2b reachabilityCounts          3 seeds, 3 hops
    //    6  productReviews>=300                            2c pathLengthDistribution      subset 5
    //    7  productReviews>=300                            3a localClusteringCoefficient
    //    8  productReviews>=300                            3b kCoreDecomposition          k=1..5
    //    9  productReviews>=300                            3c pageRankAggregation         10 iters
    //
    //   INSTANCE 2 - 5-star reviews on products with >= 100 reviews  (RATING dice)
    //   10  score=[5.0,5.0] on productReviews>=100         1a multiHopDegreeDistribution  hops=2
    //   11  score=[5.0,5.0] on productReviews>=100         1c ratingSummary               min 50
    //   12  score=[5.0,5.0] on productReviews>=100         3c pageRankAggregation         10 iters
    //   13  score=[5.0,5.0] on productReviews>=100         3b kCoreDecomposition          k=1..4
    //
    //   INSTANCE 3 - helpfulness >= 0.8 on products with >= 100 reviews  (QUALITY dice)
    //   14  helpfulness>=0.80 on productReviews>=100       1b egoNetworkAggregation       top 5
    //   15  helpfulness>=0.80 on productReviews>=100       2a shortestPathMetrics         3 landmarks
    //   16  helpfulness>=0.80 on productReviews>=100       3a localClusteringCoefficient
    //
    //   INSTANCE 4 - the whole graph, no dicing  (APEX cuboid)
    //   17  all                                            1c ratingSummary               min 500
    //   18  all                                            3c pageRankAggregation         5 iters
    //
    // Why 18 and not 36: running all nine on all four cuboids would repeat the same
    // measurement four times. Completeness is shown once, on INSTANCE 1; each other
    // instance carries only the queries whose result the dice actually changes. The
    // exhaustive size sweep is a separate job - PerformanceExperiment, 60 queries.
    public static void main(String[] args) {
        String source = args.length > 0 ? args[0] : "memgraph";
        String boltUri = args.length > 1 ? args[1] : "bolt://localhost:7687";

        SparkSession spark = SparkSession.builder()
                .appName("Graph OLAP - Demo Input Instances")
                .getOrCreate();
        spark.sparkContext().setLogLevel("WARN");

        GraphFrame full = "memgraph".equalsIgnoreCase(source)
                ? GraphReader.fromMemgraph(spark, boltUri, Integer.MAX_VALUE)
                : GraphReader.fromCsv(spark, source);

        System.out.println("\n" + "=".repeat(78));
        System.out.println(" GRAPH OLAP FRAMEWORK - INPUT INSTANCES");
        System.out.println(" Amazon Fine Foods Reviews, bipartite User -[REVIEWED]-> Product");
        System.out.printf(" source: %s%n", source);
        System.out.printf(" full graph: %d vertices, %d edges%n",
                full.vertices().count(), full.edges().count());
        System.out.println("=".repeat(78));

        List<QueryResult> results = new ArrayList<>();

        // ---------------------------------------------------------------- //
        // INSTANCE 1 - a small, dense cuboid: heavily-reviewed products only
        // ---------------------------------------------------------------- //
        banner("INSTANCE 1", "products with >= 300 reviews (small dense cuboid)");
        OlapCuboid dense = OlapCuboid.byProductPopularity(full, 300);
        System.out.printf("cuboid [%s]: %d vertices, %d edges%n",
                dense.description(), dense.vertexCount(), dense.edgeCount());

        results.add(NeighborhoodQueries.multiHopDegreeDistribution(dense, 2));
        results.add(NeighborhoodQueries.egoNetworkAggregation(dense, 5));
        results.add(NeighborhoodQueries.ratingSummary(dense, 300));
        results.add(PathQueries.shortestPathMetrics(dense, 3));
        results.add(PathQueries.reachabilityCounts(dense, 3, 3));
        results.add(PathQueries.pathLengthDistribution(dense, 5));
        results.add(CentralityQueries.localClusteringCoefficient(dense));
        results.add(CentralityQueries.kCoreDecomposition(dense, 5));
        results.add(CentralityQueries.pageRankAggregation(dense, 10));

        // ---------------------------------------------------------------- //
        // INSTANCE 2 - dice on the rating dimension: 5-star reviews only
        // ---------------------------------------------------------------- //
        banner("INSTANCE 2", "5-star reviews on products with >= 100 reviews (rating dice)");
        OlapCuboid fiveStar = OlapCuboid.byScore(
                OlapCuboid.byProductPopularity(full, 100).graph(), 5.0, 5.0);
        System.out.printf("cuboid [%s]: %d vertices, %d edges%n",
                fiveStar.description(), fiveStar.vertexCount(), fiveStar.edgeCount());

        results.add(NeighborhoodQueries.multiHopDegreeDistribution(fiveStar, 2));
        results.add(NeighborhoodQueries.ratingSummary(fiveStar, 50));
        results.add(CentralityQueries.pageRankAggregation(fiveStar, 10));
        results.add(CentralityQueries.kCoreDecomposition(fiveStar, 4));

        // ---------------------------------------------------------------- //
        // INSTANCE 3 - dice on review quality: helpful reviews only
        // ---------------------------------------------------------------- //
        banner("INSTANCE 3", "helpfulness >= 0.8 on products with >= 100 reviews (quality dice)");
        OlapCuboid helpful = OlapCuboid.byHelpfulness(
                OlapCuboid.byProductPopularity(full, 100).graph(), 0.8);
        System.out.printf("cuboid [%s]: %d vertices, %d edges%n",
                helpful.description(), helpful.vertexCount(), helpful.edgeCount());

        results.add(NeighborhoodQueries.egoNetworkAggregation(helpful, 5));
        results.add(PathQueries.shortestPathMetrics(helpful, 3));
        results.add(CentralityQueries.localClusteringCoefficient(helpful));

        // ---------------------------------------------------------------- //
        // INSTANCE 4 - the apex cuboid: the entire graph, no dicing
        // ---------------------------------------------------------------- //
        banner("INSTANCE 4", "the full graph, no dicing (apex cuboid)");
        OlapCuboid all = OlapCuboid.all(full);
        System.out.printf("cuboid [%s]: %d vertices, %d edges%n",
                all.description(), all.vertexCount(), all.edgeCount());

        results.add(NeighborhoodQueries.ratingSummary(all, 500));
        results.add(CentralityQueries.pageRankAggregation(all, 5));

        // ---------------------------------------------------------------- //
        // SUMMARY
        // ---------------------------------------------------------------- //
        System.out.println("\n" + "=".repeat(78));
        System.out.println(" SUMMARY - every query evaluated");
        System.out.println("=".repeat(78));
        for (QueryResult r : results) System.out.println(r);
        System.out.printf("%n%d queries evaluated across 3 classes and 4 input instances.%n",
                results.size());

        spark.stop();
    }

    // just the section header printed between the four instances.
    private static void banner(String tag, String what) {
        System.out.printf("%n%n%s%n %s - %s%n%s%n", "#".repeat(78), tag, what, "#".repeat(78));
    }
}
