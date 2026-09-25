package com.idealab.grapholap.experiment;

import com.idealab.grapholap.model.OlapCuboid;
import com.idealab.grapholap.model.QueryResult;
import com.idealab.grapholap.queries.CentralityQueries;
import com.idealab.grapholap.queries.NeighborhoodQueries;
import com.idealab.grapholap.queries.PathQueries;
import com.idealab.grapholap.storage.GraphReader;
import org.apache.spark.sql.SparkSession;
import org.graphframes.GraphFrame;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

/**
 * EXPERIMENTAL ANALYSIS.
 *
 * Runs a collection of RANDOM OLAP graph queries and records, for each:
 *   X axis -> number of vertices / edges involved by the query
 *   Y axis -> time necessary to evaluate the query
 * Results are sorted by increasing number of vertices, as the exercise requires.
 *
 * Randomisation has two axes:
 *   1. the cuboid    - a random popularity threshold / rating dice / sample
 *   2. the query     - a random pick from the nine implemented queries
 * so the sweep covers subgraphs from a few thousand to over a hundred thousand
 * vertices, across all three query classes.
 *
 * Usage:
 *   spark-submit --class com.idealab.grapholap.experiment.PerformanceExperiment \
 *       graph-olap.jar <edgesCsv> <nQueries> <outCsv> [seed]
 */
public class PerformanceExperiment {

    // the required experiment: N random queries on random slices, each one timed,
    // sorted by vertex count and written to the CSV the plots are built from.
    public static void main(String[] args) {
        if (args.length < 3) {
            System.err.println("usage: PerformanceExperiment <edgesCsv> <nQueries> <outCsv> [seed]");
            System.exit(1);
        }
        String edgesCsv = args[0];
        int nQueries = Integer.parseInt(args[1]);
        String outCsv = args[2];
        long seed = args.length > 3 ? Long.parseLong(args[3]) : 42L;

        SparkSession spark = SparkSession.builder()
                .appName("Graph OLAP - Performance Experiment")
                .getOrCreate();
        spark.sparkContext().setLogLevel("WARN");

        GraphFrame full = GraphReader.fromCsv(spark, edgesCsv);
        long fullV = full.vertices().count();
        long fullE = full.edges().count();
        System.out.printf("[experiment] full graph: %d vertices, %d edges%n", fullV, fullE);
        System.out.printf("[experiment] running %d random OLAP queries (seed=%d)%n", nQueries, seed);

        Random rnd = new Random(seed);
        List<QueryResult> results = new ArrayList<>();

        // popularity thresholds chosen to spread subgraph size as widely as the
        // dataset allows — measured span 3,764 to 135,420 vertices, a factor of 36.
        // This is what spreads the points along the X axis.
        int[] thresholds = {500, 400, 300, 250, 200, 150, 120, 100, 80, 60, 50, 40, 30, 25, 20};

        for (int i = 0; i < nQueries; i++) {
            // [0] = the popularity base, [1] = the effective cuboid (may be the
            // same object). Both are tracked so both can be released afterwards.
            OlapCuboid[] pair = randomCuboid(full, rnd, thresholds);
            OlapCuboid base = pair[0];
            OlapCuboid cuboid = pair[1];
            long v = cuboid.vertexCount();
            long e = cuboid.edgeCount();
            if (v == 0 || e == 0) {
                cuboid.release();
                if (base != cuboid) base.release();
                continue;
            }

            int pick = rnd.nextInt(9);
            System.out.printf("%n[experiment] query %d/%d  cuboid=[%s] V=%d E=%d  pick=%d%n",
                    i + 1, nQueries, cuboid.description(), v, e, pick);

            QueryResult r;
            try {
                r = switch (pick) {
                    // class 1 - neighborhood & degree
                    case 0 -> NeighborhoodQueries.multiHopDegreeDistribution(cuboid, 1 + rnd.nextInt(2));
                    case 1 -> NeighborhoodQueries.egoNetworkAggregation(cuboid, 3 + rnd.nextInt(5));
                    case 2 -> NeighborhoodQueries.ratingSummary(cuboid, 1 + rnd.nextInt(50));
                    // class 2 - path & distance
                    case 3 -> PathQueries.shortestPathMetrics(cuboid, 1 + rnd.nextInt(3));
                    case 4 -> PathQueries.reachabilityCounts(cuboid, 1 + rnd.nextInt(3), 1 + rnd.nextInt(3));
                    case 5 -> PathQueries.pathLengthDistribution(cuboid, 2 + rnd.nextInt(5));
                    // class 3 - centrality & density
                    case 6 -> CentralityQueries.localClusteringCoefficient(cuboid);
                    case 7 -> CentralityQueries.kCoreDecomposition(cuboid, 2 + rnd.nextInt(3));
                    default -> CentralityQueries.pageRankAggregation(cuboid, 5 + rnd.nextInt(10));
                };
            } catch (Throwable ex) {
                // a failed query must not kill a 60-query sweep; log and continue.
                // Throwable, not Exception: an OutOfMemoryError in one query must
                // not silently abort the whole sweep with no record of why.
                System.err.printf("[experiment] query %d FAILED (%s): %s%n",
                        i + 1, ex.getClass().getSimpleName(), ex.getMessage());
                continue;
            } finally {
                // release regardless of outcome, so query i+1 is timed on a clean
                // cache instead of inheriting query i's storage pressure
                cuboid.release();
                if (base != cuboid) base.release();
            }
            results.add(r);
            System.out.printf("[experiment] -> %s%n", r);
        }

        // sort by increasing number of vertices involved, as required
        results.sort(Comparator.comparingLong(a -> a.verticesInvolved));

        try (PrintWriter pw = new PrintWriter(outCsv)) {
            pw.println(QueryResult.csvHeader());
            for (QueryResult r : results) pw.println(r.toCsv());
        } catch (Exception ex) {
            System.err.println("[experiment] could not write " + outCsv + ": " + ex.getMessage());
        }

        System.out.println("\n" + "=".repeat(78));
        System.out.println(" EXPERIMENT RESULTS (sorted by increasing vertex count)");
        System.out.println("=".repeat(78));
        for (QueryResult r : results) System.out.println(r);
        System.out.printf("%n%d queries completed, written to %s%n", results.size(), outCsv);

        spark.stop();
    }

    /**
     * A random OLAP cuboid: random dicing dimension, random selectivity.
     *
     * In one line: this is where the randomness lives - a random popularity
     * threshold, and half the time a second dice on top, so density varies
     * independently of size.
     *
     * Returns {base, cuboid}. When a second dice is applied the base is a
     * distinct intermediate that also holds cached data, so the caller needs
     * both handles to release everything.
     */
    private static OlapCuboid[] randomCuboid(GraphFrame full, Random rnd, int[] thresholds) {
        int t = thresholds[rnd.nextInt(thresholds.length)];
        OlapCuboid base = OlapCuboid.byProductPopularity(full, t);

        // 50% of the time apply a second dice, to vary density independently of size
        int extra = rnd.nextInt(4);
        OlapCuboid diced = switch (extra) {
            case 0 -> OlapCuboid.byScore(base.graph(), 4.0, 5.0);
            case 1 -> OlapCuboid.byHelpfulness(base.graph(), 0.5);
            default -> base;
        };
        return new OlapCuboid[]{base, diced};
    }
}
