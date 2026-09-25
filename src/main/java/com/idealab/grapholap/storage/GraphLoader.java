package com.idealab.grapholap.storage;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Stage 3 of the pipeline: HDFS (CSV) -> Memgraph (persistent graph storage).
 *
 *   raw reviews file -> HDFS raw     (Hadoop = distributed landing zone)
 *   CSV       -> Memgraph            (this class; Memgraph = graph persistence)
 *   Memgraph  -> Spark GraphFrames   (GraphReader; Spark = the OLAP engine)
 *
 * Usage:
 *   spark-submit --class com.idealab.grapholap.storage.GraphLoader graph-olap.jar \
 *       hdfs://localhost:9000/user/finefoods/edges.csv  bolt://localhost:7687  [maxEdges]
 */
public class GraphLoader {

    /** Bolt round trips are the bottleneck; batch to keep them few and large. */
    private static final int BATCH_SIZE = 10_000;

    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("usage: GraphLoader <edgesCsvPath> [boltUri] [maxEdges]");
            System.exit(1);
        }
        String edgesPath = args[0];
        String boltUri = args.length > 1 ? args[1] : "bolt://localhost:7687";
        long maxEdges = args.length > 2 ? Long.parseLong(args[2]) : -1L;

        SparkSession spark = SparkSession.builder()
                .appName("GraphLoader: HDFS -> Memgraph")
                .getOrCreate();
        spark.sparkContext().setLogLevel("WARN");

        Dataset<Row> edges = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(edgesPath);
        if (maxEdges > 0) {
            edges = edges.limit((int) maxEdges);
        }

        long total = edges.count();
        System.out.printf("[loader] edges to persist: %d%n", total);

        try (MemgraphStorage store = new MemgraphStorage(boltUri)) {
            store.verifyConnectivity();
            System.out.println("[loader] Memgraph reachable, clearing + indexing");
            store.clear();
            store.createIndexes();

            // toLocalIterator streams partition-by-partition, so the driver never
            // holds all 568k rows at once.
            Iterator<Row> it = edges.toLocalIterator();
            List<Map<String, Object>> batch = new ArrayList<>(BATCH_SIZE);
            long written = 0;
            long t0 = System.currentTimeMillis();

            while (it.hasNext()) {
                Row r = it.next();
                batch.add(Map.of(
                        "src", r.getAs("src").toString(),
                        "dst", r.getAs("dst").toString(),
                        "score", toDouble(r.getAs("score")),
                        "helpfulness", toDouble(r.getAs("helpfulness")),
                        "timestamp", toLong(r.getAs("timestamp"))
                ));
                if (batch.size() >= BATCH_SIZE) {
                    store.writeEdgeBatch(batch);
                    written += batch.size();
                    batch.clear();
                    System.out.printf("[loader] %d / %d edges persisted%n", written, total);
                }
            }
            if (!batch.isEmpty()) {
                store.writeEdgeBatch(batch);
                written += batch.size();
            }

            long secs = (System.currentTimeMillis() - t0) / 1000;
            System.out.printf("[loader] DONE %d edges in %ds%n", written, secs);
            System.out.printf("[loader] Memgraph now holds: %d nodes, %d edges%n",
                    store.countNodes(), store.countEdges());
        }
        spark.stop();
    }

    private static double toDouble(Object o) {
        if (o == null) return 0.0;
        if (o instanceof Number n) return n.doubleValue();
        try { return Double.parseDouble(o.toString()); } catch (NumberFormatException e) { return 0.0; }
    }

    private static long toLong(Object o) {
        if (o == null) return 0L;
        if (o instanceof Number n) return n.longValue();
        try { return Long.parseLong(o.toString()); } catch (NumberFormatException e) { return 0L; }
    }
}
