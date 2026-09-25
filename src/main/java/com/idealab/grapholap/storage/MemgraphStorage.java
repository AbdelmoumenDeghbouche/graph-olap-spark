package com.idealab.grapholap.storage;

import org.neo4j.driver.AuthTokens;
import org.neo4j.driver.Driver;
import org.neo4j.driver.GraphDatabase;
import org.neo4j.driver.Session;
import org.neo4j.driver.Result;
import org.neo4j.driver.Record;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Memgraph as the PERSISTENT STORAGE LAYER only.
 *
 * Contract (exercise NOTE): Memgraph stores the bipartite graph and hands raw
 * vertices/edges back to Spark. It is NOT used as the aggregation query engine —
 * every OLAP metric is computed in Spark GraphX / GraphFrames.
 *
 * Therefore the Cypher used here is restricted to:
 *   - CREATE / MERGE ....... write vertices and edges
 *   - MATCH (n) RETURN n ... full scan to hand rows to Spark
 *   - index + count/health . bookkeeping
 * No aggregation (no collect(), avg(), degree(), shortestPath(), pageRank()) is
 * ever delegated to Memgraph.
 */
public class MemgraphStorage implements AutoCloseable {

    private final Driver driver;

    public MemgraphStorage(String boltUri) {
        // Memgraph's default config is auth-less; AuthTokens.none() matches it.
        this.driver = GraphDatabase.driver(boltUri, AuthTokens.none());
    }

    public MemgraphStorage() {
        this("bolt://localhost:7687");
    }

    /** Fail fast with a clear message if the storage layer is not reachable. */
    public void verifyConnectivity() {
        driver.verifyConnectivity();
    }

    /** Drop everything so a re-run starts from a known state. */
    public void clear() {
        try (Session s = driver.session()) {
            s.run("MATCH (n) DETACH DELETE n").consume();
        }
    }

    /** Indexes on the id property — makes the MERGE-heavy load path usable. */
    public void createIndexes() {
        try (Session s = driver.session()) {
            s.run("CREATE INDEX ON :User(id)").consume();
            s.run("CREATE INDEX ON :Product(id)").consume();
        }
    }

    /**
     * Bulk-load one batch of bipartite edges: (User)-[:REVIEWED]->(Product).
     * UNWIND + MERGE keeps it to a single round trip per batch.
     */
    public void writeEdgeBatch(List<Map<String, Object>> rows) {
        try (Session s = driver.session()) {
            s.executeWrite(tx -> {
                tx.run("""
                       UNWIND $rows AS row
                       MERGE (u:User {id: row.src})
                       MERGE (p:Product {id: row.dst})
                       CREATE (u)-[:REVIEWED {score: row.score,
                                              helpfulness: row.helpfulness,
                                              timestamp: row.timestamp}]->(p)
                       """,
                       Map.of("rows", rows)).consume();
                return null;
            });
        }
    }

    /** Vertex count per label — bookkeeping for the report, not an OLAP measure. */
    public long countNodes() {
        try (Session s = driver.session()) {
            return s.run("MATCH (n) RETURN count(n) AS c").single().get("c").asLong();
        }
    }

    public long countEdges() {
        try (Session s = driver.session()) {
            return s.run("MATCH ()-[r:REVIEWED]->() RETURN count(r) AS c")
                    .single().get("c").asLong();
        }
    }

    /**
     * Read the whole edge set back out of storage as plain triples.
     * This is the ONLY read path used by the OLAP layer: raw rows in, Spark does
     * the rest. Returns [srcId, dstId, score, helpfulness, timestamp].
     */
    public List<Object[]> readAllEdges(int limit) {
        List<Object[]> out = new ArrayList<>();
        try (Session s = driver.session()) {
            Result r = s.run("""
                            MATCH (u:User)-[e:REVIEWED]->(p:Product)
                            RETURN u.id AS src, p.id AS dst, e.score AS score,
                                   e.helpfulness AS helpfulness, e.timestamp AS ts
                            LIMIT $lim
                            """,
                            Map.of("lim", limit));
            while (r.hasNext()) {
                Record rec = r.next();
                out.add(new Object[]{
                        rec.get("src").asString(),
                        rec.get("dst").asString(),
                        rec.get("score").asDouble(0.0),
                        rec.get("helpfulness").asDouble(0.0),
                        rec.get("ts").asLong(0L)
                });
            }
        }
        return out;
    }

    @Override
    public void close() {
        driver.close();
    }
}
