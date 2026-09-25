package com.idealab.grapholap.model;

/**
 * One measured OLAP query evaluation.
 *
 * This is the unit the experimental analysis plots:
 *   X axis = verticesInvolved / edgesInvolved
 *   Y axis = elapsedMs
 */
public class QueryResult {

    public final String queryClass;   // "neighborhood" | "path" | "centrality"
    public final String queryName;    // e.g. "multiHopDegreeDistribution"
    public final long verticesInvolved;
    public final long edgesInvolved;
    public final long elapsedMs;
    public final long resultRows;
    public final String params;       // the OLAP selection that defined the subgraph

    // every query fills exactly this object: which query, on how much graph, how
    // long it took, how many rows came out, and which slice it ran on.
    public QueryResult(String queryClass, String queryName, long verticesInvolved,
                       long edgesInvolved, long elapsedMs, long resultRows, String params) {
        this.queryClass = queryClass;
        this.queryName = queryName;
        this.verticesInvolved = verticesInvolved;
        this.edgesInvolved = edgesInvolved;
        this.elapsedMs = elapsedMs;
        this.resultRows = resultRows;
        this.params = params;
    }

    // the header line of the experiment CSV.
    public static String csvHeader() {
        return "query_class,query_name,vertices_involved,edges_involved,elapsed_ms,result_rows,params";
    }

    // one measurement as one CSV row - this is what the plotting script reads.
    public String toCsv() {
        // params is quoted: it contains commas (e.g. "minScore=4.0,hops=2")
        return String.format("%s,%s,%d,%d,%d,%d,\"%s\"",
                queryClass, queryName, verticesInvolved, edgesInvolved,
                elapsedMs, resultRows, params);
    }

    @Override
    // the aligned one-line form you see in the live console output.
    public String toString() {
        return String.format("[%-12s] %-32s V=%-8d E=%-9d %6d ms  rows=%-8d %s",
                queryClass, queryName, verticesInvolved, edgesInvolved,
                elapsedMs, resultRows, params);
    }
}
