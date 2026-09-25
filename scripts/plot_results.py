#!/usr/bin/env python3
"""
Performance plots for the experimental analysis.

X axis = number of graph vertices/edges involved by the query
Y axis = time necessary to evaluate the query
Points are ordered by increasing vertex count, as the exercise requires.

usage: plot_results.py experiment.csv outdir/
"""
import csv
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_STYLE = {
    "neighborhood": ("#1f77b4", "o", "Class 1 - Neighborhood & Degree"),
    "path":         ("#d62728", "s", "Class 2 - Path & Distance"),
    "centrality":   ("#2ca02c", "^", "Class 3 - Centrality & Density"),
}


def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "cls": r["query_class"],
                "name": r["query_name"],
                "v": int(r["vertices_involved"]),
                "e": int(r["edges_involved"]),
                "ms": int(r["elapsed_ms"]),
            })
    rows.sort(key=lambda x: x["v"])          # increasing-in-number-of-nodes
    return rows


def scatter_by_class(rows, xkey, xlabel, title, out):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r["cls"]].append(r)

    for cls, pts in by_cls.items():
        colour, marker, label = CLASS_STYLE.get(cls, ("#666", "x", cls))
        pts.sort(key=lambda x: x[xkey])
        ax.plot([p[xkey] for p in pts], [p["ms"] for p in pts],
                marker=marker, color=colour, label=label,
                linestyle="-", linewidth=1.1, markersize=6, alpha=0.85)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Query evaluation time (ms)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def scatter_loglog(rows, out):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r["cls"]].append(r)
    for cls, pts in by_cls.items():
        colour, marker, label = CLASS_STYLE.get(cls, ("#666", "x", cls))
        ax.scatter([p["v"] for p in pts], [max(p["ms"], 1) for p in pts],
                   color=colour, marker=marker, label=label, s=45, alpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Vertices involved by the query (log scale)", fontsize=11)
    ax.set_ylabel("Query evaluation time, ms (log scale)", fontsize=11)
    ax.set_title("OLAP graph query cost vs subgraph size (log-log)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both", linestyle="--")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def mean_per_query(rows, out):
    """Mean evaluation time per individual query, grouped by class."""
    agg = defaultdict(lambda: {"ms": 0, "n": 0, "cls": ""})
    for r in rows:
        k = r["name"]
        agg[k]["ms"] += r["ms"]
        agg[k]["n"] += 1
        agg[k]["cls"] = r["cls"]

    items = sorted(agg.items(), key=lambda kv: (kv[1]["cls"], kv[1]["ms"] / kv[1]["n"]))
    names = [k for k, _ in items]
    means = [v["ms"] / v["n"] for _, v in items]
    colours = [CLASS_STYLE.get(v["cls"], ("#666",))[0] for _, v in items]

    fig, ax = plt.subplots(figsize=(11, max(5, 0.42 * len(names))))
    ax.barh(range(len(names)), means, color=colours, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Mean evaluation time (ms)", fontsize=11)
    ax.set_title("Mean cost per OLAP query type", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x", linestyle="--")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for c, _, _ in CLASS_STYLE.values()]
    ax.legend(handles, [l for _, _, l in CLASS_STYLE.values()], fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: plot_results.py <experiment.csv> <outdir>")
    csv_path, outdir = sys.argv[1], sys.argv[2].rstrip("/")
    data = load(csv_path)
    if not data:
        sys.exit("no rows in " + csv_path)
    print(f"loaded {len(data)} query measurements")

    scatter_by_class(data, "v", "Vertices involved by the query",
                     "OLAP graph query evaluation time vs number of vertices",
                     f"{outdir}/perf_vertices.png")
    scatter_by_class(data, "e", "Edges involved by the query",
                     "OLAP graph query evaluation time vs number of edges",
                     f"{outdir}/perf_edges.png")
    scatter_loglog(data, f"{outdir}/perf_loglog.png")
    mean_per_query(data, f"{outdir}/perf_mean_per_query.png")
