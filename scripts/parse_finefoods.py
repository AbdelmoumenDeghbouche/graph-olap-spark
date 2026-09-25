#!/usr/bin/env python3
"""
Parse the Amazon Fine Foods Reviews file into a bipartite graph.

Input : finefoods.txt.gz  (records separated by blank lines)
Output: vertices.csv  -> id,label,name          (Users and Products)
        edges.csv     -> src,dst,score,helpfulness,timestamp

The bipartite graph is  User --reviewed--> Product  with the review score
carried on the edge (the edge weight used by the OLAP aggregations).

Record format:
    product/productId: B001E4KFG0
    review/userId: A3SGXH7AUHU8GW
    review/profileName: delmartian
    review/helpfulness: 1/1
    review/score: 5.0
    review/time: 1303862400
    review/summary: Good Quality Dog Food
    review/text: I have bought several of the Vitality canned...
"""
import csv
import gzip
import sys


def parse(path_in, vertices_out, edges_out):
    users, products = {}, {}
    n_edges = 0
    cur = {}

    def flush(rec):
        """Emit one review record as a bipartite edge."""
        nonlocal n_edges
        uid = rec.get("review/userId", "").strip()
        pid = rec.get("product/productId", "").strip()
        if not uid or not pid or uid == "unknown":
            return None
        if uid not in users:
            users[uid] = rec.get("review/profileName", "").strip()
        if pid not in products:
            products[pid] = ""
        try:
            score = float(rec.get("review/score", "0") or 0)
        except ValueError:
            score = 0.0
        # helpfulness arrives as "num/den" -> keep the ratio, guard den == 0
        h_raw = rec.get("review/helpfulness", "0/0").strip()
        try:
            num, den = h_raw.split("/")
            helpful = (float(num) / float(den)) if float(den) > 0 else 0.0
        except (ValueError, ZeroDivisionError):
            helpful = 0.0
        try:
            ts = int(rec.get("review/time", "0") or 0)
        except ValueError:
            ts = 0
        n_edges += 1
        return (f"u_{uid}", f"p_{pid}", score, round(helpful, 4), ts)

    opener = gzip.open if path_in.endswith(".gz") else open
    with opener(path_in, "rt", encoding="latin-1", errors="replace") as fh, \
         open(edges_out, "w", newline="", encoding="utf-8") as ef:
        ew = csv.writer(ef)
        ew.writerow(["src", "dst", "score", "helpfulness", "timestamp"])
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                if cur:
                    row = flush(cur)
                    if row:
                        ew.writerow(row)
                    cur = {}
                continue
            if ":" in line:
                k, _, v = line.partition(":")
                cur[k.strip()] = v.strip()
        if cur:  # final record may not be followed by a blank line
            row = flush(cur)
            if row:
                ew.writerow(row)

    with open(vertices_out, "w", newline="", encoding="utf-8") as vf:
        vw = csv.writer(vf)
        vw.writerow(["id", "label", "name"])
        for uid, name in users.items():
            vw.writerow([f"u_{uid}", "User", name])
        for pid in products:
            vw.writerow([f"p_{pid}", "Product", ""])

    print(f"users    : {len(users)}")
    print(f"products : {len(products)}")
    print(f"vertices : {len(users) + len(products)}")
    print(f"edges    : {n_edges}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: parse_finefoods.py <in.txt.gz> <vertices.csv> <edges.csv>")
    parse(sys.argv[1], sys.argv[2], sys.argv[3])
