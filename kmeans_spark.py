#!/usr/bin/env python3
"""
K-Means via PySpark RDD MapReduce
===================================
Replaces the Hadoop Streaming mapper.py + reducer.py pair.

The Map and Reduce logic is preserved exactly:
  MAP    : assign each point to its nearest centroid
  REDUCE : compute new centroid = mean of all assigned points

Run standalone:
    python3 kmeans_spark.py --input data.tsv --k 5 --iters 20

Or import and call run_kmeans() from the notebook.
"""

import argparse
import math
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# MAP  (mirrors mapper.py logic)
# ─────────────────────────────────────────────────────────────────────────────
def mapper(point, centroids):
    """
    Given a data point (tuple of floats) and a list of centroid arrays,
    return (cluster_id, point) — identical output to mapper.py's
      print(f"{nearest}\\t{','.join(map(str, features))}")
    """
    min_dist  = float("inf")
    nearest   = -1
    for cluster_id, centroid in enumerate(centroids):
        dist = math.sqrt(sum((p - c) ** 2 for p, c in zip(point, centroid)))
        if dist < min_dist:
            min_dist = dist
            nearest  = cluster_id
    return (nearest, point)


# ─────────────────────────────────────────────────────────────────────────────
# REDUCE  (mirrors reducer.py logic)
# ─────────────────────────────────────────────────────────────────────────────
def reducer(cluster_id, points):
    """
    Given a cluster_id and an iterable of points assigned to it,
    return (cluster_id, new_centroid) — identical to reducer.py's
      print(f"{current_cluster}\\t{','.join(map(str, centroid))}")
    """
    point_list  = list(points)
    n           = len(point_list)
    dim         = len(point_list[0])
    centroid    = [sum(p[d] for p in point_list) / n for d in range(dim)]
    return (cluster_id, tuple(centroid))


# ─────────────────────────────────────────────────────────────────────────────
# K-Means++ initialisation
# ─────────────────────────────────────────────────────────────────────────────
def kmeans_plus_plus_init(data_rdd, k, seed=42):
    """
    K-Means++ seeding on an RDD.
    Returns a list of k centroid tuples.
    """
    rng        = np.random.default_rng(seed)
    all_points = data_rdd.collect()
    n          = len(all_points)

    first_idx  = rng.integers(0, n)
    centroids  = [all_points[first_idx]]

    for _ in range(1, k):
        dists = np.array([
            min(sum((p - c) ** 2 for p, c in zip(pt, cen))
                for cen in centroids)
            for pt in all_points
        ])
        probs      = dists / dists.sum()
        next_idx   = rng.choice(n, p=probs)
        centroids.append(all_points[next_idx])

    return [tuple(c) for c in centroids]


# ─────────────────────────────────────────────────────────────────────────────
# Main iterative MapReduce loop
# ─────────────────────────────────────────────────────────────────────────────
def run_kmeans(spark, tsv_path, k=5, max_iter=20, convergence=1e-4, seed=42):
    """
    Run K-Means using PySpark RDD map / reduceByKey.

    Parameters
    ----------
    spark       : active SparkSession
    tsv_path    : path to TSV file (tab-separated, no header, numeric features)
    k           : number of clusters
    max_iter    : maximum iterations
    convergence : stop when max centroid shift < this value
    seed        : random seed for K-Means++ init

    Returns
    -------
    assignments      : list of (point_index, cluster_id)
    final_centroids  : list of k centroid tuples
    history          : list of centroid arrays per iteration
    deltas           : list of convergence deltas per iteration
    """
    sc = spark.sparkContext

    # ── Load data as RDD of tuples ────────────────────────────────────────────
    raw_rdd = sc.textFile(tsv_path)
    data_rdd = (
        raw_rdd
        .filter(lambda line: line.strip() != "")
        .map(lambda line: tuple(map(float, line.strip().split("\t"))))
        .cache()
    )

    n_points = data_rdd.count()
    print(f"  ✓ Loaded {n_points:,} data points from {tsv_path}")

    # ── K-Means++ initialisation ──────────────────────────────────────────────
    print("  Initialising centroids with K-Means++...")
    centroids = kmeans_plus_plus_init(data_rdd, k, seed=seed)
    print(f"  ✓ {k} initial centroids ready")

    history = [list(centroids)]
    deltas  = []

    print(f"\n  Running MapReduce K-Means  (k={k}, max_iter={max_iter})")
    print("  " + "─" * 50)

    for iteration in range(1, max_iter + 1):

        # ── MAP: broadcast centroids, assign each point ───────────────────────
        bc_centroids = sc.broadcast(centroids)

        assigned_rdd = data_rdd.map(
            lambda point: mapper(point, bc_centroids.value)
        )
        # assigned_rdd: RDD of (cluster_id, point)

        # ── REDUCE: compute new centroid per cluster ──────────────────────────
        new_centroids_rdd = (
            assigned_rdd
            .groupByKey()                                   # (cluster_id, [points])
            .map(lambda kv: reducer(kv[0], kv[1]))          # (cluster_id, centroid)
            .sortByKey()
        )
        new_centroids_list = new_centroids_rdd.collect()    # [(0, c0), (1, c1), ...]
        new_centroids      = [c for _, c in sorted(new_centroids_list)]

        # ── Convergence check ─────────────────────────────────────────────────
        delta = max(
            math.sqrt(sum((a - b) ** 2 for a, b in zip(old, new)))
            for old, new in zip(centroids, new_centroids)
        )
        deltas.append(delta)
        history.append(list(new_centroids))

        print(f"  Iter {iteration:02d}  |  max centroid shift = {delta:.6f}")

        bc_centroids.unpersist()
        centroids = new_centroids

        if delta < convergence:
            print(f"\n  ✓ Converged after {iteration} iterations  (Δ = {delta:.2e})")
            break
    else:
        print(f"\n  ⚠ Reached max iterations ({max_iter}) without full convergence")

    # ── Final assignment pass ─────────────────────────────────────────────────
    bc_final = sc.broadcast(centroids)
    indexed_rdd = data_rdd.zipWithIndex()   # (point, idx)
    final_assignments = (
        indexed_rdd
        .map(lambda pi: (pi[1], mapper(pi[0], bc_final.value)[0]))   # (idx, cluster_id)
        .collect()
    )
    bc_final.unpersist()

    return final_assignments, centroids, history, deltas


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point (optional standalone usage)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PySpark K-Means MapReduce")
    parser.add_argument("--input",  required=True, help="Path to TSV data file")
    parser.add_argument("--k",      type=int, default=5,    help="Number of clusters")
    parser.add_argument("--iters",  type=int, default=20,   help="Max iterations")
    parser.add_argument("--conv",   type=float, default=1e-4, help="Convergence threshold")
    parser.add_argument("--seed",   type=int, default=42,   help="Random seed")
    args = parser.parse_args()

    from pyspark.sql import SparkSession
    spark = (
        SparkSession.builder
        .appName("PlayerKMeans")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    assignments, centroids, history, deltas = run_kmeans(
        spark, args.input, k=args.k,
        max_iter=args.iters, convergence=args.conv, seed=args.seed
    )

    print(f"\nDone. Final centroid shift history: {[round(d, 6) for d in deltas]}")
    spark.stop()
