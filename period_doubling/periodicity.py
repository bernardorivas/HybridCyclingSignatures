"""
Periodicity detection for period-doubling cascades.

Cluster-based detection: period-k orbits have k distinct recurrent states;
chaos has a near-continuum.
"""

import numpy as np


def _gap_clusters(sorted_values, gap):
    """Split an ascending 1-d array into clusters at consecutive gaps > gap."""
    clusters = []
    current = [sorted_values[0]]
    for i in range(1, len(sorted_values)):
        if sorted_values[i] - sorted_values[i - 1] > gap:
            clusters.append(current)
            current = [sorted_values[i]]
        else:
            current.append(sorted_values[i])
    clusters.append(current)
    return clusters


def detect_roessler_period(ts, peak_gap=0.05):
    """Detect period of a Roessler trajectory from x-component local maxima.

    Parameters
    ----------
    ts : Timeseries
        Must have .t (N,), .x (N, 3), .meta dict.
    peak_gap : float
        Threshold for clustering adjacent refined peak values.

    Returns
    -------
    dict
        n_clusters : int
            Number of distinct peak-value clusters.
        peak_values : np.ndarray
            Sorted refined peak values across all peaks.
        cluster_values : list
            Mean value of each cluster.
        mean_return_time : float
            Mean time between consecutive peak detections.
    """
    x_series = ts.x[:, 0]  # First component
    n = len(x_series)

    # Find strict local maxima (interior points where x[i] > x[i-1] and x[i] > x[i+1])
    peaks = []
    peak_times = []
    for i in range(1, n - 1):
        if x_series[i] > x_series[i - 1] and x_series[i] > x_series[i + 1]:
            peaks.append(i)
            peak_times.append(ts.t[i])

    if len(peaks) < 4:  # Need at least 4 to drop first 3
        # Short-series fallback: cluster whatever peaks exist. Zero peaks
        # means no oscillation at all (e.g. collapse to equilibrium) and must
        # NOT validate as period-1, hence n_clusters=0.
        if len(peaks) == 0:
            return {
                "n_clusters": 0,
                "peak_values": np.array([]),
                "cluster_values": [],
                "mean_return_time": 0.0,
            }
        peak_values = np.sort(np.array([x_series[p] for p in peaks]))
        clusters = _gap_clusters(peak_values, peak_gap)
        return {
            "n_clusters": len(clusters),
            "peak_values": peak_values,
            "cluster_values": [np.mean(c) for c in clusters],
            "mean_return_time": np.mean(np.diff(peak_times)) if len(peak_times) > 1 else 0.0,
        }

    # Refine each peak by quadratic (parabolic) interpolation through 3 points
    refined_peaks = []
    for idx in peaks:
        # Three points: (i-1, y[i-1]), (i, y[i]), (i+1, y[i+1])
        if idx > 0 and idx < n - 1:
            y0, y1, y2 = x_series[idx - 1], x_series[idx], x_series[idx + 1]
            # Parabola through 3 equally-spaced points: refined peak is at y_refined
            # Using discrete Lagrange form: the maximum of a parabola through
            # (-1, y0), (0, y1), (1, y2) is at offset = (y0 - y2) / (2*(y0 - 2*y1 + y2))
            denom = 2.0 * (y0 - 2.0 * y1 + y2)
            if abs(denom) > 1e-14:
                offset = (y0 - y2) / denom
                # Clamp offset to [-0.5, 0.5] (peak must be near the discrete max)
                offset = np.clip(offset, -0.5, 0.5)
            else:
                offset = 0.0
            # Vertex value of the parabola: f(x*) = y1 - (y0 - y2)^2 / (8*(y0 - 2*y1 + y2))
            # = y1 - 0.25*(y0 - y2)*offset
            y_refined = y1 - 0.25 * (y0 - y2) * offset
            refined_peaks.append(y_refined)

    # Drop first 3 peaks (transient)
    refined_peaks = refined_peaks[3:]
    peak_times = peak_times[3:]

    if len(refined_peaks) == 0:
        return {
            "n_clusters": 0,
            "peak_values": np.array([]),
            "cluster_values": [],
            "mean_return_time": 0.0,
        }

    # Sort refined peaks
    refined_peaks = np.array(refined_peaks)
    sorted_indices = np.argsort(refined_peaks)
    peak_values = refined_peaks[sorted_indices]
    peak_times_sorted = np.array(peak_times)[sorted_indices]

    # Cluster by gaps > peak_gap
    clusters = _gap_clusters(peak_values, peak_gap)

    # Cluster values (means)
    cluster_values = [np.mean(c) for c in clusters]
    n_clusters = len(clusters)

    # Mean return time: mean of consecutive peak-time differences
    mean_return_time = np.mean(np.diff(peak_times)) if len(peak_times) > 1 else 0.0

    return {
        "n_clusters": n_clusters,
        "peak_values": peak_values,
        "cluster_values": cluster_values,
        "mean_return_time": mean_return_time,
    }


def detect_compass_period(ts, link_tol=0.002):
    """Detect period of a compass-gait trajectory via single-linkage clustering.

    Parameters
    ----------
    ts : HybridTimeseries
        Must have .jump_plus (K, 4), .meta dict, and .impact_times.
    link_tol : float
        Linkage distance threshold for clustering.

    Returns
    -------
    dict
        n_clusters : int
            Number of distinct gait clusters.
        cluster_sizes : list
            Size of each cluster.
        min_intercluster_dist : float or None
            Smallest pairwise distance between cluster centroids.
        n_impacts : int
            Total number of impacts.
    """
    jump_plus = ts.jump_plus
    n_impacts = len(jump_plus)

    # Drop first 10 impacts as transient
    if n_impacts <= 10:
        # Return sensible fallback
        if n_impacts == 0:
            return {
                "n_clusters": 0,
                "cluster_sizes": [],
                "min_intercluster_dist": None,
                "n_impacts": 0,
            }
        # Cluster all remaining impacts as one cluster
        return {
            "n_clusters": 1,
            "cluster_sizes": [n_impacts],
            "min_intercluster_dist": None,
            "n_impacts": n_impacts,
        }

    jump_plus_active = jump_plus[10:]
    n_active = len(jump_plus_active)

    # Subsample if needed
    if n_active > 2000:
        # Keep last 2000
        jump_plus_active = jump_plus_active[-2000:]
        n_active = len(jump_plus_active)

    # Threshold single-linkage clustering = connected components of the graph
    # with an edge wherever the pairwise distance is <= link_tol. Vectorized
    # distance matrix + union-find keeps this fast at the 2000-point cap.
    sq = np.sum(jump_plus_active ** 2, axis=1)
    dist_sq = sq[:, None] + sq[None, :] - 2.0 * (jump_plus_active @ jump_plus_active.T)
    np.maximum(dist_sq, 0.0, out=dist_sq)

    parent = np.arange(n_active)

    def _find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    edge_i, edge_j = np.nonzero(np.triu(dist_sq <= link_tol ** 2, k=1))
    for a, b in zip(edge_i, edge_j):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    roots = np.array([_find(i) for i in range(n_active)])
    clusters = [np.nonzero(roots == r)[0].tolist() for r in np.unique(roots)]

    # Cluster sizes and centroids
    cluster_sizes = [len(c) for c in clusters]
    n_clusters = len(clusters)

    # Compute centroids
    centroids = []
    for cluster in clusters:
        centroid = np.mean(jump_plus_active[cluster], axis=0)
        centroids.append(centroid)

    # Minimum intercluster distance
    min_intercluster_dist = None
    if n_clusters > 1:
        min_intercluster_dist = np.inf
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                d = np.linalg.norm(centroids[i] - centroids[j])
                if d < min_intercluster_dist:
                    min_intercluster_dist = d

    return {
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes,
        "min_intercluster_dist": min_intercluster_dist,
        "n_impacts": n_impacts,
    }


def check_period(detected_n_clusters, expected_period, chaos_min_clusters=16):
    """Verify periodicity detection against expected period.

    Parameters
    ----------
    detected_n_clusters : int
        Number of clusters detected.
    expected_period : int or None
        Expected period (None for chaos).
    chaos_min_clusters : int
        Threshold cluster count above which to classify as chaos.

    Returns
    -------
    bool
        True if detection matches expectation.
    """
    if expected_period is None:
        # Chaos: expect many clusters
        return detected_n_clusters >= chaos_min_clusters
    else:
        # Periodic: expect exact match
        return detected_n_clusters == expected_period
