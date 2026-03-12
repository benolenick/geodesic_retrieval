#!/usr/bin/env python3
"""
Shape Matcher — compares attack path shapes using Dynamic Time Warping.

An attack path is a curve through 7-dimensional structure space. Two paths
with different techniques but similar structural progressions (e.g., both
start with enumeration, escalate privileges gradually, then jump to admin)
have similar shapes.

DTW measures similarity between two sequences that may vary in speed/length.

Math applied:
  - Dynamic Time Warping (Sakoe & Chiba 1978)
  - Path signature comparison
  - Shape-guided fact retrieval from matched known chains
"""

import json
import logging
import os

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("shape_matcher")

KNOWN_CHAINS_PATH = os.environ.get("KNOWN_CHAINS_PATH", "known_chains.json")


def dtw_distance(path_a: np.ndarray, path_b: np.ndarray) -> float:
    """
    Dynamic Time Warping distance between two paths in structure space.

    Each path is (n_steps, 7) array of structure vectors.
    Returns normalized DTW distance (lower = more similar).
    """
    n = len(path_a)
    m = len(path_b)

    # Cost matrix
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Euclidean distance between structure vectors
            d = np.linalg.norm(path_a[i - 1] - path_b[j - 1])
            cost[i, j] = d + min(
                cost[i - 1, j],      # insertion (stretch path_a)
                cost[i, j - 1],      # deletion (stretch path_b)
                cost[i - 1, j - 1],  # match
            )

    # Normalize by path length
    return cost[n, m] / max(n, m)


def dtw_alignment(path_a: np.ndarray, path_b: np.ndarray) -> list[tuple[int, int]]:
    """
    Compute DTW alignment path (which steps in path_a map to which in path_b).
    Returns list of (i, j) index pairs.
    """
    n = len(path_a)
    m = len(path_b)

    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = np.linalg.norm(path_a[i - 1] - path_b[j - 1])
            cost[i, j] = d + min(
                cost[i - 1, j],
                cost[i, j - 1],
                cost[i - 1, j - 1],
            )

    # Traceback
    alignment = []
    i, j = n, m
    while i > 0 and j > 0:
        alignment.append((i - 1, j - 1))
        candidates = [
            (cost[i - 1, j - 1], i - 1, j - 1),
            (cost[i - 1, j], i - 1, j),
            (cost[i, j - 1], i, j - 1),
        ]
        _, i, j = min(candidates)
    alignment.reverse()
    return alignment


def path_shape_descriptor(path: np.ndarray) -> np.ndarray:
    """
    Compute a compact shape descriptor for a path.

    The descriptor captures the trajectory's delta patterns:
    - Mean delta per dimension (overall direction)
    - Std delta per dimension (smoothness vs jumpiness)
    - Max delta per dimension (biggest jump)
    - Total path length in structure space

    Returns a 22-dimensional descriptor vector.
    """
    if len(path) < 2:
        return np.zeros(22)

    deltas = np.diff(path, axis=0)  # (n-1, 7) per-step changes

    mean_delta = deltas.mean(axis=0)      # 7 dims: average change direction
    std_delta = deltas.std(axis=0)        # 7 dims: how smooth/jumpy
    max_delta = np.abs(deltas).max(axis=0) # 7 dims: biggest single jump

    # Total path length (sum of step distances)
    step_lengths = np.linalg.norm(deltas, axis=1)
    total_length = step_lengths.sum()

    return np.concatenate([mean_delta, std_delta, max_delta, [total_length]])


class ShapeMatcher:
    """Match partial attack paths against known chain shapes."""

    def __init__(self, chains_path: str = KNOWN_CHAINS_PATH):
        self.chains = []
        self.chain_paths = []       # np arrays of structure vectors
        self.chain_descriptors = []  # compact shape descriptors
        self._load_chains(chains_path)

    def _load_chains(self, path: str):
        """Load known attack chains and precompute their shapes."""
        if not os.path.exists(path):
            log.warning(f"Known chains file not found: {path}")
            return

        with open(path, "r") as f:
            self.chains = json.load(f)

        for chain in self.chains:
            steps = chain.get("steps", [])
            if not steps:
                continue
            path_array = np.array([s["structure"] for s in steps])
            self.chain_paths.append(path_array)
            self.chain_descriptors.append(path_shape_descriptor(path_array))

        log.info(f"Loaded {len(self.chains)} known chains with shape descriptors")

    def match_shape(self, partial_path: np.ndarray, top_k: int = 3) -> list[dict]:
        """
        Given a partial path (sequence of structure vectors),
        find the top-k known chains with the most similar shape.

        Uses DTW for accurate matching.

        Args:
            partial_path: (n_steps, 7) array of structure vectors
            top_k: number of matches to return

        Returns:
            List of {chain_name, dtw_distance, alignment, remaining_steps}
        """
        if len(partial_path) < 2 or not self.chain_paths:
            return []

        matches = []
        for i, (chain, known_path) in enumerate(zip(self.chains, self.chain_paths)):
            dist = dtw_distance(partial_path, known_path)
            alignment = dtw_alignment(partial_path, known_path)

            # Figure out which steps in the known chain are NOT covered
            # by the alignment (= remaining steps the hiker needs)
            covered_known = set(j for _, j in alignment)
            remaining = [
                {
                    "step_idx": j,
                    "query": chain["steps"][j]["query"],
                    "structure": chain["steps"][j]["structure"],
                }
                for j in range(len(known_path))
                if j not in covered_known
            ]

            matches.append({
                "chain_name": chain["name"],
                "chain_idx": i,
                "dtw_distance": round(dist, 4),
                "alignment_length": len(alignment),
                "known_path_length": len(known_path),
                "remaining_steps": remaining,
                "coverage": round(len(covered_known) / len(known_path), 2),
            })

        matches.sort(key=lambda x: x["dtw_distance"])
        return matches[:top_k]

    def fast_match(self, partial_path: np.ndarray, top_k: int = 3) -> list[dict]:
        """
        Fast approximate shape matching using shape descriptors.

        Compares compact 22-dim descriptors using cosine distance.
        Much faster than full DTW for initial filtering.
        """
        if len(partial_path) < 2 or not self.chain_descriptors:
            return []

        query_desc = path_shape_descriptor(partial_path)
        q_norm = np.linalg.norm(query_desc)
        if q_norm < 1e-8:
            return []

        matches = []
        for i, (chain, desc) in enumerate(zip(self.chains, self.chain_descriptors)):
            d_norm = np.linalg.norm(desc)
            if d_norm < 1e-8:
                continue
            cos_sim = np.dot(query_desc, desc) / (q_norm * d_norm)
            matches.append({
                "chain_name": chain["name"],
                "chain_idx": i,
                "descriptor_similarity": round(float(cos_sim), 4),
            })

        matches.sort(key=lambda x: x["descriptor_similarity"], reverse=True)
        return matches[:top_k]

    def get_chain_queries(self, chain_idx: int) -> list[str]:
        """Get all search queries from a known chain."""
        if 0 <= chain_idx < len(self.chains):
            return [s["query"] for s in self.chains[chain_idx]["steps"]]
        return []

    def shape_guided_retrieval_queries(
        self,
        partial_structure: list[list[float]],
        top_k_chains: int = 3,
    ) -> list[str]:
        """
        Given a partial path's structure vectors, find matching known chains
        and return search queries for the REMAINING steps.

        This is the "maps in the fog" — queries to find facts the hiker
        hasn't reached yet but needs.
        """
        if not partial_structure or not self.chain_paths:
            return []

        partial = np.array(partial_structure)
        matches = self.match_shape(partial, top_k=top_k_chains)

        queries = []
        seen = set()
        for match in matches:
            for step in match["remaining_steps"]:
                q = step["query"]
                if q not in seen:
                    seen.add(q)
                    queries.append(q)

        return queries
