#!/usr/bin/env python3
"""
Geodesic Engine — shortest paths on the attack surface manifold.

Computes geodesics (minimum-cost paths) between facts on the manifold graph,
finds facts in the geodesic neighborhood, and identifies tangent directions
(branching paths).

Math applied:
  - Dijkstra's algorithm with -log(weight) cost transform
  - Geodesic neighborhood: facts within radius r of the path in manifold coords
  - Tangent space: graph edges at each waypoint that diverge from the geodesic
"""

import heapq
import json
import logging
import math
import os
import time

import numpy as np
from scipy.sparse import csr_matrix, load_npz

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("geodesic")

DATA_DIR = os.environ.get("MANIFOLD_DATA_DIR", "data")


class GeodesicEngine:
    """Geodesic pathfinding on the manifold graph."""

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.W = None              # adjacency matrix
        self.manifold_coords = None  # (n, d) manifold coordinates
        self.fact_index = None     # {id: {idx, text}}
        self.idx_to_id = None     # list[str] — index to fact ID
        self.embeddings = None     # (n, dim) sentence embeddings
        self._loaded = False

    def load(self):
        """Load precomputed manifold data."""
        log.info(f"Loading manifold from {self.data_dir}/")
        self.W = load_npz(os.path.join(self.data_dir, "adjacency.npz"))
        self.manifold_coords = np.load(os.path.join(self.data_dir, "manifold_coords.npy"))
        self.embeddings = np.load(os.path.join(self.data_dir, "embeddings.npy"))

        with open(os.path.join(self.data_dir, "fact_index.json"), "r") as f:
            self.fact_index = json.load(f)

        # Build index-to-id mapping
        self.idx_to_id = [""] * len(self.fact_index)
        for fid, info in self.fact_index.items():
            self.idx_to_id[info["idx"]] = fid

        self._loaded = True
        n = self.W.shape[0]
        d = self.manifold_coords.shape[1]
        log.info(f"Loaded: {n} facts, {d}D manifold, {self.W.nnz} edges")

    def _ensure_loaded(self):
        if not self._loaded:
            self.load()

    # ------------------------------------------------------------------
    # Core: find facts closest to a query in manifold space
    # ------------------------------------------------------------------
    def locate_on_manifold(self, query_embedding: np.ndarray, top_k: int = 5) -> list[int]:
        """
        Find the manifold position of a query by locating nearest facts
        in embedding space.

        Returns list of fact indices.
        """
        self._ensure_loaded()
        # Cosine similarity
        sims = query_embedding @ self.embeddings.T
        top_idx = np.argpartition(sims, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return top_idx.tolist()

    def manifold_centroid(self, indices: list[int]) -> np.ndarray:
        """Compute centroid of given facts in manifold coordinates."""
        self._ensure_loaded()
        coords = self.manifold_coords[indices]
        return coords.mean(axis=0)

    # ------------------------------------------------------------------
    # Geodesic computation (Dijkstra with -log cost)
    # ------------------------------------------------------------------
    def compute_geodesic(self, source_idx: int, target_idx: int,
                          max_path_len: int = 20) -> tuple[list[int], float]:
        """
        Find shortest path from source to target on the manifold graph.

        Uses Dijkstra with cost = -log(weight), so the path maximizes
        the product of edge weights (= strongest connection chain).

        Returns:
            (path_indices, total_cost)
        """
        self._ensure_loaded()
        n = self.W.shape[0]

        dist = np.full(n, np.inf)
        prev = np.full(n, -1, dtype=np.int32)
        dist[source_idx] = 0.0
        pq = [(0.0, source_idx)]
        visited = set()

        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)

            if u == target_idx:
                break

            # Get neighbors from sparse matrix
            row = self.W.getrow(u)
            neighbors = row.indices
            weights = row.data

            for v, w in zip(neighbors, weights):
                if v in visited or w < 1e-10:
                    continue
                cost = -math.log(w + 1e-10)
                new_dist = d + cost
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(pq, (new_dist, int(v)))

        # Reconstruct path
        if dist[target_idx] == np.inf:
            return [], float('inf')

        path = []
        node = target_idx
        while node != -1 and len(path) <= max_path_len:
            path.append(int(node))
            node = int(prev[node])
        path.reverse()

        return path, float(dist[target_idx])

    def compute_geodesic_multi(self, source_indices: list[int],
                                target_indices: list[int]) -> tuple[list[int], float]:
        """
        Find the shortest geodesic between any source and any target.
        Useful when query/goal map to multiple candidate facts.
        """
        best_path = []
        best_cost = float('inf')

        for s in source_indices[:3]:  # limit search space
            for t in target_indices[:3]:
                if s == t:
                    continue
                path, cost = self.compute_geodesic(s, t)
                if path and cost < best_cost:
                    best_path = path
                    best_cost = cost

        return best_path, best_cost

    # ------------------------------------------------------------------
    # Geodesic neighborhood
    # ------------------------------------------------------------------
    def geodesic_neighborhood(self, path: list[int], radius: float = 0.1,
                               max_neighbors: int = 20) -> list[dict]:
        """
        Find facts within radius of the geodesic in manifold coordinates.

        These are facts "near the trail" — not on the exact path but
        potentially relevant (alternative routes, context, gotchas).

        Returns list of {idx, id, text, distance_to_path, nearest_waypoint}.
        """
        self._ensure_loaded()
        if not path:
            return []

        path_coords = self.manifold_coords[path]  # (path_len, d)
        path_set = set(path)
        neighbors = []

        # For each fact, compute distance to nearest point on geodesic
        for i in range(self.manifold_coords.shape[0]):
            if i in path_set:
                continue

            point = self.manifold_coords[i]
            # Distance to each waypoint
            dists = np.linalg.norm(path_coords - point, axis=1)
            min_dist = float(dists.min())
            nearest_wp = int(dists.argmin())

            if min_dist < radius:
                fid = self.idx_to_id[i]
                neighbors.append({
                    "idx": i,
                    "id": fid,
                    "text": self.fact_index.get(fid, {}).get("text", ""),
                    "distance_to_path": round(min_dist, 4),
                    "nearest_waypoint": nearest_wp,
                    "waypoint_fact_idx": path[nearest_wp],
                })

        # Sort by distance, cap
        neighbors.sort(key=lambda x: x["distance_to_path"])
        return neighbors[:max_neighbors]

    # ------------------------------------------------------------------
    # Tangent space — branching directions
    # ------------------------------------------------------------------
    def tangent_directions(self, waypoint_idx: int, path: list[int],
                            top_k: int = 5) -> list[dict]:
        """
        At a waypoint on the geodesic, find the strongest edges that
        go in different directions than the path.

        These represent alternative routes the "hiker" could take.
        """
        self._ensure_loaded()
        path_set = set(path)

        row = self.W.getrow(waypoint_idx)
        neighbors = row.indices
        weights = row.data

        # Filter to non-path neighbors
        tangents = []
        for v, w in zip(neighbors, weights):
            v = int(v)
            if v in path_set:
                continue
            fid = self.idx_to_id[v]
            tangents.append({
                "idx": v,
                "id": fid,
                "text": self.fact_index.get(fid, {}).get("text", ""),
                "weight": float(w),
            })

        tangents.sort(key=lambda x: x["weight"], reverse=True)
        return tangents[:top_k]

    # ------------------------------------------------------------------
    # Full geodesic retrieval pipeline
    # ------------------------------------------------------------------
    def geodesic_retrieve(
        self,
        query_embedding: np.ndarray,
        goal_embedding: np.ndarray | None = None,
        top_k: int = 10,
        neighborhood_radius: float = 0.1,
    ) -> dict:
        """
        Full geodesic retrieval:
        1. Locate query and goal on manifold
        2. Compute geodesic between them
        3. Collect facts on the geodesic
        4. Collect facts near the geodesic
        5. Identify tangent directions at key waypoints

        If no goal_embedding is provided, returns facts from the query's
        manifold neighborhood (no geodesic).
        """
        self._ensure_loaded()
        t0 = time.time()

        # Locate query on manifold
        source_indices = self.locate_on_manifold(query_embedding, top_k=5)

        if goal_embedding is not None:
            # Locate goal on manifold
            target_indices = self.locate_on_manifold(goal_embedding, top_k=5)

            # Compute geodesic
            path, cost = self.compute_geodesic_multi(source_indices, target_indices)

            if not path:
                # Fallback: return nearest facts to query
                return self._fallback_retrieve(source_indices, top_k, t0)

            # Collect path facts
            path_facts = []
            for pos, idx in enumerate(path):
                fid = self.idx_to_id[idx]
                path_facts.append({
                    "idx": idx,
                    "id": fid,
                    "text": self.fact_index.get(fid, {}).get("text", ""),
                    "position": pos,
                    "on_geodesic": True,
                    "source": "geodesic_path",
                })

            # Neighborhood facts
            neighborhood = self.geodesic_neighborhood(path, radius=neighborhood_radius)
            for n in neighborhood:
                n["on_geodesic"] = False
                n["source"] = "geodesic_neighborhood"

            # Tangent directions at start, middle, and end
            tangents = {}
            waypoint_positions = [0, len(path) // 2, -1]
            for wp_pos in waypoint_positions:
                wp_idx = path[wp_pos]
                tangents[wp_pos] = self.tangent_directions(wp_idx, path)

            elapsed = time.time() - t0

            # Combine and rank: path facts first, then neighborhood by distance
            all_results = path_facts + [
                {**n, "combined_score": 1.0 / (1.0 + n["distance_to_path"])}
                for n in neighborhood
            ]

            return {
                "results": all_results[:top_k],
                "geodesic": {
                    "path_length": len(path),
                    "total_cost": round(cost, 4),
                    "path_facts": [f["text"][:80] for f in path_facts],
                },
                "tangent_directions": tangents,
                "stats": {
                    "source_facts": len(source_indices),
                    "target_facts": len(target_indices),
                    "path_facts": len(path_facts),
                    "neighborhood_facts": len(neighborhood),
                    "elapsed_sec": round(elapsed, 3),
                },
            }
        else:
            return self._fallback_retrieve(source_indices, top_k, t0)

    def _fallback_retrieve(self, seed_indices: list[int], top_k: int,
                            t0: float) -> dict:
        """When no goal is specified, return manifold neighborhood of query."""
        centroid = self.manifold_centroid(seed_indices)
        dists = np.linalg.norm(self.manifold_coords - centroid, axis=1)
        nearest = np.argsort(dists)[:top_k]

        results = []
        for idx in nearest:
            idx = int(idx)
            fid = self.idx_to_id[idx]
            results.append({
                "idx": idx,
                "id": fid,
                "text": self.fact_index.get(fid, {}).get("text", ""),
                "manifold_distance": float(dists[idx]),
                "source": "manifold_neighborhood",
            })

        return {
            "results": results,
            "geodesic": None,
            "stats": {
                "seed_facts": len(seed_indices),
                "elapsed_sec": round(time.time() - t0, 3),
            },
        }

    # ------------------------------------------------------------------
    # Manifold distance between two facts
    # ------------------------------------------------------------------
    def manifold_distance(self, idx_a: int, idx_b: int) -> float:
        """Euclidean distance in manifold coordinates ≈ geodesic distance."""
        self._ensure_loaded()
        return float(np.linalg.norm(
            self.manifold_coords[idx_a] - self.manifold_coords[idx_b]
        ))
