#!/usr/bin/env python3
"""
Manifold Retrieval Server — HTTP API for geodesic-based knowledge retrieval.

Wraps the full pipeline:
  1. Query → FAISS seeds → manifold position
  2. Goal identification → manifold target
  3. Geodesic path computation
  4. Chart selection (facts along/near geodesic)
  5. Shape matching against known attack chains
  6. Structure-aware re-ranking

Listens on port 8003. Backend: Memoria (:8000) for FAISS, precomputed manifold.

Endpoints:
  POST /manifold-search — full geodesic retrieval
  POST /search          — proxy to Memoria FAISS (baseline)
  GET  /health          — status
  GET  /manifold-stats  — manifold metadata
"""

import json
import logging
import os
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import Request, urlopen

import numpy as np

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geodesic_engine import GeodesicEngine
from shape_matcher import ShapeMatcher
from structure_classifier import classify_fact, heuristic_classify

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("manifold_server")

PORT = int(os.environ.get("MANIFOLD_PORT", 8003))
MEMORIA_URL = os.environ.get("MEMORIA_URL", "http://127.0.0.1:8000")
DATA_DIR = os.environ.get("MANIFOLD_DATA_DIR", "data")
KNOWN_CHAINS_PATH = os.environ.get("KNOWN_CHAINS_PATH", "known_chains.json")

# Global instances
engine: GeodesicEngine | None = None
matcher: ShapeMatcher | None = None
embedding_model = None
text_to_manifold_idx: dict[str, int] = {}  # text[:200].lower() → manifold index


def get_embedding_model():
    """Lazy-load sentence-transformers model."""
    global embedding_model
    if embedding_model is None:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("Loaded embedding model: all-MiniLM-L6-v2")
    return embedding_model


def embed_text(text: str) -> np.ndarray:
    """Embed a single text string."""
    model = get_embedding_model()
    return model.encode([text], normalize_embeddings=True)[0]


def search_memoria(query: str, top_k: int = 10) -> list[dict]:
    """Proxy search to Memoria FAISS."""
    try:
        payload = json.dumps({"query": query, "top_k": top_k}).encode()
        req = Request(
            f"{MEMORIA_URL}/search",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        return data.get("results", [])
    except Exception as e:
        log.warning(f"Memoria search failed: {e}")
        return []


def nystrom_project(fact_embedding: np.ndarray) -> np.ndarray | None:
    """
    Project a fact not on the manifold onto manifold coordinates
    using Nyström extension: weighted average of k nearest manifold facts.
    """
    if engine is None or not engine._loaded:
        return None
    sims = fact_embedding @ engine.embeddings.T
    k = min(10, engine.embeddings.shape[0])
    top_idx = np.argpartition(sims, -k)[-k:]
    top_sims = sims[top_idx]
    distances = 1.0 - top_sims
    sigma = max(float(np.median(distances)), 1e-6)
    weights = np.exp(-(distances ** 2) / (2 * sigma ** 2))
    w_sum = weights.sum()
    if w_sum < 1e-10:
        return None
    weights /= w_sum
    return (weights[:, None] * engine.manifold_coords[top_idx]).sum(axis=0)


def compute_geodesic_proximity(fact_coords: np.ndarray, path_coords: np.ndarray) -> float:
    """Distance from a point to the nearest waypoint on the geodesic path."""
    if path_coords is None or len(path_coords) == 0:
        return float('inf')
    dists = np.linalg.norm(path_coords - fact_coords, axis=1)
    return float(dists.min())


def full_manifold_search(query: str, goal: str | None = None,
                          top_k: int = 10) -> dict:
    """
    Full manifold retrieval pipeline with blended scoring.

    Instead of strict priority (geodesic > shape > FAISS), all candidates
    are scored with a blended formula:
      score = α * faiss_relevance + β * geodesic_proximity + γ * shape_bonus

    FAISS results not on the manifold get Nyström-projected, so the manifold
    can score ALL candidates even though it only contains ~4K of 75K facts.
    """
    t0 = time.time()

    # Scoring weights
    ALPHA = 0.5   # FAISS semantic relevance weight
    BETA = 0.35   # geodesic proximity weight
    GAMMA = 0.15  # shape match bonus weight

    # Embed query and goal
    query_emb = embed_text(query)
    goal_emb = embed_text(goal) if goal else None

    # Get FAISS results (extra candidates for blending)
    faiss_results = search_memoria(query, top_k=top_k * 2)

    # Geodesic retrieval
    geo_result = engine.geodesic_retrieve(
        query_embedding=query_emb,
        goal_embedding=goal_emb,
        top_k=top_k * 2,
        neighborhood_radius=0.15,
    )

    # Build geodesic path coordinates for proximity scoring
    path_coords = None
    geodesic_path = geo_result.get("geodesic")
    if geodesic_path and geodesic_path.get("path_length", 0) > 1:
        # Reconstruct path indices from geo results
        path_indices = [f["idx"] for f in geo_result.get("results", [])
                       if f.get("on_geodesic")]
        if path_indices:
            path_coords = engine.manifold_coords[path_indices]

    # Shape matching
    shape_results = []
    shape_queries = []
    shape_chain_texts = set()
    if path_coords is not None and len(path_coords) >= 2:
        path_structures = []
        for fact in geo_result["results"]:
            if fact.get("on_geodesic"):
                sv = heuristic_classify(fact.get("text", ""))
                path_structures.append(sv)

        if len(path_structures) >= 2:
            partial_path = np.array(path_structures)
            shape_results = matcher.match_shape(partial_path, top_k=3)
            shape_queries = matcher.shape_guided_retrieval_queries(
                path_structures, top_k_chains=2
            )

    # Supplement with shape-guided FAISS searches
    shape_supplemented = []
    if shape_queries:
        for sq in shape_queries[:5]:
            supp = search_memoria(sq, top_k=3)
            for r in supp:
                r["_shape_guided"] = True
                r["shape_query"] = sq
                shape_supplemented.append(r)

    # === Collect ALL candidates into a unified pool ===
    candidates = {}  # text_key -> candidate dict

    def _text_key(text):
        return text[:200].lower().strip()

    # 1. Geodesic results (have manifold position)
    for fact in geo_result.get("results", []):
        text = fact.get("text", "")
        key = _text_key(text)
        if not key:
            continue
        on_geo = fact.get("on_geodesic", False)
        # Compute geodesic proximity
        geo_prox = 0.0
        if path_coords is not None:
            idx = fact.get("idx")
            if idx is not None:
                coords = engine.manifold_coords[idx]
                dist = compute_geodesic_proximity(coords, path_coords)
                geo_prox = 1.0 / (1.0 + dist * 5.0)  # normalize to ~[0,1]
            if on_geo:
                geo_prox = 1.0  # on the path = maximum proximity

        candidates[key] = {
            "text": text,
            "faiss_relevance": 0.0,  # will be filled if also in FAISS
            "geo_proximity": geo_prox,
            "shape_bonus": 0.0,
            "retrieval_method": "geodesic_path" if on_geo else "geodesic_neighborhood",
            "on_manifold": True,
        }

    # 2. Shape-guided results
    for fact in shape_supplemented:
        text = fact.get("text", fact.get("fact", ""))
        key = _text_key(text)
        if not key:
            continue
        if key in candidates:
            candidates[key]["shape_bonus"] = 0.5
            continue
        # Nyström-project onto manifold for proximity scoring
        geo_prox = 0.0
        if path_coords is not None:
            fact_emb = embed_text(text)
            coords = nystrom_project(fact_emb)
            if coords is not None:
                dist = compute_geodesic_proximity(coords, path_coords)
                geo_prox = 1.0 / (1.0 + dist * 5.0)

        candidates[key] = {
            "text": text,
            "faiss_relevance": fact.get("relevance", 0.5),
            "geo_proximity": geo_prox,
            "shape_bonus": 0.5,
            "retrieval_method": "shape_guided",
            "on_manifold": False,
        }

    # 3. FAISS results (may or may not be on manifold)
    for fact in faiss_results:
        text = fact.get("text", fact.get("fact", ""))
        key = _text_key(text)
        if not key:
            continue
        relevance = fact.get("relevance", 0.5)

        if key in candidates:
            # Already in pool — update FAISS relevance
            candidates[key]["faiss_relevance"] = max(
                candidates[key]["faiss_relevance"], relevance
            )
            continue

        # New candidate from FAISS — check manifold cache or Nyström-project
        geo_prox = 0.0
        on_manifold = False
        if path_coords is not None:
            # Fast lookup: is this fact already on the manifold?
            midx = text_to_manifold_idx.get(key)
            if midx is not None:
                coords = engine.manifold_coords[midx]
                dist = compute_geodesic_proximity(coords, path_coords)
                geo_prox = 1.0 / (1.0 + dist * 5.0)
                on_manifold = True
            else:
                # Nyström extension — project FAISS result onto manifold
                fact_emb = embed_text(text)
                coords = nystrom_project(fact_emb)
                if coords is not None:
                    dist = compute_geodesic_proximity(coords, path_coords)
                    geo_prox = 1.0 / (1.0 + dist * 5.0)

        candidates[key] = {
            "text": text,
            "faiss_relevance": relevance,
            "geo_proximity": geo_prox,
            "shape_bonus": 0.0,
            "retrieval_method": "faiss",
            "on_manifold": on_manifold,
        }

    # === Compute blended scores ===
    for key, c in candidates.items():
        c["blended_score"] = (
            ALPHA * c["faiss_relevance"] +
            BETA * c["geo_proximity"] +
            GAMMA * c["shape_bonus"]
        )

    # Sort by blended score
    ranked = sorted(candidates.values(), key=lambda x: x["blended_score"], reverse=True)

    elapsed = time.time() - t0

    return {
        "results": ranked[:top_k],
        "geodesic": geo_result.get("geodesic"),
        "shape_matches": shape_results,
        "shape_guided_queries": shape_queries,
        "tangent_directions": geo_result.get("tangent_directions"),
        "stats": {
            "total_candidates": len(ranked),
            "geodesic_facts": sum(1 for r in ranked if r.get("retrieval_method") == "geodesic_path"),
            "neighborhood_facts": sum(1 for r in ranked if r.get("retrieval_method") == "geodesic_neighborhood"),
            "shape_guided_facts": sum(1 for r in ranked if r.get("retrieval_method") == "shape_guided"),
            "faiss_facts": sum(1 for r in ranked if r.get("retrieval_method") == "faiss"),
            "on_manifold": sum(1 for r in ranked if r.get("on_manifold")),
            "nystrom_projected": sum(1 for r in ranked if not r.get("on_manifold")),
            "elapsed_sec": round(elapsed, 3),
            "scoring_weights": {"alpha_faiss": ALPHA, "beta_geodesic": BETA, "gamma_shape": GAMMA},
        },
    }


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------
class ManifoldHandler(BaseHTTPRequestHandler):
    server_version = "ManifoldRetrieval/1.0"

    def log_message(self, format, *args):
        log.info(f"{self.address_string()} {format % args}")

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _respond(self, code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {
                "status": "ok",
                "engine": "manifold_retrieval",
                "manifold_loaded": engine._loaded if engine else False,
                "n_known_chains": len(matcher.chains) if matcher else 0,
            })
        elif self.path == "/manifold-stats":
            if engine and engine._loaded:
                self._respond(200, {
                    "n_facts": engine.manifold_coords.shape[0],
                    "manifold_dims": engine.manifold_coords.shape[1],
                    "n_edges": engine.W.nnz,
                    "n_known_chains": len(matcher.chains) if matcher else 0,
                })
            else:
                self._respond(503, {"error": "manifold not loaded"})
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/manifold-search":
            self._handle_manifold_search()
        elif path == "/search":
            self._handle_search_proxy()
        else:
            self._respond(404, {"error": "not found"})

    def _handle_manifold_search(self):
        body = self._read_body()
        query = body.get("query", "").strip()
        if not query:
            self._respond(400, {"error": "query required"})
            return
        goal = body.get("goal", "").strip() or None
        top_k = int(body.get("top_k", 10))

        try:
            result = full_manifold_search(query, goal=goal, top_k=top_k)
            self._respond(200, result)
        except Exception as e:
            log.error(f"Manifold search error: {e}", exc_info=True)
            self._respond(500, {"error": str(e)})

    def _handle_search_proxy(self):
        """Proxy to Memoria FAISS for baseline comparison."""
        body = self._read_body()
        query = body.get("query", "").strip()
        if not query:
            self._respond(400, {"error": "query required"})
            return
        top_k = int(body.get("top_k", 10))
        results = search_memoria(query, top_k=top_k)
        self._respond(200, {"results": results})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global engine, matcher

    log.info("Initializing Manifold Retrieval Server...")

    # Load geodesic engine
    engine = GeodesicEngine(data_dir=DATA_DIR)
    try:
        engine.load()
    except FileNotFoundError as e:
        log.error(f"Manifold data not found: {e}")
        log.error(f"Run manifold_builder.py first to build the manifold")
        sys.exit(1)

    # Load shape matcher
    matcher = ShapeMatcher(chains_path=KNOWN_CHAINS_PATH)

    # Build text-to-index lookup for fast manifold matching
    global text_to_manifold_idx
    for fid, info in engine.fact_index.items():
        key = info.get("text", "")[:200].lower().strip()
        if key:
            text_to_manifold_idx[key] = info["idx"]
    log.info(f"Built text lookup cache: {len(text_to_manifold_idx)} entries")

    # Warm up embedding model
    log.info("Warming up embedding model...")
    embed_text("test query warm up")

    # Start server
    server = HTTPServer(("0.0.0.0", PORT), ManifoldHandler)
    log.info(f"Manifold Retrieval Server listening on port {PORT}")
    log.info(f"  Manifold: {engine.manifold_coords.shape[0]} facts, "
             f"{engine.manifold_coords.shape[1]}D")
    log.info(f"  Known chains: {len(matcher.chains)}")
    log.info(f"  Endpoints: /manifold-search, /search, /health, /manifold-stats")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
