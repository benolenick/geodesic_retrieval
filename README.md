# Geodesic Retrieval over Learned Manifolds

A differential geometry approach to knowledge base search. Instead of retrieving by cosine similarity in flat embedding spaces, we construct a Riemannian manifold over the knowledge base using diffusion maps and retrieve along geodesic shortest paths.

**Key results** (75,867 cybersecurity knowledge facts):
- **+22% path completeness** over FAISS cosine baseline
- **+5% path completeness** over LLM-augmented retrieval
- **5x faster** than LLM-augmented approach (2.4s vs 12s)
- **Zero regression** on Recall@5

## Method

1. **Manifold Construction** — k-NN graph with locally-scaled Gaussian kernels (Zelnik-Manor & Perona 2004), normalized graph Laplacian, diffusion map embedding (Coifman & Lafon 2006) into 12 dimensions
2. **Geodesic Retrieval** — Dijkstra shortest paths on the manifold graph with -log(weight) cost transform
3. **DTW Shape Matching** — Dynamic Time Warping against known procedural chain signatures encoded as 7-dimensional structure vectors
4. **Nystrom Extension** — project unseen queries onto the prebuilt manifold at inference time
5. **Blended Scoring** — unified ranking: 0.5 FAISS + 0.35 geodesic + 0.15 shape

## Why geodesics?

Cosine similarity retrieves facts *near* the query. For chain-structured knowledge (A -> B -> C -> D), it finds A and D but misses the intermediate steps B and C. Geodesic retrieval follows the curved geometry of the embedding space, naturally discovering intermediate chain steps.

## Repository Structure

```
src/
  manifold_builder.py    # Builds the diffusion map manifold from embeddings
  geodesic_engine.py     # Geodesic shortest path retrieval
  shape_matcher.py       # DTW shape matching against known chains
  structure_classifier.py # 7-dim structure vector classification
  server.py              # HTTP server with blended scoring
  smart_router.py        # Query-type routing between retrieval backends
eval/
  comparison_test.py     # 10-case core benchmark
  comparison_test_expanded.py  # 30-case expanded benchmark
  weight_sweep.py        # Scoring weight sensitivity analysis
  expanded_test_cases.py # Test case definitions
paper/
  paper_full.md          # Full paper (markdown)
  Geodesic_Retrieval_Paper.docx  # Full paper (formatted)
  figures/               # All paper figures
```

## Results

| Engine | R@5 | R@10 | Path Completeness | Latency |
|--------|-----|------|-------------------|---------|
| FAISS baseline | 52% | 52% | 51% | 57ms |
| LLM-augmented (Methodic v2.0) | 47% | 57% | 68% | 12.0s |
| **Manifold Retrieval** | **52%** | **60%** | **73%** | **2.4s** |

Path completeness on the 30-case expanded benchmark reaches **80%**.

## Requirements

- Python 3.10+
- numpy, scipy, scikit-learn
- faiss-cpu or faiss-gpu
- An embedding model (we used all-MiniLM-L6-v2 via Ollama)
- A knowledge base with embeddings (we used Memoria)

## Citation

If you use this work, please cite:

```
Olenick, B. (2026). Geodesic Retrieval over Learned Manifolds: A Differential
Geometry Approach to Knowledge Base Search. Zenodo.
```

## License

CC BY 4.0
