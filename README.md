# Geodesic Retrieval over Learned Manifolds

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18971939.svg)](https://doi.org/10.5281/zenodo.18971939)

A differential geometry approach to knowledge base search. Instead of retrieving by cosine similarity in flat embedding spaces, we construct a Riemannian manifold over the knowledge base using diffusion maps and retrieve along geodesic shortest paths.

## The problem with flat retrieval

FAISS, Pinecone, ChromaDB, and every other vector database retrieve by cosine similarity — they find facts *near* the query in embedding space. This works for simple lookups, but fails when the answer is a **chain of connected facts**.

Consider any domain where knowledge forms sequences:

- **Medical diagnosis**: symptom → differential → diagnostic test → confirmation → treatment
- **Legal reasoning**: statute → judicial interpretation → exception → application
- **Troubleshooting**: symptom → root cause analysis → diagnostic procedure → fix → verification
- **Chemistry**: reagent → reaction conditions → intermediate → purification → product

In all of these, cosine similarity finds facts near the start and end of the chain, but **misses the intermediate steps** — the critical connective tissue that makes the sequence actionable. Step B is textually similar to A and to C, but not to the original query. Flat retrieval can't follow the curve.

## The insight

Embedding spaces have intrinsic geometry that flat similarity search ignores. Facts connected through procedural chains lie along **curved paths** through the space. We make this geometry explicit using diffusion maps, then retrieve along **geodesic shortest paths** on the resulting manifold — following the curve instead of cutting across it.

**Key results** (evaluated on 75,867 knowledge facts):
- **+22% path completeness** over FAISS cosine baseline
- **+5% path completeness** over LLM-augmented retrieval
- **5x faster** than LLM-augmented approach (2.4s vs 12s)
- **Zero regression** on Recall@5 for simple queries

## Method

1. **Manifold Construction** — k-NN graph with locally-scaled Gaussian kernels (Zelnik-Manor & Perona 2004), normalized graph Laplacian, diffusion map embedding (Coifman & Lafon 2006) into 12 dimensions
2. **Geodesic Retrieval** — Dijkstra shortest paths on the manifold graph with -log(weight) cost transform. High-affinity edges have low cost, so paths follow the natural geometry of the knowledge space.
3. **DTW Shape Matching** — Dynamic Time Warping against known procedural chain signatures encoded as 7-dimensional structure vectors. Catches cases where the right facts are retrieved but in a pattern that doesn't match known workflows.
4. **Nystrom Extension** — project unseen queries onto the prebuilt manifold at inference time without rebuilding
5. **Blended Scoring** — unified ranking: 0.5 FAISS + 0.35 geodesic + 0.15 shape

The method is **domain-agnostic**. It requires only embeddings — no entity extraction, no schema design, no graph construction. The manifold emerges entirely from the geometry of the embedding space itself.

## Why this matters for RAG

Any RAG system retrieving over chain-structured knowledge (which is most real-world knowledge) is leaving performance on the table with flat cosine similarity. Geodesic retrieval is a drop-in improvement: build the manifold offline, query it at inference time via Nystrom projection. No LLM calls needed at query time.

The 87% path completeness on multi-hop queries (vs 59% for FAISS) is the headline number — that's the case where following the geometry matters most.

## Results

Evaluated on a cybersecurity knowledge base (75,867 facts) as a test domain, since cybersecurity procedures are inherently chain-structured (reconnaissance → exploitation → privilege escalation → lateral movement).

| Engine | R@5 | R@10 | Path Completeness | Latency |
|--------|-----|------|-------------------|---------|
| FAISS baseline | 52% | 52% | 51% | 57ms |
| LLM-augmented (Methodic v2.0) | 47% | 57% | 68% | 12.0s |
| **Manifold Retrieval** | **52%** | **60%** | **73%** | **2.4s** |

Path completeness on the 30-case expanded benchmark reaches **80%**.

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
Geometry Approach to Knowledge Base Search. Zenodo. https://doi.org/10.5281/zenodo.18971939
```

## License

CC BY 4.0
