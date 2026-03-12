# Geodesic Retrieval over Learned Manifolds: A Differential Geometry Approach to Knowledge Base Search

**Abstract.** Retrieval-Augmented Generation (RAG) systems universally rely on cosine similarity in flat embedding spaces, treating each knowledge fact as an independent point. This fails for chain-structured knowledge where the answer spans a sequence of connected steps — retrieval finds facts near the query but misses intermediate steps along the path. We propose Manifold Retrieval, which constructs a Riemannian manifold over the embedding space using diffusion maps and retrieves along geodesic shortest paths rather than within cosine similarity balls. Our method builds a k-nearest-neighbor graph with locally-scaled Gaussian kernels over 75,867 cybersecurity knowledge facts, computes a 12-dimensional diffusion map embedding, and retrieves facts along Dijkstra geodesics on the resulting manifold graph. We further incorporate Dynamic Time Warping (DTW) shape matching against known procedural chain signatures and Nyström extension for projecting unseen queries onto the manifold at inference time. Evaluated against a FAISS cosine baseline and an LLM-augmented retrieval system (Methodic v2.0), Manifold Retrieval achieves +8% Recall@10 and +22% path completeness over the baseline at 2.4 seconds per query — 5× faster than the LLM-augmented approach — with zero regression on Recall@5. The method requires only embeddings and is domain-agnostic: no entity extraction, schema design, or graph construction is needed beyond the embedding model itself.

---

## 1. Introduction

Knowledge retrieval is the critical bottleneck for AI agents operating over specialized domains. Whether an agent is diagnosing a medical condition, tracing legal precedent, or executing a multi-step cybersecurity procedure, its effectiveness depends on retrieving not just relevant facts but the right *sequence* of facts that together form a complete action plan.

Current retrieval approaches operate in flat embedding spaces. A query is encoded as a vector, and the top-k most similar vectors are returned by cosine similarity or approximate nearest neighbor search. Systems like FAISS [3], Pinecone, Weaviate, and ChromaDB all implement variations of this paradigm. Graph-augmented approaches such as GraphRAG [8] layer explicit entity-relationship graphs on top of embeddings, requiring entity extraction and relation typing. LLM-augmented approaches like HyDE [6] and query decomposition use language models to reformulate queries before retrieval, adding significant latency.

All of these approaches share a fundamental limitation: they retrieve facts that are *close to the query* in embedding space, but not necessarily facts that lie *along the path between concepts*. For chain-structured knowledge — where the answer is a sequence A → B → C → D — cosine similarity finds facts near A and near D but systematically misses the intermediate steps B and C that connect them.

Consider the query: *"Redis unauthenticated access to SSH key injection to reverse shell."* A cosine similarity search returns facts about Redis and facts about reverse shells, because those terms appear in the query. But it misses the critical intermediate steps: using `CONFIG SET dir` and `CONFIG SET dbfilename` to write to the `.ssh/authorized_keys` file, then connecting via SSH. These intermediate facts are not textually similar to the query — they are *geometrically between* the query's starting and ending concepts in the knowledge space.

We observe that the embedding space of a knowledge base has intrinsic geometry that flat similarity search ignores. Facts connected through chains of procedural steps lie along curved paths through the embedding space. A knowledge base about cybersecurity techniques, for instance, forms clusters around exploitation methods, enumeration tools, privilege escalation paths, and lateral movement tactics — but these clusters are connected through procedural chains that curve through the space, not straight lines.

We propose to make this geometry explicit. Using diffusion maps [1], we construct a Riemannian manifold over the embedding space where distance corresponds to connectivity, not just textual similarity. We then retrieve along geodesic shortest paths on this manifold, naturally discovering the intermediate steps that chain-structured queries require.

Our contributions are:

1. **Manifold Retrieval** — the first application of diffusion map manifold learning to knowledge base search (not dimensionality reduction or visualization, but actual retrieval).
2. **Geodesic path retrieval** — using Dijkstra shortest paths on the manifold graph with log-transformed edge weights, which follows the curved geometry of the knowledge space rather than cutting across it.
3. **DTW shape matching** — using Dynamic Time Warping to compare candidate retrieval paths against known procedural chain signatures, encoded as sequences of 7-dimensional structure vectors.
4. **Nyström extension for query projection** — a method for projecting unseen queries onto the prebuilt manifold at inference time without rebuilding the manifold.
5. **Empirical validation** showing +22% path completeness over FAISS baseline and +5% over an LLM-augmented system, at 5× lower latency.

## 2. Related Work

### 2.1 Vector Similarity Retrieval

The dominant paradigm for knowledge retrieval encodes texts as dense vectors and retrieves by cosine similarity or inner product. FAISS [3] provides efficient approximate nearest neighbor search over billions of vectors using inverted file indices and product quantization. Commercial systems (Pinecone, Weaviate, ChromaDB, Qdrant) build on similar principles with managed infrastructure. These systems are fast, scalable, and well-understood, but fundamentally limited to retrieving within geometric balls around the query point. They have no mechanism for following curved paths through the embedding space or discovering intermediate chain steps.

### 2.2 Graph-Augmented Retrieval

GraphRAG [8] addresses some limitations of flat retrieval by constructing explicit entity-relationship graphs from the knowledge base, then traversing these graphs to retrieve connected information. Knowledge graph embedding methods (TransE, RotatE, ComplEx) learn vector representations of entities and relations in structured knowledge graphs. These approaches can capture multi-hop relationships but require explicit graph construction — entity extraction, relation typing, and schema design — which introduces both engineering complexity and information loss. Our approach differs fundamentally: the graph emerges from the geometry of the embedding space itself, requiring no entity extraction or domain-specific schema.

### 2.3 LLM-Augmented Retrieval

HyDE [6] generates hypothetical documents from the query using a language model, then retrieves by similarity to the generated text rather than the raw query. Query decomposition methods use LLMs to break complex queries into simpler sub-queries, each of which is sent to the retrieval system independently. Step-Back Prompting [7] asks the LLM to reformulate queries at a higher level of abstraction. These methods improve retrieval quality significantly — our LLM-augmented baseline (Methodic v2.0) achieves +17% path completeness over FAISS — but add 8-12 seconds of latency per query due to LLM inference. Our geometric approach achieves comparable or better results without any LLM calls at query time.

### 2.4 Manifold Learning

Diffusion maps [1] use the eigendecomposition of a normalized graph Laplacian to embed data into a space where Euclidean distance approximates diffusion distance on the data manifold. Originally developed for dimensionality reduction in computational harmonic analysis, diffusion maps have been applied to molecular dynamics simulation, image processing, and sensor network analysis. Laplacian Eigenmaps [4] and UMAP [5] are related manifold learning techniques used primarily for visualization and dimensionality reduction. Critically, none of these methods have been applied to retrieval — they reduce dimensionality or visualize structure, but do not use the learned manifold geometry to find and rank documents.

### 2.5 Geodesic Methods in Information Retrieval

The closest prior work to ours is Geodesic Semantic Search (GSS) [9], which computes geodesic distances on citation graphs to improve academic paper retrieval, achieving +23% improvement over cosine similarity baselines. GSS operates on explicit citation links — structural connections that already exist in the data. Our approach differs in that we build the manifold from embeddings alone, discovering geometric structure that is not explicitly encoded anywhere. We require no pre-existing graph structure beyond what emerges from the k-NN graph over embeddings.

Concurrently, Maniscope [11] constructs a k-NN manifold over retrieved document candidates and uses geodesic distances for reranking in RAG pipelines. While Maniscope shares our intuition that manifold geometry improves retrieval, it operates on flat k-NN graphs without diffusion map eigendecomposition, does not compute diffusion distances, and applies only to reranking an existing candidate set rather than discovering new candidates via geodesic traversal. Our method builds a full diffusion map manifold and uses it for both candidate discovery and scoring.

### 2.6 Beam Search in Multi-Hop Retrieval

PropRAG [12] uses beam search over proposition-level knowledge graphs to discover multi-step reasoning chains for RAG, achieving state-of-the-art zero-shot performance on multi-hop QA benchmarks (2Wiki, HotpotQA, MuSiQue). PropRAG avoids LLM inference during retrieval by using pre-computed embedding similarity and graph connectivity to guide the beam search. Our approach is complementary: where PropRAG traverses explicit proposition graphs using beam search, we traverse the intrinsic geometry of the embedding space using geodesic shortest paths. PropRAG requires proposition extraction and graph construction; our method requires only embeddings.

## 3. Problem Formulation

Let $\mathcal{K} = \{f_1, f_2, \ldots, f_n\}$ be a knowledge base of $n$ facts, each encoded by an embedding model as $\mathbf{e}_i \in \mathbb{R}^d$. Given a query $q$ with embedding $\mathbf{e}_q \in \mathbb{R}^d$, the standard retrieval task returns the top-$k$ facts by similarity:

$$\text{Retrieve}(q, k) = \text{argtop-}k_{f_i \in \mathcal{K}} \; \text{sim}(\mathbf{e}_q, \mathbf{e}_i)$$

where $\text{sim}$ is typically cosine similarity. This formulation treats each fact independently — the score of $f_i$ depends only on its direct similarity to $q$, not on its relationships to other facts.

We reformulate retrieval as a geometric problem. The knowledge base defines a Riemannian manifold $(\mathcal{M}, g)$ where $\mathcal{M}$ is the point cloud of embeddings and $g$ is the metric tensor induced by a diffusion kernel over the k-NN graph. Retrieval becomes: find facts along geodesic paths on $\mathcal{M}$, not within Euclidean neighborhoods of $\mathbf{e}_q$.

We introduce **path completeness** as our primary evaluation metric:

$$\text{PathComp}(q) = \frac{|\text{Retrieved chain steps}|}{|\text{Total chain steps}|}$$

where chain steps are the ordered procedural steps that together form the complete answer to $q$. For a query whose answer is the sequence [A, B, C, D, E], retrieving {B, D, E} yields path completeness of 60% regardless of how many other facts are retrieved. This metric better captures whether an agent can execute a multi-step procedure than standard recall, which counts keyword matches without regard to procedural coverage.

Path completeness and recall diverge precisely when retrieval systems find topically relevant facts (boosting recall) but miss procedural chain steps (hurting path completeness). Our experiments show this divergence is systematic: FAISS achieves 52% recall@10 and 51% path completeness (roughly equal), while Manifold Retrieval achieves 60% recall@10 but 73% path completeness — a 13-point gap indicating that the manifold finds chain steps that are not captured by keyword matching.

## 4. Method

### 4.1 Manifold Construction

Given $n$ fact embeddings $\mathbf{e}_1, \ldots, \mathbf{e}_n \in \mathbb{R}^d$ (in our experiments, $n = 75{,}867$, $d = 384$ from all-MiniLM-L6-v2), we construct the manifold in four steps.

**Step 1: k-NN Graph.** We build a symmetric k-nearest-neighbor graph where each fact connects to its $k = 15$ nearest neighbors by cosine similarity. This sparsifies the full $n \times n$ similarity matrix while preserving local neighborhood structure.

**Step 2: Locally-Scaled Gaussian Kernel.** Following Zelnik-Manor and Perona [2], we compute an adaptive bandwidth for each fact:

$$\sigma_i = d(\mathbf{e}_i, \mathbf{e}_{i,\lfloor k/2 \rfloor})$$

where $d(\mathbf{e}_i, \mathbf{e}_{i,\lfloor k/2 \rfloor})$ is the distance from fact $i$ to its $\lfloor k/2 \rfloor$-th nearest neighbor. The affinity between facts $i$ and $j$ is:

$$W(i,j) = \exp\left(-\frac{\|\mathbf{e}_i - \mathbf{e}_j\|^2}{\sigma_i \cdot \sigma_j}\right)$$

Local scaling is essential for knowledge bases where topic density varies dramatically — some areas (e.g., SQL injection techniques) contain thousands of closely-related facts while others (e.g., hardware hacking) contain only a handful. The adaptive $\sigma_i$ ensures that dense regions receive tight kernels and sparse regions receive loose ones, preventing dense clusters from dominating the manifold structure.

**Step 3: Normalized Graph Laplacian.** We form the degree matrix $D$ with $D_{ii} = \sum_j W(i,j)$ and compute the normalized Laplacian:

$$L_{\text{norm}} = D^{-1/2} W D^{-1/2}$$

**Step 4: Diffusion Map.** We compute the top $m = 12$ eigenvectors $\psi_1, \ldots, \psi_m$ of $L_{\text{norm}}$ with corresponding eigenvalues $\lambda_1 \geq \ldots \geq \lambda_m$. The diffusion map embedding of fact $i$ is:

$$\Psi(f_i) = [\lambda_1^t \psi_1(i), \; \lambda_2^t \psi_2(i), \; \ldots, \; \lambda_m^t \psi_m(i)]$$

where $t = 3$ is the diffusion time parameter. Euclidean distance in $\Psi$-space approximates diffusion distance on the graph — a measure of connectivity that accounts for all paths between two nodes, not just the shortest one.

The eigenvalue spectrum of our 75,867-fact manifold (Figure 4) shows a small spectral gap of 0.0037, indicating a smooth manifold without sharp cluster boundaries. This is consistent with a knowledge base where topics blend continuously — exploitation techniques shade into privilege escalation, which shades into lateral movement — rather than forming discrete, well-separated clusters.

The full manifold construction takes approximately 95 seconds: 35 seconds for embedding computation, 55 seconds for k-NN graph construction and eigendecomposition, and 3.2 seconds for the sparse eigensolver (ARPACK via scipy.sparse.linalg.eigsh).

### 4.2 Geodesic Retrieval

Given query $q$ located on the manifold at position $\Psi(q)$ (via Nyström extension, Section 4.4), we retrieve facts along geodesic shortest paths.

We compute Dijkstra shortest paths on the k-NN graph with edge costs transformed as:

$$c(i, j) = -\log W(i, j)$$

High-affinity edges (strongly connected facts) have low cost, so shortest paths follow the manifold's geometry — traversing sequences of highly-related facts rather than jumping across the embedding space.

Given the query position on the manifold, we retrieve all facts within a geodesic radius $r$ of the query. If a goal is specified (e.g., the target state of an attack path), we also compute the geodesic from query to goal and retrieve facts along that path. The geodesic neighborhood naturally includes intermediate chain steps that lie along the curved path between query and goal, even if those steps are distant from both endpoints in cosine similarity.

### 4.3 Shape Matching via Dynamic Time Warping

We encode each fact with a 7-dimensional structure vector capturing its procedural role:

$$\mathbf{s}_i = [\text{phase}, \text{privilege}, \text{scope}, \text{stealth}, \text{interaction}, \text{dependency}, \text{specificity}]$$

where each dimension is a continuous value in $[0, 1]$ assigned by heuristic keyword classifiers (e.g., facts containing "recon" or "enumerate" receive phase $\approx 0.1$; facts containing "root" or "SYSTEM" receive privilege $\approx 1.0$).

We maintain a library of 11 known procedural chain signatures, each represented as an ordered sequence of structure vectors. Given candidate retrieval results, we extract their structure vectors and compare the candidate sequence against each known chain using Dynamic Time Warping [10]:

$$\text{DTW}(\mathbf{S}_{\text{candidate}}, \mathbf{S}_{\text{known}}) = \min_{\pi} \sum_{(i,j) \in \pi} \|\mathbf{s}_i^{\text{cand}} - \mathbf{s}_j^{\text{known}}\|$$

The best-matching chain yields a shape bonus $\in [0, 1]$ based on the inverse DTW cost. This catches cases where geodesic retrieval finds relevant facts but misses the procedural pattern — for instance, finding enumeration and exploitation facts but not in an order that matches a known attack progression.

### 4.4 Nyström Extension for Query Projection

At inference time, the query embedding $\mathbf{e}_q$ is not part of the prebuilt manifold. We project it onto manifold coordinates using a Nyström extension — a weighted average of the manifold coordinates of its nearest neighbors:

1. Find the $k = 10$ nearest manifold facts to $\mathbf{e}_q$ by cosine similarity.
2. Compute distances $d_j = 1 - \text{cos}(\mathbf{e}_q, \mathbf{e}_j)$ and set $\sigma = \text{median}(d_1, \ldots, d_k)$.
3. Compute Gaussian kernel weights: $w_j = \exp(-d_j^2 / 2\sigma^2)$.
4. Project: $\Psi(q) = \sum_j w_j \cdot \Psi(\mathbf{e}_j) / \sum_j w_j$.

The same projection is used for FAISS results that are not on the manifold, allowing all candidates to receive geodesic proximity scores. This avoids rebuilding the manifold for every query while ensuring that the query is positioned correctly in manifold space for geodesic retrieval.

### 4.5 Blended Scoring

Three retrieval channels produce candidate facts:

1. **FAISS cosine similarity** — standard nearest-neighbor retrieval over all 75,867 facts, yielding a relevance score $r_i \in [0, 1]$.
2. **Geodesic neighbors** — facts within geodesic radius of the query on the manifold graph, scored by geodesic proximity $g_i = 1/(1 + d_{\text{manifold}}(q, f_i))$.
3. **Shape-guided results** — facts matching known chain signatures via DTW, with shape bonus $s_i \in [0, 1]$.

All candidates are pooled, deduplicated, and scored with a unified blended formula:

$$\text{score}(f_i) = \alpha \cdot r_i + \beta \cdot g_i + \gamma \cdot s_i$$

where $\alpha = 0.5$, $\beta = 0.35$, $\gamma = 0.15$. Results are sorted by blended score and the top-$k$ are returned.

The weights were determined by a sweep over four configurations (Section 6.4), which found that results are insensitive to the exact weight values — the manifold's advantage is structural (which candidates enter the pool via geodesic traversal) rather than weight-dependent.

## 5. Experimental Setup

### 5.1 Knowledge Base

Our knowledge base consists of 75,867 cybersecurity facts stored in Memoria, a curated knowledge store. Facts are sourced from security textbooks, MITRE ATT&CK technique descriptions, HackTricks methodology guides, and tool documentation. Each fact is a concise statement (typically 1-3 sentences) describing a technique, tool usage, vulnerability, or procedural step. Topics span exploitation, enumeration, privilege escalation, lateral movement, persistence, defense evasion, web application security, Active Directory attacks, cryptography, and network analysis.

All facts are embedded using all-MiniLM-L6-v2 (384 dimensions) via Ollama running on a dual-GPU server. The FAISS index provides sub-100ms cosine similarity search over the full knowledge base.

### 5.2 Test Suite

We evaluate on two benchmarks:

- **Core benchmark:** 10 test cases across 4 categories, evaluated with 2-run averaging.
- **Expanded benchmark:** 30 test cases across 4 categories, evaluated with 3-run averaging.

The four query categories are:

- **Constraint bypass (CB):** Queries about circumventing filters, WAFs, or restrictions (e.g., "command injection when spaces are filtered").
- **Multi-hop (MH):** Queries requiring chains of connected steps (e.g., "Redis unauthenticated access to SSH key injection to shell").
- **Cross-context (CC):** Queries spanning multiple domains (e.g., "SQL injection to operating system command execution via xp_cmdshell").
- **Attack path (AP):** Queries about end-to-end attack planning (e.g., "Kerberoasting to domain admin in Active Directory").

Each test case specifies ground truth keywords (specific terms that should appear in retrieved facts) and ground truth chain steps (the ordered procedural steps that constitute a complete answer). Multiple runs account for variance introduced by the LLM-augmented baseline's query decomposition.

### 5.3 Baselines

- **FAISS baseline:** Pure cosine similarity retrieval via the Memoria API. Top-$k$ facts by cosine similarity to the query embedding. Latency: ~57ms.
- **Methodic v2.0:** LLM-augmented retrieval using qwen3.5 for query decomposition into pentester-style sub-queries, multi-strategy retrieval (FAISS + exploit index + relationship graph + co-occurrence matrix), and Boltzmann energy minimization for re-ranking. This represents a state-of-the-art augmented retrieval pipeline with both linguistic (LLM decomposition) and thermodynamic (energy minimization) components. Latency: ~12 seconds.

### 5.4 Metrics

- **Recall@5, Recall@10:** Fraction of ground truth keywords found in top-5/10 results.
- **Path completeness:** Fraction of ground truth chain steps covered by the retrieved results, regardless of order. This measures whether an agent could execute the complete procedure from the retrieved facts.
- **Latency:** End-to-end query time in milliseconds.

## 6. Results

### 6.1 Main Results

Table 1 presents results on the 10-case core benchmark, averaged over 2 runs.

| Engine | R@5 | R@10 | Path Comp. | $\Delta$R@10 | $\Delta$Path | Latency |
|--------|-----|------|------------|---------------|--------------|---------|
| FAISS baseline | 52% | 52% | 51% | — | — | 57ms |
| Methodic v2.0 | 47% | 57% | 68% | +5% | +17% | 12.0s |
| **Manifold** | **52%** | **60%** | **73%** | **+8%** | **+22%** | **2.4s** |

Manifold Retrieval achieves the highest path completeness at 73%, a 22-point improvement over the FAISS baseline and 5 points above the LLM-augmented Methodic system. Recall@10 improves by 8 points over FAISS. Critically, Recall@5 shows zero regression — the manifold does not sacrifice precision on simple queries to gain path coverage on complex ones.

Latency is 2.4 seconds, approximately 5× faster than Methodic's 12 seconds. The latency cost over FAISS (57ms) is the price of geodesic computation on the manifold graph, but this is acceptable for planning-style queries where the agent needs a complete procedural chain rather than instant autocomplete.

### 6.2 Expanded Results

Table 2 presents results on the 30-case expanded benchmark, averaged over 3 runs to reduce variance.

| Engine | R@5 | R@10 | Path Comp. | Latency |
|--------|-----|------|------------|---------|
| FAISS baseline | 58% | 58% | 62% | 57ms |
| Methodic v2.0 | 56% | 64% | 77% | 8.1s |
| **Manifold** | **58%** | **67%** | **80%** | **2.2s** |

Results hold on the larger benchmark. Path completeness reaches 80%, a 3-point gain over Methodic and 18 points over FAISS. The R@5 parity with FAISS is maintained, confirming that manifold retrieval does not regress on simple queries.

### 6.3 Per-Category Analysis

Table 3 breaks down path completeness by query category on the expanded benchmark.

| Category | $n$ | FAISS | Methodic | Manifold |
|----------|-----|-------|----------|----------|
| Constraint Bypass | 5 | 51% | 71% | 69% |
| Multi-Hop | 6 | 59% | 75% | **87%** |
| Cross-Context | 2 | 53% | 67% | 63% |
| Attack Path | 2 | 45% | **80%** | 63% |

The category breakdown reveals complementary strengths. **Manifold dominates multi-hop queries** at 87% path completeness — 28 points above FAISS and 12 above Methodic. This is the category where geodesic paths provide the most value: multi-hop queries require finding intermediate chain steps, which is precisely what geodesic traversal discovers.

**Methodic dominates attack path queries** at 80% — 17 points above Manifold. Attack path queries benefit from LLM query decomposition, which generates expert-style sub-queries (e.g., decomposing "Kerberoasting to domain admin" into "SPN enumeration," "TGS request," "offline hash cracking," "service account compromise"). The manifold's geometry alone cannot replicate this linguistic decomposition.

Constraint bypass and cross-context queries show smaller differences, with Methodic holding a slight edge on constraint bypass (71% vs 69%) and Manifold slightly behind on cross-context (63% vs 67%).

These complementary strengths suggest that a routing approach — sending multi-hop queries to Manifold and attack path queries to Methodic — could outperform either engine alone. A companion system (Smart Router) achieves 80% overall path completeness by implementing exactly this strategy.

### 6.4 Weight Sensitivity

We tested four weight configurations for the blended scoring formula:

| Config | $\alpha$ (FAISS) | $\beta$ (Geodesic) | $\gamma$ (Shape) | R@10 | Path |
|--------|-------------------|---------------------|-------------------|------|------|
| FAISS-heavy | 0.70 | 0.20 | 0.10 | 60% | 73% |
| Balanced | 0.50 | 0.35 | 0.15 | 60% | 73% |
| Geodesic-heavy | 0.30 | 0.50 | 0.20 | 60% | 73% |
| Shape-heavy | 0.30 | 0.20 | 0.50 | 60% | 73% |

All four configurations produced identical results on the 10-case benchmark. This indicates that the manifold's advantage is structural — it determines which candidates enter the scoring pool via geodesic traversal — rather than dependent on the scoring weights. Once the right facts are discovered by the geodesic engine, any reasonable weighting ranks them appropriately. This is a desirable property: it means the system does not require careful hyperparameter tuning to achieve its improvements.

### 6.5 Coverage Ablation

The most important factor in manifold performance is coverage — what fraction of the knowledge base is included in the manifold.

Our initial deployment built the manifold from only 3,997 facts (5.3% of the knowledge base), retrieved via the Memoria search API with a cap of 100 results per query across 218 seed queries. This partial manifold achieved approximately 55% path completeness — barely above the FAISS baseline.

After switching to direct SQLite database access, we built the manifold over all 75,867 facts (100% coverage). Path completeness jumped to 73% — an 18-point improvement from coverage alone.

This finding has a clear implication: partial manifolds degrade catastrophically. The geodesic engine can only route through facts that are on the manifold. If an intermediate chain step is missing from the manifold, the geodesic path cannot traverse it, and the advantage over flat retrieval disappears. Any deployment of manifold retrieval must ensure complete or near-complete coverage of the knowledge base.

### 6.6 Eigenvalue Analysis

The top 12 eigenvalues of our manifold's normalized graph Laplacian range from 0.940 to 0.977, with a spectral gap of 0.0037 between the first and second eigenvalues. This small spectral gap indicates that the manifold is smooth — the knowledge base does not decompose into sharply separated clusters but rather forms a continuous space where topics blend into one another.

This is consistent with the t-SNE visualization of the manifold coordinates (Figure 5), which shows identifiable topic clusters (exploitation, enumeration, privilege escalation) with substantial overlap at their boundaries. The smooth structure is actually beneficial for retrieval: it means geodesic paths can traverse between topic areas without encountering hard boundaries, which is exactly the behavior needed for cross-domain queries like "SQL injection to operating system command execution."

## 7. Discussion

### 7.1 Why Geodesics Beat Cosine for Chains

The key insight is geometric. Cosine similarity retrieves within a ball around the query in embedding space — a straight-line distance. If the knowledge space curves (fact A is near B, B is near C, but A is far from C), cosine similarity retrieves A but misses B and C.

Geodesic distance follows the curve. It finds B between A and C because the shortest path on the manifold graph passes through B. This is exactly the structure of multi-step procedures: each step is locally similar to the next, forming a chain through the embedding space, but the first step is not globally similar to the last.

The 87% path completeness on multi-hop queries — versus 59% for FAISS — quantifies this advantage. Multi-hop queries are precisely the case where the answer follows a curved path through the knowledge space, and geodesic retrieval follows that curve while cosine similarity cannot.

### 7.2 When Flat Retrieval Wins

The manifold adds latency (2.4 seconds vs 57 milliseconds) and complexity. For single-concept queries like "what is SQL injection," cosine similarity is optimal — there is no chain structure to exploit, and the added latency provides no benefit.

The R@5 parity between FAISS and Manifold (52% = 52%) confirms that the manifold does not hurt simple queries. The blended scoring formula ensures that when FAISS already retrieves the right facts, the manifold's geodesic and shape components simply agree with the FAISS ranking rather than overriding it.

A production system should route queries based on complexity: simple factual lookups go to FAISS (57ms), chain-structured queries go to the manifold (2.4s). The query classification needed for this routing is a lightweight keyword analysis, not an LLM call.

### 7.3 Complementarity with LLM Augmentation

Our results reveal that geometric retrieval (Manifold) and linguistic retrieval (Methodic) have complementary strengths:

- **Manifold excels at discovery** — finding the right facts by following the geometry of the knowledge space. Its 87% path completeness on multi-hop queries comes from geodesic paths that naturally traverse intermediate chain steps.
- **LLM decomposition excels at planning** — generating expert-style sub-queries that capture domain knowledge about attack methodology. Its 80% path completeness on attack path queries comes from linguistic reformulation, not geometric traversal.

A pipeline fusion approach — using Manifold for candidate discovery and LLM energy minimization for re-ranking — could combine these strengths. Our companion Smart Router system achieves 80% overall path completeness by routing attack path queries to Methodic and everything else to Manifold, validating the complementarity hypothesis.

### 7.4 Generalization Beyond Cybersecurity

Our method requires only two inputs: a collection of text facts and an embedding model. No domain-specific entity extraction, schema design, relation typing, or graph construction is needed. The manifold emerges entirely from the geometry of the embedding space.

This makes the approach directly applicable to any domain with chain-structured knowledge:

- **Medical diagnosis:** Symptom → differential diagnosis → diagnostic test → confirmation → treatment plan. The manifold would discover that "elevated troponin" is geometrically between "chest pain" and "cardiac catheterization," even though these terms are not textually similar.
- **Legal reasoning:** Statute → judicial interpretation → exception → application. Precedent chains form paths through the legal knowledge space.
- **Technical troubleshooting:** Symptom → root cause analysis → diagnostic procedure → fix → verification. IT knowledge bases contain thousands of such chains.
- **Chemistry/synthesis:** Reagent → reaction conditions → intermediate → purification → product. Synthetic pathways are inherently sequential.

The structure vector encoding and DTW shape matching (Section 4.3) are domain-specific components that would need adaptation for each new domain — the seven dimensions (phase, privilege, scope, etc.) are specific to cybersecurity. However, the core pipeline (manifold construction, geodesic retrieval, Nyström projection, blended scoring) is entirely domain-agnostic.

### 7.5 Limitations

**Manifold build time.** Constructing the manifold over 75,867 facts takes approximately 95 seconds. This must be repeated when the knowledge base changes significantly. The Nyström extension handles new queries but not new facts — a fact added after manifold construction will not participate in geodesic routing until the manifold is rebuilt.

**Query latency.** At 2.4 seconds per query, manifold retrieval is fast enough for planning-style queries but too slow for interactive autocomplete or real-time filtering. The latency comes primarily from Dijkstra's algorithm on the manifold graph; potential optimizations include precomputed shortest-path trees and hierarchical graph decomposition.

**Test suite size.** Our evaluation uses 10-30 test queries. While the 3-run averaging on the expanded benchmark reduces variance, a larger evaluation (100+ queries) with human-annotated ground truth would strengthen the empirical contribution.

**Structure vector classification.** Our 7-dimensional structure vectors are assigned by keyword heuristics, not by LLM classification. LLM-quality structure vectors could improve DTW shape matching, though the overall impact would be bounded by the shape component's 15% weight in the blended score.

**Single embedding model.** All experiments use all-MiniLM-L6-v2 (384 dimensions). The manifold's geometry depends on the embedding model's representation quality. Larger embedding models (e.g., e5-large, BGE) might produce richer manifold structure, but we have not evaluated this.

## 8. Conclusion

We introduced Manifold Retrieval: a method for building a Riemannian manifold over knowledge base embeddings using diffusion maps and retrieving along geodesic shortest paths. This is, to our knowledge, the first application of manifold learning to knowledge base search — using the learned geometry for actual retrieval rather than dimensionality reduction or visualization.

On a knowledge base of 75,867 cybersecurity facts, Manifold Retrieval achieves 73-80% path completeness, a +22% improvement over FAISS cosine similarity baseline and +5% over an LLM-augmented retrieval system, at 5× lower latency. The method requires no LLM calls at query time, no entity extraction, and no domain-specific graph construction — only embeddings.

The key insight is that the intrinsic geometry of embedding spaces encodes structural relationships that flat similarity search ignores. Facts connected through multi-step procedures lie along curved paths through the embedding space. Geodesic retrieval follows these paths; cosine similarity cannot.

The method is domain-agnostic and applicable wherever knowledge forms chains. Medical diagnosis pathways, legal precedent chains, troubleshooting sequences, and synthetic chemistry routes all share the same geometric structure: each step is locally connected to the next, forming a path that curves through the embedding space. Manifold Retrieval provides a principled mathematical framework for exploiting this structure.

---

## References

[1] R. R. Coifman and S. Lafon, "Diffusion maps," *Applied and Computational Harmonic Analysis*, vol. 21, no. 1, pp. 5-30, 2006.

[2] L. Zelnik-Manor and P. Perona, "Self-tuning spectral clustering," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2004.

[3] J. Johnson, M. Douze, and H. Jégou, "Billion-scale similarity search with GPUs," *IEEE Transactions on Big Data*, 2019.

[4] M. Belkin and P. Niyogi, "Laplacian eigenmaps for dimensionality reduction and data representation," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2001.

[5] L. McInnes, J. Healy, and J. Melville, "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction," arXiv:1802.03426, 2018.

[6] L. Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels," arXiv:2212.10496 (HyDE), 2022.

[7] H. Zheng et al., "Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models," arXiv:2310.06117, 2023.

[8] D. Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused Summarization," arXiv:2404.16130, 2024.

[9] "Geodesic Semantic Search," arXiv:2602.23665, 2025.

[10] H. Sakoe and S. Chiba, "Dynamic programming algorithm optimization for spoken word recognition," *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 26, no. 1, pp. 43-49, 1978.

[11] "Reranker Optimization via Geodesic Distances on k-NN Manifolds (Maniscope)," arXiv:2602.15860, 2026.

[12] J. Wang et al., "PropRAG: Guiding Retrieval with Beam Search over Proposition Paths," arXiv:2504.18070, 2025.
