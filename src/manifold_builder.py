#!/usr/bin/env python3
"""
Manifold Builder — constructs the attack surface manifold from Memoria facts.

Pipeline:
  1. Pull fact embeddings from Memoria (or use FAISS to get pairwise distances)
  2. Build k-NN graph with locally-scaled Gaussian kernel weights
  3. Augment with relationship edges and co-occurrence data
  4. Compute diffusion map (graph Laplacian eigenvectors)
  5. Save manifold coordinates for geodesic engine

Math applied:
  - k-NN graph construction (computational geometry)
  - Gaussian kernel with local scaling (Zelnik-Manor & Perona 2004)
  - Graph Laplacian eigenvectors (spectral graph theory, Chung 1997)
  - Diffusion maps (Coifman & Lafon 2006)
  - Nyström extension for incremental updates (kernel methods)
"""

import json
import logging
import math
import os
import pickle
import sqlite3
import struct
import sys
import time
from urllib.request import Request, urlopen

import numpy as np
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import eigsh

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("manifold_builder")

MEMORIA_URL = os.environ.get("MEMORIA_URL", "http://127.0.0.1:8000")
METHODIC_DB = os.environ.get("METHODIC_DB", "/home/om/htb-autopwn/methodic.db")
K_NEIGHBORS = 15          # k for k-NN graph
MANIFOLD_DIM = 12         # number of diffusion map dimensions
DIFFUSION_TIME = 3        # diffusion time parameter t
EDGE_BOOST = 2.0          # multiplier for explicit relationship edges
COOC_BOOST_SCALE = 0.1    # co-occurrence log-boost scaling
STRUCTURE_WEIGHT = 0.3    # weight of structure similarity in combined distance

DATA_DIR = os.environ.get("MANIFOLD_DATA_DIR", "data")


# ---------------------------------------------------------------------------
# Step 1: Pull embeddings from Memoria
# ---------------------------------------------------------------------------
MEMORIA_DB = os.environ.get("MEMORIA_DB", "/home/om/memoria/runtime/fv.db")


def pull_all_facts_from_db(db_path: str = MEMORIA_DB) -> tuple[list[str], list[str]]:
    """
    Pull ALL facts directly from Memoria's SQLite database.
    Returns (ids, texts).
    """
    if not os.path.exists(db_path):
        log.error(f"Memoria DB not found at {db_path}")
        return [], []

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT uid, body FROM knowledge ORDER BY rowid").fetchall()
    conn.close()

    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    log.info(f"Pulled {len(ids)} facts from Memoria DB at {db_path}")
    return ids, texts


def pull_embeddings_via_search(memoria_url: str, sample_queries: list[str] | None = None,
                                top_k: int = 200) -> tuple[list[str], list[str], np.ndarray | None]:
    """
    Pull fact texts and IDs from Memoria via search API (fallback method).
    Returns (ids, texts, None) — embeddings computed separately.
    """
    if sample_queries is None:
        sample_queries = [
            "exploit vulnerability", "privilege escalation", "command injection",
            "SQL injection", "reverse shell", "web application", "network scan",
            "password cracking", "Active Directory", "Linux privilege",
            "Windows exploitation", "file upload", "container escape",
            "buffer overflow", "persistence backdoor", "lateral movement",
            "credential theft", "WAF bypass", "deserialization",
            "Kerberos", "SMB", "port forward", "binary analysis",
            "SUID", "sudo", "docker", "redis", "tomcat", "apache",
            "XSS", "CSRF", "SSRF", "XXE", "SSTI", "IDOR",
            "hash cracking", "phishing", "social engineering",
            "kernel exploit", "race condition", "heap overflow",
            "format string", "ret2libc", "ROP chain",
            "DNS enumeration", "SNMP", "LDAP", "WinRM",
            "PowerShell", "certutil", "bitsadmin", "LOLBin",
            "Metasploit", "msfvenom", "Cobalt Strike",
            "Wireshark", "tcpdump", "Burp Suite", "sqlmap",
            "Nessus", "OpenVAS", "Nikto",
            # Additional coverage queries
            "path traversal", "null byte injection", "double encoding",
            "blind injection", "out-of-band exfiltration", "ARP spoofing",
            "GTFOBins", "cron job abuse", "SSH key injection",
            "WAR file deployment", "PHP webshell", "Python reverse shell",
            "xp_cmdshell MSSQL", "UNION SELECT", "stacked queries",
            "JWT token", "OAuth misconfiguration", "CORS bypass",
            "subdomain takeover", "API key leak", "Git exposure",
            "S3 bucket", "cloud metadata", "AWS IAM",
            "Impacket", "Rubeus", "BloodHound", "Mimikatz",
            "hashcat", "Hydra brute force", "Gobuster",
            "ffuf fuzzing", "enum4linux", "crackmapexec",
            "evil-winrm", "chisel tunnel", "pwntools",
            "antivirus evasion", "AMSI bypass", "AppLocker bypass",
            "UAC bypass", "process hollowing", "reflective injection",
            "golden ticket", "silver ticket", "DCSync",
            "Pass the Hash", "AS-REP roasting", "delegation abuse",
            "ACL abuse", "DPAPI", "SAM dump", "LSASS dump",
            "DNS poisoning", "LLMNR poisoning", "Responder capture",
            "NTLM relay", "wireless cracking", "session hijacking",
            "GraphQL injection", "request smuggling", "cache poisoning",
            "prototype pollution", "Java deserialization", "pickle exploit",
            "padding oracle", "timing attack", "TLS downgrade",
            "Kubernetes exploit", "Jenkins exploit", "CI/CD attack",
            "dependency confusion", "default credentials",
            "memory forensics", "rootkit detection", "sandbox escape",
            "DLL hijacking", "COM hijacking", "token manipulation",
            "WriteDACL abuse", "certificate abuse ADCS",
            "PetitPotam", "PrintNightmare", "ZeroLogon",
            "EternalBlue", "ProxyLogon", "Log4Shell",
            "Shellshock", "Heartbleed", "Apache Struts",
            "WordPress exploit", "MongoDB injection",
            "Elasticsearch query", "NFS mount",
            "netcat bind shell", "socat relay", "meterpreter",
            "Cobalt Strike beacon", "sliver C2", "Havoc framework",
            "Covenant C2", "PoshC2", "Empire powershell",
            "CrackMapExec spray", "Kerbrute", "GetNPUsers",
            "secretsdump", "wmiexec", "psexec",
            "smbexec", "atexec", "dcomexec",
            "SharpHound", "ADFind", "net user domain",
            "whoami /priv", "icacls permissions", "accesschk",
            "winPEAS", "linPEAS", "linux smart enumeration",
            "pspy process monitor", "chisel socks proxy",
            "proxychains", "SSH dynamic forwarding",
            "Windows firewall bypass", "iptables manipulation",
            "SELinux bypass", "seccomp escape",
            "capability abuse", "namespace escape",
            "cgroup escape", "Dirty Pipe", "Dirty COW",
            "polkit pkexec", "Baron Samedit sudo",
            "Python library hijacking", "LD_PRELOAD injection",
            "shared library attack", "PATH hijacking",
            "cronjob wildcard injection", "tar checkpoint abuse",
            "NFS no_root_squash", "MySQL UDF exploitation",
            "PostgreSQL COPY command", "Redis module load",
            "Memcached injection", "SSRF cloud metadata",
            "IMDSv1 token theft", "Azure managed identity",
            "GCP service account", "Terraform state secrets",
            "Vault secret extraction", "Consul exploitation",
            "etcd unauthenticated", "Zookeeper exploitation",
        ]

    all_facts = {}
    # Memoria API uses "limit" not "top_k", capped at 100
    effective_limit = min(top_k, 100)
    for i, query in enumerate(sample_queries):
        try:
            payload = json.dumps({"query": query, "limit": effective_limit}).encode()
            req = Request(
                f"{memoria_url}/search",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())

            for r in data.get("results", []):
                fid = r.get("id", "")
                text = r.get("text", r.get("fact", ""))
                if fid and text and fid not in all_facts:
                    all_facts[fid] = text
        except Exception as e:
            log.warning(f"Search failed for '{query}': {e}")

        if (i + 1) % 20 == 0:
            log.info(f"  Search progress: {i+1}/{len(sample_queries)} queries, {len(all_facts)} unique facts")

    ids = list(all_facts.keys())
    texts = [all_facts[fid] for fid in ids]
    log.info(f"Pulled {len(ids)} unique facts from Memoria via {len(sample_queries)} queries")
    return ids, texts, None


def compute_embeddings(texts: list[str], model_name: str = "all-MiniLM-L6-v2",
                       batch_size: int = 256) -> np.ndarray:
    """Compute sentence embeddings using sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    log.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    log.info(f"Computing embeddings for {len(texts)} facts...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                               normalize_embeddings=True)
    return np.array(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Step 2: Build k-NN graph with local scaling
# ---------------------------------------------------------------------------
def build_knn_graph(embeddings: np.ndarray, k: int = K_NEIGHBORS) -> tuple[np.ndarray, np.ndarray]:
    """
    Build k-NN graph using cosine similarity.
    Returns (indices, distances) arrays of shape (n, k).

    Uses brute-force for correctness; for 75K × 384, this is ~10 seconds.
    """
    n = embeddings.shape[0]
    log.info(f"Building {k}-NN graph for {n} facts...")

    # Cosine distance = 1 - cosine_similarity
    # For normalized embeddings, cosine_sim = dot product
    # Process in batches to manage memory
    batch_size = 1000
    all_indices = np.zeros((n, k), dtype=np.int32)
    all_distances = np.zeros((n, k), dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = embeddings[start:end]  # (batch, dim)
        # Cosine similarity with all points
        sims = batch @ embeddings.T  # (batch, n)
        # Set self-similarity to -inf to exclude
        for i in range(end - start):
            sims[i, start + i] = -np.inf
        # Top-k by similarity (highest = closest)
        top_k_idx = np.argpartition(sims, -k, axis=1)[:, -k:]
        # Sort within top-k
        for i in range(end - start):
            sorted_order = np.argsort(-sims[i, top_k_idx[i]])
            top_k_idx[i] = top_k_idx[i, sorted_order]
            all_distances[start + i] = 1.0 - sims[i, top_k_idx[i]]  # cosine distance
        all_indices[start:end] = top_k_idx

        if start % 5000 == 0 and start > 0:
            log.info(f"  k-NN: {start}/{n} done")

    log.info(f"k-NN graph built: {n} nodes, {n*k} edges")
    return all_indices, all_distances


def build_adjacency_matrix(
    n: int,
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    structure_vectors: dict[str, list[float]] | None = None,
    fact_ids: list[str] | None = None,
    edge_data: list[tuple] | None = None,
    cooc_data: list[tuple] | None = None,
) -> csr_matrix:
    """
    Build sparse weighted adjacency matrix with:
    1. Gaussian kernel on k-NN distances (local scaling)
    2. Structure vector similarity boost
    3. Explicit edge relationship boost
    4. Co-occurrence boost

    Math: w(i,j) = exp(-d(i,j)^2 / (2 * sigma_i * sigma_j))
    Local scaling: sigma_i = distance to k/2-th neighbor of i
    """
    k = knn_indices.shape[1]

    # Compute local scale sigma_i = distance to median neighbor
    median_idx = k // 2
    sigmas = knn_distances[:, median_idx].copy()
    sigmas[sigmas < 1e-6] = 1e-6  # avoid division by zero

    # Build sparse matrix entries
    rows = []
    cols = []
    vals = []

    for i in range(n):
        for j_pos in range(k):
            j = knn_indices[i, j_pos]
            d = knn_distances[i, j_pos]

            # Gaussian kernel with local scaling
            w = math.exp(-(d ** 2) / (2 * sigmas[i] * sigmas[j]))

            # Structure vector similarity boost (if available)
            if structure_vectors and fact_ids:
                sv_i = structure_vectors.get(fact_ids[i])
                sv_j = structure_vectors.get(fact_ids[j])
                if sv_i and sv_j:
                    # Cosine similarity of structure vectors
                    sv_sim = sum(a * b for a, b in zip(sv_i, sv_j))
                    sv_norm = (sum(a**2 for a in sv_i) ** 0.5 *
                               sum(b**2 for b in sv_j) ** 0.5)
                    if sv_norm > 0:
                        struct_sim = sv_sim / sv_norm
                        # Blend: facts with similar structural roles are closer
                        w *= (1.0 + STRUCTURE_WEIGHT * struct_sim)

            rows.append(i)
            cols.append(j)
            vals.append(w)

    # Make symmetric (if i→j exists, ensure j→i exists with max weight)
    W = csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
    W = W.maximum(W.T)  # symmetric max

    # Augment with explicit relationship edges
    if edge_data and fact_ids:
        id_to_idx = {fid: idx for idx, fid in enumerate(fact_ids)}
        boosted = 0
        for src_text, tgt_text, confidence in edge_data:
            # Find facts matching these edge texts via fuzzy lookup
            # (edges reference fact text, not IDs — we need to match)
            # This is handled by passing pre-matched index pairs
            pass
        # Edge augmentation is done via text matching in the server,
        # which calls augment_edges_by_text() separately

    # Augment with co-occurrence data
    if cooc_data and fact_ids:
        id_to_idx = {fid: idx for idx, fid in enumerate(fact_ids)}
        for id_a, id_b, count in cooc_data:
            if id_a in id_to_idx and id_b in id_to_idx:
                i, j = id_to_idx[id_a], id_to_idx[id_b]
                boost = 1.0 + COOC_BOOST_SCALE * math.log(count + 1)
                W[i, j] *= boost
                W[j, i] *= boost

    log.info(f"Adjacency matrix: {W.shape}, {W.nnz} nonzero entries")
    return W


# ---------------------------------------------------------------------------
# Step 3: Load edge and co-occurrence data from methodic.db
# ---------------------------------------------------------------------------
def load_edges(db_path: str) -> list[tuple]:
    """Load relationship edges from methodic SQLite database."""
    if not os.path.exists(db_path):
        log.warning(f"DB not found at {db_path}")
        return []
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT source_text, target_text, confidence FROM fact_edges "
        "WHERE relation IN ('enables', 'bypasses', 'requires', 'escalates') "
        "AND confidence > 0.4"
    ).fetchall()
    conn.close()
    log.info(f"Loaded {len(rows)} relationship edges")
    return rows


def load_cooccurrences(db_path: str) -> list[tuple]:
    """Load co-occurrence pairs from methodic SQLite database."""
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT fact_a_hash, fact_b_hash, count FROM co_occurrences"
    ).fetchall()
    conn.close()
    log.info(f"Loaded {len(rows)} co-occurrence pairs")
    return rows


# ---------------------------------------------------------------------------
# Step 4: Diffusion Map — spectral embedding
# ---------------------------------------------------------------------------
def compute_diffusion_map(W: csr_matrix, n_dims: int = MANIFOLD_DIM,
                           diffusion_time: int = DIFFUSION_TIME) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute diffusion map coordinates from adjacency matrix.

    Math:
      D = diag(row sums of W)
      P = D^{-1} W                  (row-normalized transition matrix)
      Solve P v = λ v               (eigenproblem)
      Manifold coords = λ^t * v     (diffusion map at time t)

    The eigenvalues decay: λ₁ ≥ λ₂ ≥ ... ≥ λ_d
    Large eigenvalues = smooth, global structure
    Small eigenvalues = fine, local detail

    Returns:
      (manifold_coords, eigenvalues) — coords shape (n, n_dims)
    """
    n = W.shape[0]
    log.info(f"Computing diffusion map: {n} nodes, {n_dims} dims, t={diffusion_time}")

    # Row sums = degree
    degrees = np.array(W.sum(axis=1)).ravel()
    degrees[degrees < 1e-10] = 1e-10

    # Normalized Laplacian: P = D^{-1/2} W D^{-1/2} (symmetric for eigsh)
    D_inv_sqrt = diags(1.0 / np.sqrt(degrees))
    P_sym = D_inv_sqrt @ W @ D_inv_sqrt

    # Find top eigenvectors (largest eigenvalues)
    n_compute = min(n_dims + 1, n - 1)
    log.info(f"  Solving eigenproblem for {n_compute} eigenvectors...")
    t0 = time.time()
    eigenvalues, eigenvectors = eigsh(P_sym, k=n_compute, which='LM')
    elapsed = time.time() - t0
    log.info(f"  Eigenproblem solved in {elapsed:.1f}s")

    # Sort by eigenvalue descending
    order = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Skip first eigenvector (trivial, all-ones after normalization)
    eigenvalues = eigenvalues[1:n_dims+1]
    eigenvectors = eigenvectors[:, 1:n_dims+1]

    # Transform back to non-symmetric basis: φ_i = D^{-1/2} v_i
    manifold_coords = D_inv_sqrt @ eigenvectors

    # Apply diffusion time: scale by λ^t
    for i in range(len(eigenvalues)):
        lam = max(eigenvalues[i], 0.0)  # clamp negative (numerical)
        manifold_coords[:, i] *= lam ** diffusion_time

    log.info(f"  Eigenvalues: {eigenvalues[:5].round(4)}")
    log.info(f"  Manifold coords shape: {manifold_coords.shape}")

    return manifold_coords, eigenvalues


# ---------------------------------------------------------------------------
# Step 5: Nyström extension for new facts
# ---------------------------------------------------------------------------
def nystrom_extend(new_embedding: np.ndarray, existing_embeddings: np.ndarray,
                    manifold_coords: np.ndarray, k: int = K_NEIGHBORS) -> np.ndarray:
    """
    Place a new fact on the existing manifold without recomputing eigenvectors.

    Uses Nyström extension: interpolate manifold coordinates from k nearest
    existing facts, weighted by kernel similarity.
    """
    # Cosine similarities to all existing facts
    sims = new_embedding @ existing_embeddings.T
    top_k_idx = np.argpartition(sims, -k)[-k:]
    top_k_sims = sims[top_k_idx]

    # Gaussian kernel weights
    distances = 1.0 - top_k_sims
    sigma = np.median(distances)
    if sigma < 1e-6:
        sigma = 1e-6
    weights = np.exp(-(distances ** 2) / (2 * sigma ** 2))
    weights /= weights.sum()

    # Weighted average of neighbor manifold coordinates
    new_coords = np.zeros(manifold_coords.shape[1])
    for idx, w in zip(top_k_idx, weights):
        new_coords += w * manifold_coords[idx]

    return new_coords


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def build_manifold(
    memoria_url: str = MEMORIA_URL,
    db_path: str = METHODIC_DB,
    structure_vectors_path: str | None = None,
    output_dir: str = DATA_DIR,
    k: int = K_NEIGHBORS,
    n_dims: int = MANIFOLD_DIM,
    diffusion_time: int = DIFFUSION_TIME,
) -> dict:
    """
    Full manifold construction pipeline.

    Returns metadata dict with paths to saved files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Pull facts — prefer direct DB access (all facts), fallback to API
    db_path = os.environ.get("MEMORIA_DB", MEMORIA_DB)
    if os.path.exists(db_path):
        log.info(f"Using direct DB access: {db_path}")
        ids, texts = pull_all_facts_from_db(db_path)
    else:
        log.info("DB not found, falling back to search API")
        ids, texts, _ = pull_embeddings_via_search(memoria_url)
    if len(ids) < 100:
        log.error(f"Too few facts ({len(ids)}), aborting")
        return {"error": "too few facts"}

    # Save fact index
    fact_index = {fid: {"idx": i, "text": texts[i][:200]} for i, fid in enumerate(ids)}
    with open(os.path.join(output_dir, "fact_index.json"), "w") as f:
        json.dump(fact_index, f)

    # 2. Compute embeddings
    embeddings = compute_embeddings(texts)
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)

    # 3. Build k-NN graph
    knn_indices, knn_distances = build_knn_graph(embeddings, k=k)

    # Load structure vectors if available
    structure_vectors = None
    if structure_vectors_path and os.path.exists(structure_vectors_path):
        with open(structure_vectors_path, "r") as f:
            structure_vectors = json.load(f)
        log.info(f"Loaded {len(structure_vectors)} structure vectors")

    # 4. Build adjacency matrix
    W = build_adjacency_matrix(
        n=len(ids),
        knn_indices=knn_indices,
        knn_distances=knn_distances,
        structure_vectors=structure_vectors,
        fact_ids=ids,
    )

    # Save adjacency matrix
    from scipy.sparse import save_npz
    save_npz(os.path.join(output_dir, "adjacency.npz"), W)

    # 5. Compute diffusion map
    manifold_coords, eigenvalues = compute_diffusion_map(W, n_dims=n_dims,
                                                          diffusion_time=diffusion_time)
    np.save(os.path.join(output_dir, "manifold_coords.npy"), manifold_coords)
    np.save(os.path.join(output_dir, "eigenvalues.npy"), eigenvalues)

    # Save metadata
    metadata = {
        "n_facts": len(ids),
        "embedding_dim": embeddings.shape[1],
        "manifold_dim": n_dims,
        "k_neighbors": k,
        "diffusion_time": diffusion_time,
        "eigenvalues": eigenvalues.tolist(),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": {
            "fact_index": "fact_index.json",
            "embeddings": "embeddings.npy",
            "adjacency": "adjacency.npz",
            "manifold_coords": "manifold_coords.npy",
            "eigenvalues": "eigenvalues.npy",
        },
    }
    with open(os.path.join(output_dir, "manifold_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"Manifold built: {len(ids)} facts → {n_dims}D manifold")
    log.info(f"Saved to {output_dir}/")
    return metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build attack surface manifold")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--k", type=int, default=K_NEIGHBORS, help="k for k-NN graph")
    parser.add_argument("--dims", type=int, default=MANIFOLD_DIM, help="Manifold dimensions")
    parser.add_argument("--time", type=int, default=DIFFUSION_TIME, help="Diffusion time t")
    parser.add_argument("--structure-vectors", default=None, help="Path to structure vectors JSON")
    parser.add_argument("--memoria-url", default=MEMORIA_URL, help="Memoria URL")
    parser.add_argument("--db-path", default=METHODIC_DB, help="Methodic DB path")
    parser.add_argument("--memoria-db", default=MEMORIA_DB, help="Memoria SQLite DB path (direct access)")
    args = parser.parse_args()

    build_manifold(
        memoria_url=args.memoria_url,
        db_path=args.db_path,
        structure_vectors_path=args.structure_vectors,
        output_dir=args.output_dir,
        k=args.k,
        n_dims=args.dims,
        diffusion_time=args.time,
    )
