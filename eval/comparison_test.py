#!/usr/bin/env python3
"""
Comparison Test — FAISS vs Methodic v1.5 vs Manifold Retrieval

Uses the same 10 test cases from methodic_trial.py to compare:
  A) Raw FAISS (Memoria /search)
  B) Methodic v1.5 (smart-search with LLM intents + edges + energy)
  C) Manifold Retrieval (geodesic + shape matching + structure)

Metrics:
  - Recall@5: fraction of expected keywords in top 5
  - Recall@10: fraction of expected keywords in top 10
  - Path Completeness: fraction of attack chain steps covered
  - Latency: milliseconds per query

Run from jagg against local services:
  - Memoria: http://127.0.0.1:8000
  - Methodic: http://127.0.0.1:8002
  - Manifold: http://127.0.0.1:8003
"""

import json
import sys
import time
from dataclasses import dataclass
from textwrap import shorten
from urllib.request import Request, urlopen
from urllib.error import URLError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MEMORIA_URL = "http://127.0.0.1:8000"
METHODIC_URL = "http://127.0.0.1:8002"
MANIFOLD_URL = "http://127.0.0.1:8003"

TIMEOUT = 120

ENGINES = {
    "faiss": {"url": MEMORIA_URL, "endpoint": "/search", "payload_key": "query"},
    "methodic": {"url": METHODIC_URL, "endpoint": "/smart-search", "payload_key": "query"},
    "manifold": {"url": MANIFOLD_URL, "endpoint": "/manifold-search", "payload_key": "query"},
}

# ---------------------------------------------------------------------------
# Test cases (same as methodic_trial.py)
# ---------------------------------------------------------------------------
@dataclass
class TestCase:
    id: str
    category: str
    query: str
    goal: str               # explicit goal for manifold retrieval
    expected_keywords: list
    attack_chain: list

TEST_CASES = [
    TestCase(
        id="CB-01", category="constraint_bypass",
        query="command injection when spaces are filtered",
        goal="achieve command execution without literal spaces",
        expected_keywords=["${IFS}", "{cat,/etc/passwd}", "brace expansion", "$IFS$9",
                           "tab", "%09", "command injection"],
        attack_chain=["identify space filtering in input",
                      "use ${IFS} or $IFS$9 as space substitute",
                      "construct payload with brace expansion {cmd,arg}",
                      "achieve command execution without literal spaces"],
    ),
    TestCase(
        id="CB-02", category="constraint_bypass",
        query="exfiltrate data from blind command injection with no output",
        goal="extract data from target via out-of-band channel",
        expected_keywords=["out-of-band", "DNS exfiltration", "curl", "wget",
                           "ping", "sleep", "time-based", "blind"],
        attack_chain=["confirm blind injection via time delay (sleep)",
                      "set up OOB listener (netcat/burp collaborator)",
                      "exfiltrate via DNS subdomain encoding",
                      "or exfiltrate via HTTP request to attacker server"],
    ),
    TestCase(
        id="CB-03", category="constraint_bypass",
        query="bypass WAF blocking slashes and dots in path traversal",
        goal="read sensitive files despite WAF filtering",
        expected_keywords=["double encoding", "%252e%252e", "..%c0%af", "path traversal",
                           "null byte", "%00", "overlong UTF-8", "WAF bypass"],
        attack_chain=["identify WAF blocking ../ patterns",
                      "attempt URL encoding (%2e%2e%2f)",
                      "attempt double encoding (%252e%252e%252f)",
                      "attempt overlong UTF-8 encoding (..%c0%af)",
                      "read sensitive file"],
    ),
    TestCase(
        id="MH-01", category="multi_hop",
        query="Apache Tomcat manager to reverse shell",
        goal="get a reverse shell on the server running Tomcat",
        expected_keywords=["Tomcat", "manager", "WAR", "msfvenom", "deploy",
                           "reverse shell", "default credentials", "/manager/html"],
        attack_chain=["discover Tomcat manager interface",
                      "authenticate with default/brute-forced credentials",
                      "generate malicious WAR file (msfvenom)",
                      "deploy WAR via manager upload",
                      "trigger reverse shell by accessing deployed app"],
    ),
    TestCase(
        id="MH-02", category="multi_hop",
        query="SUID binary to root privilege escalation on Linux",
        goal="escalate from user to root via SUID binary",
        expected_keywords=["SUID", "find / -perm", "GTFOBins", "privilege escalation",
                           "/bin/bash -p", "nmap --interactive", "cp /bin/bash", "chmod +s"],
        attack_chain=["enumerate SUID binaries (find / -perm -4000)",
                      "identify exploitable SUID binary (GTFOBins)",
                      "abuse SUID binary to spawn root shell",
                      "verify root access"],
    ),
    TestCase(
        id="MH-03", category="multi_hop",
        query="Redis unauthenticated access to SSH key injection to shell",
        goal="get SSH shell access via Redis key injection",
        expected_keywords=["Redis", "CONFIG SET dir", "CONFIG SET dbfilename",
                           "authorized_keys", "SSH", "redis-cli", ".ssh"],
        attack_chain=["connect to unauthenticated Redis",
                      "set dir to /root/.ssh or /home/user/.ssh",
                      "set dbfilename to authorized_keys",
                      "write SSH public key as Redis value",
                      "SAVE and SSH in with corresponding private key"],
    ),
    TestCase(
        id="CC-01", category="cross_context",
        query="living off the land binaries for file download on Windows",
        goal="download files to Windows target using built-in binaries",
        expected_keywords=["certutil", "bitsadmin", "PowerShell", "Invoke-WebRequest",
                           "LOLBAS", "LOLBin", "download", "transfer"],
        attack_chain=["identify need to download file to Windows target",
                      "use certutil -urlcache -split -f to download",
                      "or use bitsadmin /transfer",
                      "or use PowerShell Invoke-WebRequest / wget alias",
                      "execute downloaded payload"],
    ),
    TestCase(
        id="CC-02", category="cross_context",
        query="SQL injection to operating system command execution",
        goal="get OS command execution from SQL injection",
        expected_keywords=["xp_cmdshell", "INTO OUTFILE", "LOAD_FILE", "COPY TO",
                           "stacked queries", "SQL injection", "os-shell", "sqlmap --os-shell"],
        attack_chain=["identify SQL injection vulnerability",
                      "determine database type (MSSQL/MySQL/Postgres)",
                      "escalate to OS command execution via DB-specific method",
                      "achieve command execution on underlying OS"],
    ),
    TestCase(
        id="AP-01", category="attack_path",
        query="Kerberoasting to domain admin in Active Directory",
        goal="escalate from domain user to domain admin via Kerberoasting",
        expected_keywords=["Kerberoast", "GetUserSPNs", "TGS", "hashcat", "SPN",
                           "service account", "domain admin", "impacket"],
        attack_chain=["enumerate SPNs with domain user credentials",
                      "request TGS tickets for service accounts (GetUserSPNs.py)",
                      "extract ticket hashes",
                      "crack TGS hashes offline (hashcat mode 13100)",
                      "use cracked service account password to escalate"],
    ),
    TestCase(
        id="AP-02", category="attack_path",
        query="man in the middle ARP spoofing to credential capture",
        goal="capture credentials by intercepting network traffic via ARP spoofing",
        expected_keywords=["ARP spoofing", "arpspoof", "ettercap", "bettercap", "MITM",
                           "credential", "sniff", "IP forwarding"],
        attack_chain=["enable IP forwarding on attacker machine",
                      "perform ARP spoofing (arpspoof/ettercap/bettercap)",
                      "intercept network traffic between victim and gateway",
                      "capture credentials from HTTP/FTP/other cleartext protocols"],
    ),
]


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------
def search(engine_name: str, query: str, goal: str | None = None,
           top_k: int = 10) -> tuple[list[dict], float]:
    """
    Search an engine and return (results, latency_ms).
    """
    cfg = ENGINES[engine_name]
    payload = {"query": query, "top_k": top_k}

    # Manifold engine supports goal parameter
    if engine_name == "manifold" and goal:
        payload["goal"] = goal

    data_bytes = json.dumps(payload).encode()
    req = Request(
        f"{cfg['url']}{cfg['endpoint']}",
        data=data_bytes,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    try:
        with urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  [!] {engine_name} failed: {e}")
        return [], 0.0
    latency = (time.time() - t0) * 1000

    # Normalize results
    results = data.get("results", [])
    normalized = []
    for r in results:
        if isinstance(r, str):
            normalized.append({"text": r})
        elif isinstance(r, dict):
            text = r.get("text") or r.get("fact") or r.get("content") or str(r)
            normalized.append({"text": text, **r})
    return normalized, latency


def extract_text_blob(results: list[dict]) -> str:
    return "\n".join(r.get("text", "") for r in results).lower()


def score_recall(results: list[dict], keywords: list[str], k: int) -> float:
    if not keywords:
        return 0.0
    blob = extract_text_blob(results[:k])
    return sum(1 for kw in keywords if kw.lower() in blob) / len(keywords)


def score_path_completeness(results: list[dict], chain: list[str]) -> float:
    if not chain:
        return 0.0
    blob = extract_text_blob(results[:10])
    covered = 0
    for step in chain:
        words = [w.lower() for w in step.split() if len(w) > 3]
        if not words:
            covered += 1
            continue
        if sum(1 for w in words if w in blob) / len(words) >= 0.5:
            covered += 1
    return covered / len(chain)


# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------
@dataclass
class Result:
    test_id: str
    engine: str
    recall_5: float
    recall_10: float
    path_completeness: float
    latency_ms: float
    keywords_found: list
    keywords_missing: list
    chain_covered: int
    chain_total: int


def evaluate(tc: TestCase, engine_name: str) -> Result:
    """Evaluate a single test case against an engine."""
    goal = tc.goal if engine_name == "manifold" else None
    results, latency = search(engine_name, tc.query, goal=goal, top_k=10)

    r5 = score_recall(results, tc.expected_keywords, 5)
    r10 = score_recall(results, tc.expected_keywords, 10)
    pc = score_path_completeness(results, tc.attack_chain)

    blob = extract_text_blob(results[:10])
    found = [kw for kw in tc.expected_keywords if kw.lower() in blob]
    missing = [kw for kw in tc.expected_keywords if kw.lower() not in blob]

    chain_covered = 0
    for step in tc.attack_chain:
        words = [w.lower() for w in step.split() if len(w) > 3]
        if not words or sum(1 for w in words if w in blob) / len(words) >= 0.5:
            chain_covered += 1

    return Result(
        test_id=tc.id, engine=engine_name,
        recall_5=r5, recall_10=r10, path_completeness=pc,
        latency_ms=latency,
        keywords_found=found, keywords_missing=missing,
        chain_covered=chain_covered, chain_total=len(tc.attack_chain),
    )


def check_engine(name: str) -> bool:
    """Check if an engine is reachable."""
    cfg = ENGINES[name]
    try:
        req = Request(f"{cfg['url']}/health")
        with urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def run_comparison(engines: list[str] | None = None, cases: list[str] | None = None,
                    runs: int = 1):
    """
    Run full comparison test.

    Args:
        engines: which engines to test (default: all available)
        cases: which test case IDs to run (default: all)
        runs: number of runs to average (handles LLM variability)
    """
    # Check which engines are available
    if engines is None:
        engines = [name for name in ENGINES if check_engine(name)]
        if not engines:
            print("No engines reachable!")
            return

    print(f"Engines available: {engines}")

    selected = TEST_CASES
    if cases:
        selected = [tc for tc in TEST_CASES if tc.id in cases]

    print(f"Test cases: {len(selected)}")
    print(f"Runs per config: {runs}")
    print()

    # Collect results
    all_results = {eng: [] for eng in engines}

    for run_idx in range(runs):
        if runs > 1:
            print(f"\n--- Run {run_idx + 1}/{runs} ---")

        for i, tc in enumerate(selected):
            print(f"[{i+1}/{len(selected)}] {tc.id}: {tc.query[:50]}...")
            for eng in engines:
                r = evaluate(tc, eng)
                all_results[eng].append(r)
                print(f"  {eng:>10}: R@5={r.recall_5:.0%} R@10={r.recall_10:.0%} "
                      f"Path={r.path_completeness:.0%} {r.latency_ms:.0f}ms")

    # Average across runs
    avg_results = {}
    for eng in engines:
        eng_results = all_results[eng]
        n_cases = len(selected)
        per_case = {}
        for r in eng_results:
            if r.test_id not in per_case:
                per_case[r.test_id] = []
            per_case[r.test_id].append(r)

        avg_results[eng] = {}
        for tid, rs in per_case.items():
            avg_results[eng][tid] = {
                "recall_5": sum(r.recall_5 for r in rs) / len(rs),
                "recall_10": sum(r.recall_10 for r in rs) / len(rs),
                "path_completeness": sum(r.path_completeness for r in rs) / len(rs),
                "latency_ms": sum(r.latency_ms for r in rs) / len(rs),
            }

    # Print comparison table
    print("\n" + "=" * 120)
    print("COMPARISON: FAISS vs METHODIC vs MANIFOLD RETRIEVAL")
    print("=" * 120)

    # Header
    eng_headers = ""
    for eng in engines:
        eng_headers += f"  {'R@5':>5} {'R@10':>5} {'Path':>5} {'ms':>5}"
    print(f"{'ID':<6} {'Category':<18}{eng_headers}")
    print("-" * 120)

    # Per-case rows
    totals = {eng: {"r5": 0, "r10": 0, "pc": 0, "ms": 0} for eng in engines}
    n_cases = len(selected)

    for tc in selected:
        row = f"{tc.id:<6} {tc.category:<18}"
        for eng in engines:
            avg = avg_results[eng].get(tc.id, {})
            r5 = avg.get("recall_5", 0)
            r10 = avg.get("recall_10", 0)
            pc = avg.get("path_completeness", 0)
            ms = avg.get("latency_ms", 0)
            row += f"  {r5:>4.0%}  {r10:>4.0%}  {pc:>4.0%} {ms:>5.0f}"
            totals[eng]["r5"] += r5
            totals[eng]["r10"] += r10
            totals[eng]["pc"] += pc
            totals[eng]["ms"] += ms
        print(row)

    # Average row
    print("-" * 120)
    row = f"{'AVG':<6} {'':18}"
    for eng in engines:
        t = totals[eng]
        row += f"  {t['r5']/n_cases:>4.0%}  {t['r10']/n_cases:>4.0%}  {t['pc']/n_cases:>4.0%} {t['ms']/n_cases:>5.0f}"
    print(row)
    print("=" * 120)

    # Delta summary
    if "faiss" in engines and len(engines) > 1:
        print("\nDeltas vs FAISS baseline:")
        for eng in engines:
            if eng == "faiss":
                continue
            dr5 = totals[eng]["r5"]/n_cases - totals["faiss"]["r5"]/n_cases
            dr10 = totals[eng]["r10"]/n_cases - totals["faiss"]["r10"]/n_cases
            dpc = totals[eng]["pc"]/n_cases - totals["faiss"]["pc"]/n_cases
            sign = lambda v: f"+{v:.0%}" if v > 0 else f"{v:.0%}"
            print(f"  {eng}: R@5 {sign(dr5)}, R@10 {sign(dr10)}, Path {sign(dpc)}")

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "engines": engines,
        "runs": runs,
        "averages": {eng: {
            "recall_5": totals[eng]["r5"] / n_cases,
            "recall_10": totals[eng]["r10"] / n_cases,
            "path_completeness": totals[eng]["pc"] / n_cases,
            "latency_ms": totals[eng]["ms"] / n_cases,
        } for eng in engines},
        "per_case": avg_results,
        "raw_results": {
            eng: [{
                "test_id": r.test_id,
                "recall_5": r.recall_5,
                "recall_10": r.recall_10,
                "path_completeness": r.path_completeness,
                "latency_ms": r.latency_ms,
                "keywords_found": r.keywords_found,
                "keywords_missing": r.keywords_missing,
            } for r in results]
            for eng, results in all_results.items()
        },
    }
    with open("comparison_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to comparison_results.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare FAISS vs Methodic vs Manifold")
    parser.add_argument("--engines", nargs="+", default=None,
                        help="Engines to test (faiss, methodic, manifold)")
    parser.add_argument("--cases", nargs="+", default=None,
                        help="Test case IDs to run")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of runs to average")
    parser.add_argument("--faiss-only", action="store_true",
                        help="Only test FAISS baseline")
    parser.add_argument("--manifold-only", action="store_true",
                        help="Only test manifold retrieval")
    args = parser.parse_args()

    if args.faiss_only:
        engines = ["faiss"]
    elif args.manifold_only:
        engines = ["manifold"]
    else:
        engines = args.engines

    run_comparison(engines=engines, cases=args.cases, runs=args.runs)
