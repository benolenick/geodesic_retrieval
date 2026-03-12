#!/usr/bin/env python3
"""
Weight Sweep — test different scoring weight configurations on Manifold Retrieval.

For each weight config, sends all 10 original test queries to the manifold server,
computes recall@10 and path completeness, then saves all results.

Requires the manifold server to have the /set-weights endpoint.
"""

import json
import time
from urllib.request import Request, urlopen

MANIFOLD_URL = "http://127.0.0.1:8003"
TIMEOUT = 120

WEIGHT_CONFIGS = [
    {"alpha": 0.7, "beta": 0.2, "gamma": 0.1, "label": "faiss-heavy"},
    {"alpha": 0.5, "beta": 0.35, "gamma": 0.15, "label": "balanced (default)"},
    {"alpha": 0.3, "beta": 0.5, "gamma": 0.2, "label": "geodesic-heavy"},
    {"alpha": 0.2, "beta": 0.6, "gamma": 0.2, "label": "geodesic-dominant"},
]

# Original 10 test queries with keywords and chains for scoring
TEST_QUERIES = [
    {"id": "CB-01", "query": "command injection when spaces are filtered",
     "goal": "achieve command execution without literal spaces",
     "keywords": ["${IFS}", "{cat,/etc/passwd}", "brace expansion", "$IFS$9", "tab", "%09", "command injection"],
     "chain": ["identify space filtering in input", "use ${IFS} or $IFS$9 as space substitute", "construct payload with brace expansion {cmd,arg}", "achieve command execution without literal spaces"]},
    {"id": "CB-02", "query": "exfiltrate data from blind command injection with no output",
     "goal": "extract data from target via out-of-band channel",
     "keywords": ["out-of-band", "DNS exfiltration", "curl", "wget", "ping", "sleep", "time-based", "blind"],
     "chain": ["confirm blind injection via time delay (sleep)", "set up OOB listener (netcat/burp collaborator)", "exfiltrate via DNS subdomain encoding", "or exfiltrate via HTTP request to attacker server"]},
    {"id": "CB-03", "query": "bypass WAF blocking slashes and dots in path traversal",
     "goal": "read sensitive files despite WAF filtering",
     "keywords": ["double encoding", "%252e%252e", "..%c0%af", "path traversal", "null byte", "%00", "overlong UTF-8", "WAF bypass"],
     "chain": ["identify WAF blocking ../ patterns", "attempt URL encoding (%2e%2e%2f)", "attempt double encoding (%252e%252e%252f)", "attempt overlong UTF-8 encoding (..%c0%af)", "read sensitive file"]},
    {"id": "MH-01", "query": "Apache Tomcat manager to reverse shell",
     "goal": "get a reverse shell on the server running Tomcat",
     "keywords": ["Tomcat", "manager", "WAR", "msfvenom", "deploy", "reverse shell", "default credentials", "/manager/html"],
     "chain": ["discover Tomcat manager interface", "authenticate with default/brute-forced credentials", "generate malicious WAR file (msfvenom)", "deploy WAR via manager upload", "trigger reverse shell by accessing deployed app"]},
    {"id": "MH-02", "query": "SUID binary to root privilege escalation on Linux",
     "goal": "escalate from user to root via SUID binary",
     "keywords": ["SUID", "find / -perm", "GTFOBins", "privilege escalation", "/bin/bash -p", "nmap --interactive", "cp /bin/bash", "chmod +s"],
     "chain": ["enumerate SUID binaries (find / -perm -4000)", "identify exploitable SUID binary (GTFOBins)", "abuse SUID binary to spawn root shell", "verify root access"]},
    {"id": "MH-03", "query": "Redis unauthenticated access to SSH key injection to shell",
     "goal": "get SSH shell access via Redis key injection",
     "keywords": ["Redis", "CONFIG SET dir", "CONFIG SET dbfilename", "authorized_keys", "SSH", "redis-cli", ".ssh"],
     "chain": ["connect to unauthenticated Redis", "set dir to /root/.ssh or /home/user/.ssh", "set dbfilename to authorized_keys", "write SSH public key as Redis value", "SAVE and SSH in with corresponding private key"]},
    {"id": "CC-01", "query": "living off the land binaries for file download on Windows",
     "goal": "download files to Windows target using built-in binaries",
     "keywords": ["certutil", "bitsadmin", "PowerShell", "Invoke-WebRequest", "LOLBAS", "LOLBin", "download", "transfer"],
     "chain": ["identify need to download file to Windows target", "use certutil -urlcache -split -f to download", "or use bitsadmin /transfer", "or use PowerShell Invoke-WebRequest / wget alias", "execute downloaded payload"]},
    {"id": "CC-02", "query": "SQL injection to operating system command execution",
     "goal": "get OS command execution from SQL injection",
     "keywords": ["xp_cmdshell", "INTO OUTFILE", "LOAD_FILE", "COPY TO", "stacked queries", "SQL injection", "os-shell", "sqlmap --os-shell"],
     "chain": ["identify SQL injection vulnerability", "determine database type (MSSQL/MySQL/Postgres)", "escalate to OS command execution via DB-specific method", "achieve command execution on underlying OS"]},
    {"id": "AP-01", "query": "Kerberoasting to domain admin in Active Directory",
     "goal": "escalate from domain user to domain admin via Kerberoasting",
     "keywords": ["Kerberoast", "GetUserSPNs", "TGS", "hashcat", "SPN", "service account", "domain admin", "impacket"],
     "chain": ["enumerate SPNs with domain user credentials", "request TGS tickets for service accounts (GetUserSPNs.py)", "extract ticket hashes", "crack TGS hashes offline (hashcat mode 13100)", "use cracked service account password to escalate"]},
    {"id": "AP-02", "query": "man in the middle ARP spoofing to credential capture",
     "goal": "capture credentials by intercepting network traffic via ARP spoofing",
     "keywords": ["ARP spoofing", "arpspoof", "ettercap", "bettercap", "MITM", "credential", "sniff", "IP forwarding"],
     "chain": ["enable IP forwarding on attacker machine", "perform ARP spoofing (arpspoof/ettercap/bettercap)", "intercept network traffic between victim and gateway", "capture credentials from HTTP/FTP/other cleartext protocols"]},
]


def set_weights(alpha, beta, gamma):
    """Set scoring weights on the manifold server."""
    payload = json.dumps({"alpha": alpha, "beta": beta, "gamma": gamma}).encode()
    req = Request(f"{MANIFOLD_URL}/set-weights", data=payload,
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def manifold_search(query, goal, top_k=10):
    """Run a manifold search and return results + latency."""
    payload = json.dumps({"query": query, "goal": goal, "top_k": top_k}).encode()
    req = Request(f"{MANIFOLD_URL}/manifold-search", data=payload,
                  headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urlopen(req, timeout=TIMEOUT) as resp:
        data = json.loads(resp.read())
    latency = (time.time() - t0) * 1000
    results = data.get("results", [])
    normalized = []
    for r in results:
        if isinstance(r, str):
            normalized.append({"text": r})
        elif isinstance(r, dict):
            text = r.get("text") or r.get("fact") or r.get("content") or str(r)
            normalized.append({"text": text, **r})
    return normalized, latency, data.get("stats", {})


def extract_text_blob(results):
    return "\n".join(r.get("text", "") for r in results).lower()


def score_recall(results, keywords, k):
    if not keywords:
        return 0.0
    blob = extract_text_blob(results[:k])
    return sum(1 for kw in keywords if kw.lower() in blob) / len(keywords)


def score_path_completeness(results, chain):
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


def main():
    all_results = {}

    for config in WEIGHT_CONFIGS:
        label = config["label"]
        alpha, beta, gamma = config["alpha"], config["beta"], config["gamma"]
        print(f"\n{'='*60}")
        print(f"Config: {label} (alpha={alpha}, beta={beta}, gamma={gamma})")
        print(f"{'='*60}")

        # Set weights
        try:
            resp = set_weights(alpha, beta, gamma)
            print(f"  Weights set: {resp}")
        except Exception as e:
            print(f"  [!] Failed to set weights: {e}")
            continue

        config_results = []
        total_r5 = 0
        total_r10 = 0
        total_pc = 0

        for tq in TEST_QUERIES:
            results, latency, stats = manifold_search(tq["query"], tq["goal"])
            r5 = score_recall(results, tq["keywords"], 5)
            r10 = score_recall(results, tq["keywords"], 10)
            pc = score_path_completeness(results, tq["chain"])
            total_r5 += r5
            total_r10 += r10
            total_pc += pc

            blob = extract_text_blob(results[:10])
            found = [kw for kw in tq["keywords"] if kw.lower() in blob]
            missing = [kw for kw in tq["keywords"] if kw.lower() not in blob]

            config_results.append({
                "test_id": tq["id"],
                "recall_5": r5,
                "recall_10": r10,
                "path_completeness": pc,
                "latency_ms": latency,
                "keywords_found": found,
                "keywords_missing": missing,
            })
            print(f"  {tq['id']}: R@5={r5:.0%} R@10={r10:.0%} Path={pc:.0%} {latency:.0f}ms")

        n = len(TEST_QUERIES)
        avg_r5 = total_r5 / n
        avg_r10 = total_r10 / n
        avg_pc = total_pc / n
        print(f"  --- AVG: R@5={avg_r5:.0%} R@10={avg_r10:.0%} Path={avg_pc:.0%}")

        all_results[label] = {
            "weights": {"alpha": alpha, "beta": beta, "gamma": gamma},
            "averages": {"recall_5": avg_r5, "recall_10": avg_r10, "path_completeness": avg_pc},
            "per_case": config_results,
        }

    # Restore default weights
    try:
        set_weights(0.5, 0.35, 0.15)
        print("\nRestored default weights (0.5, 0.35, 0.15)")
    except Exception:
        pass

    # Save
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "configs_tested": len(all_results),
        "queries_per_config": len(TEST_QUERIES),
        "results": all_results,
    }
    with open("weight_sweep_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved weight_sweep_results.json")

    # Summary table
    print(f"\n{'='*70}")
    print("WEIGHT SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<30} {'R@5':>6} {'R@10':>6} {'Path':>6}")
    print("-" * 70)
    for label, data in all_results.items():
        a = data["averages"]
        print(f"{label:<30} {a['recall_5']:>5.0%}  {a['recall_10']:>5.0%}  {a['path_completeness']:>5.0%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
