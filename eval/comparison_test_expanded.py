#!/usr/bin/env python3
"""
Expanded Comparison Test - FAISS vs Methodic v1.5 vs Manifold Retrieval
30 test cases across 10 categories.
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from urllib.request import Request, urlopen

MEMORIA_URL = os.environ.get("MEMORIA_URL", "http://127.0.0.1:8000")
METHODIC_URL = os.environ.get("METHODIC_URL", "http://127.0.0.1:8002")
MANIFOLD_URL = os.environ.get("MANIFOLD_URL", "http://127.0.0.1:8003")
TIMEOUT = 120

ENGINES = {
    "faiss": {"url": MEMORIA_URL, "endpoint": "/search", "payload_key": "query"},
    "methodic": {"url": METHODIC_URL, "endpoint": "/smart-search", "payload_key": "query"},
    "manifold": {"url": MANIFOLD_URL, "endpoint": "/manifold-search", "payload_key": "query"},
}

@dataclass
class TestCase:
    id: str
    category: str
    query: str
    goal: str
    expected_keywords: list
    attack_chain: list

TEST_CASES = [
    # === ORIGINAL 10 ===
    TestCase(id="CB-01", category="constraint_bypass",
        query="command injection when spaces are filtered",
        goal="achieve command execution without literal spaces",
        expected_keywords=["${IFS}", "{cat,/etc/passwd}", "brace expansion", "$IFS$9", "tab", "%09", "command injection"],
        attack_chain=["identify space filtering in input", "use ${IFS} or $IFS$9 as space substitute", "construct payload with brace expansion {cmd,arg}", "achieve command execution without literal spaces"]),
    TestCase(id="CB-02", category="constraint_bypass",
        query="exfiltrate data from blind command injection with no output",
        goal="extract data from target via out-of-band channel",
        expected_keywords=["out-of-band", "DNS exfiltration", "curl", "wget", "ping", "sleep", "time-based", "blind"],
        attack_chain=["confirm blind injection via time delay (sleep)", "set up OOB listener (netcat/burp collaborator)", "exfiltrate via DNS subdomain encoding", "or exfiltrate via HTTP request to attacker server"]),
    TestCase(id="CB-03", category="constraint_bypass",
        query="bypass WAF blocking slashes and dots in path traversal",
        goal="read sensitive files despite WAF filtering",
        expected_keywords=["double encoding", "%252e%252e", "..%c0%af", "path traversal", "null byte", "%00", "overlong UTF-8", "WAF bypass"],
        attack_chain=["identify WAF blocking ../ patterns", "attempt URL encoding (%2e%2e%2f)", "attempt double encoding (%252e%252e%252f)", "attempt overlong UTF-8 encoding (..%c0%af)", "read sensitive file"]),
    TestCase(id="MH-01", category="multi_hop",
        query="Apache Tomcat manager to reverse shell",
        goal="get a reverse shell on the server running Tomcat",
        expected_keywords=["Tomcat", "manager", "WAR", "msfvenom", "deploy", "reverse shell", "default credentials", "/manager/html"],
        attack_chain=["discover Tomcat manager interface", "authenticate with default/brute-forced credentials", "generate malicious WAR file (msfvenom)", "deploy WAR via manager upload", "trigger reverse shell by accessing deployed app"]),
    TestCase(id="MH-02", category="multi_hop",
        query="SUID binary to root privilege escalation on Linux",
        goal="escalate from user to root via SUID binary",
        expected_keywords=["SUID", "find / -perm", "GTFOBins", "privilege escalation", "/bin/bash -p", "nmap --interactive", "cp /bin/bash", "chmod +s"],
        attack_chain=["enumerate SUID binaries (find / -perm -4000)", "identify exploitable SUID binary (GTFOBins)", "abuse SUID binary to spawn root shell", "verify root access"]),
    TestCase(id="MH-03", category="multi_hop",
        query="Redis unauthenticated access to SSH key injection to shell",
        goal="get SSH shell access via Redis key injection",
        expected_keywords=["Redis", "CONFIG SET dir", "CONFIG SET dbfilename", "authorized_keys", "SSH", "redis-cli", ".ssh"],
        attack_chain=["connect to unauthenticated Redis", "set dir to /root/.ssh or /home/user/.ssh", "set dbfilename to authorized_keys", "write SSH public key as Redis value", "SAVE and SSH in with corresponding private key"]),
    TestCase(id="CC-01", category="cross_context",
        query="living off the land binaries for file download on Windows",
        goal="download files to Windows target using built-in binaries",
        expected_keywords=["certutil", "bitsadmin", "PowerShell", "Invoke-WebRequest", "LOLBAS", "LOLBin", "download", "transfer"],
        attack_chain=["identify need to download file to Windows target", "use certutil -urlcache -split -f to download", "or use bitsadmin /transfer", "or use PowerShell Invoke-WebRequest / wget alias", "execute downloaded payload"]),
    TestCase(id="CC-02", category="cross_context",
        query="SQL injection to operating system command execution",
        goal="get OS command execution from SQL injection",
        expected_keywords=["xp_cmdshell", "INTO OUTFILE", "LOAD_FILE", "COPY TO", "stacked queries", "SQL injection", "os-shell", "sqlmap --os-shell"],
        attack_chain=["identify SQL injection vulnerability", "determine database type (MSSQL/MySQL/Postgres)", "escalate to OS command execution via DB-specific method", "achieve command execution on underlying OS"]),
    TestCase(id="AP-01", category="attack_path",
        query="Kerberoasting to domain admin in Active Directory",
        goal="escalate from domain user to domain admin via Kerberoasting",
        expected_keywords=["Kerberoast", "GetUserSPNs", "TGS", "hashcat", "SPN", "service account", "domain admin", "impacket"],
        attack_chain=["enumerate SPNs with domain user credentials", "request TGS tickets for service accounts (GetUserSPNs.py)", "extract ticket hashes", "crack TGS hashes offline (hashcat mode 13100)", "use cracked service account password to escalate"]),
    TestCase(id="AP-02", category="attack_path",
        query="man in the middle ARP spoofing to credential capture",
        goal="capture credentials by intercepting network traffic via ARP spoofing",
        expected_keywords=["ARP spoofing", "arpspoof", "ettercap", "bettercap", "MITM", "credential", "sniff", "IP forwarding"],
        attack_chain=["enable IP forwarding on attacker machine", "perform ARP spoofing (arpspoof/ettercap/bettercap)", "intercept network traffic between victim and gateway", "capture credentials from HTTP/FTP/other cleartext protocols"]),
    # === 20 NEW TEST CASES ===
    TestCase(id="MH-04", category="multi_hop",
        query="Jenkins Groovy script console to reverse shell",
        goal="achieve remote code execution on Jenkins server via script console",
        expected_keywords=["Jenkins", "Groovy", "script console", "Runtime.exec", "reverse shell", "/script", "ProcessBuilder", "deserialization"],
        attack_chain=["discover Jenkins instance and version", "access /script console with default or brute-forced credentials", "write Groovy reverse shell payload with Runtime.exec", "execute Groovy script to spawn reverse shell", "catch shell on attacker listener"]),
    TestCase(id="MH-05", category="multi_hop",
        query="GitLab CI/CD pipeline exploitation to secret extraction and lateral movement",
        goal="extract secrets from CI/CD pipeline and pivot to production",
        expected_keywords=["GitLab", "CI/CD", ".gitlab-ci.yml", "pipeline", "environment variables", "secrets", "runner", "artifact"],
        attack_chain=["gain access to GitLab repository", "modify .gitlab-ci.yml to inject malicious pipeline stage", "dump CI/CD environment variables and secrets", "use extracted credentials to pivot to production systems", "establish persistence on production host"]),
    TestCase(id="MH-06", category="multi_hop",
        query="WordPress vulnerable plugin exploitation to webshell upload",
        goal="get a webshell on WordPress server through plugin vulnerability",
        expected_keywords=["WordPress", "wp-admin", "plugin", "webshell", "upload", "wpscan", "wp-content", "PHP reverse shell"],
        attack_chain=["enumerate WordPress plugins with wpscan", "identify vulnerable plugin version", "exploit plugin vulnerability for file upload or RCE", "upload PHP webshell to wp-content/uploads", "access webshell for command execution"]),
    TestCase(id="NA-01", category="network_attack",
        query="DNS tunneling for data exfiltration and C2 communication",
        goal="exfiltrate data and maintain C2 channel using DNS queries",
        expected_keywords=["DNS tunneling", "iodine", "dnscat2", "TXT record", "subdomain encoding", "exfiltration", "C2", "covert channel"],
        attack_chain=["set up authoritative DNS server for attacker domain", "install DNS tunneling client on compromised host", "encode data into DNS subdomain queries", "exfiltrate data through recursive DNS resolvers", "establish bidirectional C2 channel over DNS"]),
    TestCase(id="NA-02", category="network_attack",
        query="LLMNR NBT-NS poisoning with Responder to capture NTLMv2 hashes",
        goal="capture NTLMv2 hashes by poisoning name resolution",
        expected_keywords=["LLMNR", "NBT-NS", "Responder", "NTLMv2", "hash", "NetNTLM", "relay", "poisoning"],
        attack_chain=["run Responder on attacker machine on the local network", "wait for LLMNR/NBT-NS broadcast name resolution requests", "poison responses to redirect authentication to attacker", "capture NTLMv2 hashes from authentication attempts", "crack NTLMv2 hashes with hashcat mode 5600"]),
    TestCase(id="NA-03", category="network_attack",
        query="SMB relay attack to execute commands on remote host",
        goal="relay captured NTLM authentication to execute commands on another host",
        expected_keywords=["SMB relay", "ntlmrelayx", "impacket", "SMB signing", "NTLM", "relay", "psexec", "secretsdump"],
        attack_chain=["identify hosts with SMB signing disabled", "set up ntlmrelayx targeting vulnerable hosts", "trigger NTLM authentication from victim", "relay captured NTLM auth to target host", "execute commands or dump SAM database on target"]),
    TestCase(id="NA-04", category="network_attack",
        query="VLAN hopping via double tagging or switch spoofing",
        goal="access traffic on a different VLAN from the attacker VLAN",
        expected_keywords=["VLAN hopping", "802.1Q", "double tagging", "DTP", "trunk", "switch spoofing", "native VLAN", "encapsulation"],
        attack_chain=["identify native VLAN and trunk port configuration", "craft double-tagged 802.1Q frames with target VLAN ID", "or negotiate trunk port via DTP spoofing", "send crafted frames to access target VLAN traffic"]),
    TestCase(id="CL-01", category="cloud",
        query="AWS SSRF to EC2 metadata service to IAM credential theft",
        goal="steal IAM credentials from EC2 instance via SSRF",
        expected_keywords=["SSRF", "169.254.169.254", "metadata", "IAM", "credential", "AWS", "instance role", "IMDSv1"],
        attack_chain=["identify SSRF vulnerability in web application", "access EC2 metadata at http://169.254.169.254/latest/meta-data/", "retrieve IAM role name from metadata", "request temporary credentials from iam/security-credentials/role", "use stolen credentials for AWS API access and privilege escalation"]),
    TestCase(id="CL-02", category="cloud",
        query="Azure AD OAuth token abuse for tenant compromise",
        goal="abuse OAuth tokens to escalate privileges in Azure AD tenant",
        expected_keywords=["Azure AD", "OAuth", "access token", "refresh token", "Microsoft Graph", "consent", "app registration", "tenant"],
        attack_chain=["phish user for OAuth consent to malicious app registration", "obtain access and refresh tokens via authorization code flow", "use Microsoft Graph API with stolen token", "enumerate users groups and roles in the tenant", "escalate to Global Admin or access sensitive data"]),
    TestCase(id="CL-03", category="cloud",
        query="GCP service account key to project takeover",
        goal="escalate from leaked service account key to project-wide access",
        expected_keywords=["GCP", "service account", "gcloud", "IAM", "impersonate", "project", "key file", "roles/owner"],
        attack_chain=["obtain leaked GCP service account JSON key file", "authenticate with gcloud auth activate-service-account", "enumerate IAM permissions with gcloud projects get-iam-policy", "escalate privileges by impersonating other service accounts", "achieve project-wide access or compute instance takeover"]),
    TestCase(id="BX-01", category="binary_exploitation",
        query="stack buffer overflow to shellcode execution bypassing NX",
        goal="achieve code execution via buffer overflow despite NX protection",
        expected_keywords=["buffer overflow", "ROP", "return-oriented programming", "NX", "gadgets", "mprotect", "stack pivot", "shellcode"],
        attack_chain=["identify stack buffer overflow with controlled EIP/RIP", "determine NX is enabled preventing stack shellcode execution", "find ROP gadgets in binary and libraries", "chain ROP gadgets to call mprotect or mmap for executable memory", "pivot to shellcode after ROP chain makes memory executable"]),
    TestCase(id="BX-02", category="binary_exploitation",
        query="format string vulnerability to arbitrary write and code execution",
        goal="achieve arbitrary memory write via format string to hijack control flow",
        expected_keywords=["format string", "%n", "%x", "GOT overwrite", "arbitrary write", "printf", "DTOR", "stack leak"],
        attack_chain=["identify format string vulnerability in printf-like function", "leak stack addresses and canary with %p/%x format specifiers", "calculate target address GOT entry or return address", "use %n writes to overwrite target with shellcode address", "trigger overwritten function pointer for code execution"]),
    TestCase(id="BX-03", category="binary_exploitation",
        query="heap overflow use-after-free exploitation for shell",
        goal="exploit heap corruption to achieve code execution",
        expected_keywords=["heap", "use-after-free", "tcache", "fastbin", "malloc", "free", "double free", "__malloc_hook"],
        attack_chain=["identify heap-based buffer overflow or use-after-free", "trigger controlled allocation and free patterns", "corrupt tcache/fastbin metadata for arbitrary allocation", "overwrite __malloc_hook or __free_hook with one_gadget", "trigger hook for shell execution"]),
    TestCase(id="CB-04", category="constraint_bypass",
        query="SQL injection bypassing WAF with inline comments and encoding",
        goal="achieve SQL injection despite web application firewall rules",
        expected_keywords=["SQL injection", "WAF bypass", "inline comment", "CHAR()", "CONCAT", "hex encoding", "obfuscation", "union select"],
        attack_chain=["identify SQL injection filtered by WAF", "attempt inline comment obfuscation", "use CHAR() or hex encoding to avoid keyword detection", "use case variation and whitespace alternatives", "extract data through obfuscated injection"]),
    TestCase(id="CB-05", category="constraint_bypass",
        query="command injection via environment variables and shell expansion",
        goal="achieve command execution through environment variable manipulation",
        expected_keywords=["environment variable", "LD_PRELOAD", "PATH hijacking", "shell expansion", "backtick", "$()", "injection", "export"],
        attack_chain=["identify application reading user-controlled environment variables", "inject malicious values via PATH hijacking or LD_PRELOAD", "use shell expansion or backticks in variable values", "achieve command execution when variable is evaluated by shell"]),
    TestCase(id="CE-01", category="container_escape",
        query="Docker socket mount container escape to host root",
        goal="escape from container to host via mounted Docker socket",
        expected_keywords=["docker.sock", "/var/run/docker.sock", "container escape", "mount", "privileged", "chroot", "nsenter", "host filesystem"],
        attack_chain=["identify Docker socket mounted inside container", "use docker client inside container to create new privileged container", "mount host root filesystem into new container", "chroot or nsenter into host namespace", "achieve root access on the host"]),
    TestCase(id="CE-02", category="container_escape",
        query="Kubernetes pod escape via service account token and API server",
        goal="escape pod and compromise Kubernetes cluster",
        expected_keywords=["Kubernetes", "service account", "token", "API server", "kubectl", "RBAC", "pod", "secrets"],
        attack_chain=["read mounted service account token from /var/run/secrets/kubernetes.io/", "authenticate to Kubernetes API server with stolen token", "enumerate RBAC permissions for the service account", "create privileged pod or read cluster secrets", "escalate to cluster-admin or escape to node"]),
    TestCase(id="CE-03", category="container_escape",
        query="Linux container escape via cgroup release_agent",
        goal="escape container to execute commands on host via cgroup manipulation",
        expected_keywords=["cgroup", "release_agent", "notify_on_release", "container escape", "mount cgroup", "privileged", "CAP_SYS_ADMIN", "host command"],
        attack_chain=["verify container has CAP_SYS_ADMIN or is privileged", "mount cgroup filesystem inside container", "set release_agent to attacker-controlled script on host", "set notify_on_release to 1", "trigger cgroup release to execute script on host"]),
    TestCase(id="DE-01", category="defense_evasion",
        query="AMSI bypass techniques for PowerShell payload execution",
        goal="bypass Antimalware Scan Interface to execute malicious PowerShell",
        expected_keywords=["AMSI", "AmsiScanBuffer", "bypass", "PowerShell", "reflection", "patch", "obfuscation", "CLR"],
        attack_chain=["identify AMSI blocking PowerShell payload execution", "use reflection to access AmsiScanBuffer in amsi.dll", "patch AmsiScanBuffer return value to always return clean", "or use string obfuscation to avoid AMSI signature detection", "execute malicious PowerShell payload successfully"]),
    TestCase(id="DE-02", category="defense_evasion",
        query="EDR evasion by unhooking ntdll and direct syscalls",
        goal="bypass endpoint detection by removing user-mode API hooks",
        expected_keywords=["EDR", "unhooking", "ntdll", "syscall", "direct syscall", "user-mode hooks", "NtWriteVirtualMemory", "SysWhispers"],
        attack_chain=["identify EDR user-mode hooks on ntdll.dll API functions", "read clean copy of ntdll.dll from disk or KnownDlls", "overwrite hooked ntdll.dll text section with clean copy", "or use direct syscalls to bypass hooks entirely", "execute malicious API calls without EDR detection"]),
]


def search(engine_name, query, goal=None, top_k=10):
    cfg = ENGINES[engine_name]
    payload = {"query": query, "top_k": top_k}
    if engine_name == "manifold" and goal:
        payload["goal"] = goal
    data_bytes = json.dumps(payload).encode()
    req = Request(f"{cfg['url']}{cfg['endpoint']}", data=data_bytes, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  [!] {engine_name} failed: {e}")
        return [], 0.0
    latency = (time.time() - t0) * 1000
    results = data.get("results", [])
    normalized = []
    for r in results:
        if isinstance(r, str):
            normalized.append({"text": r})
        elif isinstance(r, dict):
            text = r.get("text") or r.get("fact") or r.get("content") or str(r)
            normalized.append({"text": text, **r})
    return normalized, latency


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


def evaluate(tc, engine_name):
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
    return Result(test_id=tc.id, engine=engine_name, recall_5=r5, recall_10=r10, path_completeness=pc,
        latency_ms=latency, keywords_found=found, keywords_missing=missing,
        chain_covered=chain_covered, chain_total=len(tc.attack_chain))


def check_engine(name):
    cfg = ENGINES[name]
    try:
        req = Request(f"{cfg['url']}/health")
        with urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def run_comparison(engines=None, cases=None, runs=1):
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
    all_results = {eng: [] for eng in engines}
    for run_idx in range(runs):
        if runs > 1:
            print(f"\n--- Run {run_idx + 1}/{runs} ---")
        for i, tc in enumerate(selected):
            print(f"[{i+1}/{len(selected)}] {tc.id}: {tc.query[:50]}...")
            for eng in engines:
                r = evaluate(tc, eng)
                all_results[eng].append(r)
                print(f"  {eng:>10}: R@5={r.recall_5:.0%} R@10={r.recall_10:.0%} Path={r.path_completeness:.0%} {r.latency_ms:.0f}ms")
    avg_results = {}
    for eng in engines:
        per_case = {}
        for r in all_results[eng]:
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
    n_cases = len(selected)
    print("\n" + "=" * 120)
    print("EXPANDED COMPARISON: FAISS vs METHODIC vs MANIFOLD RETRIEVAL (30 cases)")
    print("=" * 120)
    eng_headers = ""
    for eng in engines:
        eng_headers += f"  {'R@5':>5} {'R@10':>5} {'Path':>5} {'ms':>5}"
    print(f"{'ID':<6} {'Category':<20}{eng_headers}")
    print("-" * 120)
    totals = {eng: {"r5": 0, "r10": 0, "pc": 0, "ms": 0} for eng in engines}
    cat_totals = {}
    for tc in selected:
        row = f"{tc.id:<6} {tc.category:<20}"
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
            if tc.category not in cat_totals:
                cat_totals[tc.category] = {e: {"r5": 0, "r10": 0, "pc": 0, "n": 0} for e in engines}
            cat_totals[tc.category][eng]["r5"] += r5
            cat_totals[tc.category][eng]["r10"] += r10
            cat_totals[tc.category][eng]["pc"] += pc
            cat_totals[tc.category][eng]["n"] += 1
        print(row)
    print("-" * 120)
    row = f"{'AVG':<6} {'':20}"
    for eng in engines:
        t = totals[eng]
        row += f"  {t['r5']/n_cases:>4.0%}  {t['r10']/n_cases:>4.0%}  {t['pc']/n_cases:>4.0%} {t['ms']/n_cases:>5.0f}"
    print(row)
    print("=" * 120)
    print("\nPER-CATEGORY AVERAGES:")
    print("-" * 80)
    for cat in sorted(cat_totals.keys()):
        row = f"  {cat:<20}"
        for eng in engines:
            ct = cat_totals[cat][eng]
            n = ct["n"] or 1
            row += f"  {eng}: R@10={ct['r10']/n:.0%} Path={ct['pc']/n:.0%}"
        print(row)
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
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "engines": engines,
        "runs": runs,
        "n_test_cases": n_cases,
        "averages": {eng: {
            "recall_5": totals[eng]["r5"] / n_cases,
            "recall_10": totals[eng]["r10"] / n_cases,
            "path_completeness": totals[eng]["pc"] / n_cases,
            "latency_ms": totals[eng]["ms"] / n_cases,
        } for eng in engines},
        "per_category": {cat: {eng: {
            "recall_10": cat_totals[cat][eng]["r10"] / max(cat_totals[cat][eng]["n"], 1),
            "path_completeness": cat_totals[cat][eng]["pc"] / max(cat_totals[cat][eng]["n"], 1),
            "n_cases": cat_totals[cat][eng]["n"],
        } for eng in engines} for cat in cat_totals},
        "per_case": avg_results,
        "raw_results": {
            eng: [{"test_id": r.test_id, "recall_5": r.recall_5, "recall_10": r.recall_10,
                "path_completeness": r.path_completeness, "latency_ms": r.latency_ms,
                "keywords_found": r.keywords_found, "keywords_missing": r.keywords_missing,
            } for r in results]
            for eng, results in all_results.items()
        },
    }
    with open("comparison_results_expanded.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to comparison_results_expanded.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Expanded Compare: FAISS vs Methodic vs Manifold")
    parser.add_argument("--engines", nargs="+", default=None)
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--faiss-only", action="store_true")
    parser.add_argument("--manifold-only", action="store_true")
    args = parser.parse_args()
    if args.faiss_only:
        engines = ["faiss"]
    elif args.manifold_only:
        engines = ["manifold"]
    else:
        engines = args.engines
    run_comparison(engines=engines, cases=args.cases, runs=args.runs)
