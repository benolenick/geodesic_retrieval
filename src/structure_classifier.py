#!/usr/bin/env python3
"""
Structure Classifier — assigns 7-dimensional structure vectors to cybersecurity facts.

Each fact gets classified along these axes (0.0 to 1.0):
  s0: attack_phase      (0=recon, 0.25=enum, 0.5=exploit, 0.75=post-exploit, 1.0=exfil)
  s1: privilege_level    (0=none, 0.25=low, 0.5=user, 0.75=admin, 1.0=system/root)
  s2: access_scope       (0=none, 0.25=single service, 0.5=single host, 0.75=subnet, 1.0=domain)
  s3: stealth            (0=very noisy, 0.5=moderate, 1.0=silent/offline)
  s4: interaction        (0=passive/read-only, 0.5=active, 1.0=destructive/write)
  s5: dependency_depth   (0=standalone, 0.5=needs some setup, 1.0=long prerequisite chain)
  s6: target_specificity (0=universal, 0.5=common config, 1.0=exact version only)

Uses qwen3.5 on Ollama for LLM classification with fallback heuristic classifier.
"""

import json
import logging
import os
import re
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import URLError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("structure_classifier")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11435")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:latest")
MEMORIA_URL = os.environ.get("MEMORIA_URL", "http://127.0.0.1:8000")
BATCH_SIZE = 50
SAVE_EVERY = 100

STRUCTURE_DIMS = [
    "attack_phase",
    "privilege_level",
    "access_scope",
    "stealth",
    "interaction",
    "dependency_depth",
    "target_specificity",
]

CLASSIFICATION_PROMPT = """You are a cybersecurity expert classifying facts for a knowledge graph.

Rate this fact on each dimension from 0.0 to 1.0:

FACT: {fact}

Dimensions:
- attack_phase: 0.0=reconnaissance/scanning, 0.25=enumeration, 0.5=exploitation, 0.75=post-exploitation/privesc, 1.0=exfiltration/persistence
- privilege_level: 0.0=no access needed, 0.25=low-priv, 0.5=standard user, 0.75=admin, 1.0=system/root/domain admin
- access_scope: 0.0=no access, 0.25=single service/port, 0.5=single host, 0.75=subnet/network segment, 1.0=full domain/enterprise
- stealth: 0.0=very noisy/detectable, 0.5=moderate, 1.0=silent/offline/undetectable
- interaction: 0.0=passive/read-only, 0.5=active/probing, 1.0=destructive/write/modify
- dependency_depth: 0.0=standalone technique, 0.5=needs some prior access/setup, 1.0=requires long chain of prerequisites
- target_specificity: 0.0=works on anything, 0.5=requires common configuration, 1.0=requires exact software version

Respond with ONLY a JSON array of 7 floats, nothing else. Example: [0.25, 0.0, 0.3, 0.5, 0.4, 0.1, 0.2]"""


# ---------------------------------------------------------------------------
# Heuristic fallback classifier (no LLM needed)
# ---------------------------------------------------------------------------
PHASE_KEYWORDS = {
    0.0: ["nmap", "scan", "recon", "discover", "fingerprint", "port scan", "masscan"],
    0.15: ["enumerate", "list", "identify", "detect", "check", "find", "search", "query"],
    0.25: ["enumerat", "dns", "whois", "banner", "version"],
    0.35: ["brute", "fuzz", "wordlist", "dirb", "gobuster", "ffuf"],
    0.50: ["exploit", "inject", "overflow", "payload", "shell", "rce", "execute", "xss", "sqli",
            "bypass", "upload"],
    0.65: ["reverse shell", "callback", "connect back", "listener", "bind shell", "webshell"],
    0.75: ["privilege escalation", "privesc", "root", "admin", "suid", "sudo", "lateral",
            "pivot", "post-exploit", "persistence", "backdoor", "cron", "scheduled task"],
    0.90: ["exfiltrat", "steal", "dump", "extract data", "credential", "hashdump", "mimikatz",
            "keylog", "domain admin"],
}

PRIV_KEYWORDS = {
    0.0: ["no auth", "unauth", "anonymous", "public", "no access", "no priv"],
    0.25: ["low priv", "guest", "limited"],
    0.50: ["user", "standard", "authenticated", "domain user", "local user"],
    0.75: ["admin", "administrator", "sudo", "wheel", "local admin"],
    1.0: ["root", "system", "nt authority", "domain admin", "enterprise admin", "kernel"],
}


def heuristic_classify(fact: str) -> list[float]:
    """Classify a fact using keyword matching. Returns 7-dim structure vector."""
    text = fact.lower()
    vec = [0.3] * 7  # default mid-range

    # s0: attack_phase
    best_phase = 0.3
    best_phase_count = 0
    for score, keywords in PHASE_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text)
        if count > best_phase_count:
            best_phase = score
            best_phase_count = count
    vec[0] = best_phase

    # s1: privilege_level
    best_priv = 0.3
    for score, keywords in PRIV_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                best_priv = max(best_priv, score)
    vec[1] = best_priv

    # s2: access_scope
    if any(w in text for w in ["domain", "enterprise", "forest", "active directory"]):
        vec[2] = 0.9
    elif any(w in text for w in ["subnet", "network", "lateral", "pivot", "segment"]):
        vec[2] = 0.7
    elif any(w in text for w in ["host", "server", "machine", "target", "box"]):
        vec[2] = 0.5
    elif any(w in text for w in ["port", "service", "endpoint"]):
        vec[2] = 0.3

    # s3: stealth
    if any(w in text for w in ["offline", "silent", "undetectable", "passive", "stealthy"]):
        vec[3] = 0.9
    elif any(w in text for w in ["noisy", "detectable", "loud", "brute force", "scan"]):
        vec[3] = 0.1
    elif any(w in text for w in ["evasion", "obfuscat", "encode", "encrypt"]):
        vec[3] = 0.7

    # s4: interaction
    if any(w in text for w in ["write", "modify", "delete", "drop", "destroy", "overwrite",
                                 "inject", "upload", "deploy", "execute", "exploit"]):
        vec[4] = 0.8
    elif any(w in text for w in ["read", "view", "list", "dump", "extract", "passive"]):
        vec[4] = 0.2
    elif any(w in text for w in ["probe", "test", "check", "active"]):
        vec[4] = 0.5

    # s5: dependency_depth
    chain_words = ["after", "then", "requires", "prerequisite", "first", "chain",
                   "assuming", "given", "once you have"]
    dep_count = sum(1 for w in chain_words if w in text)
    vec[5] = min(0.2 + dep_count * 0.2, 1.0)

    # s6: target_specificity
    if re.search(r"CVE-\d{4}-\d+", text) or re.search(r"version\s+[\d.]+", text):
        vec[6] = 0.8
    elif any(w in text for w in ["specific", "exact", "only works", "particular"]):
        vec[6] = 0.7
    elif any(w in text for w in ["universal", "any", "all", "generic", "common"]):
        vec[6] = 0.2

    return [round(v, 2) for v in vec]


# ---------------------------------------------------------------------------
# LLM classifier
# ---------------------------------------------------------------------------
def llm_classify(fact: str, timeout: int = 30) -> list[float] | None:
    """Classify a fact using qwen3.5. Returns 7-dim vector or None on failure."""
    prompt = CLASSIFICATION_PROMPT.replace("{fact}", fact[:500])
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 100},
    }).encode()

    req = Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        log.warning(f"Ollama request failed: {e}")
        return None

    raw = data.get("response", "")
    # Strip <think> tags from qwen3
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Extract JSON array
    try:
        match = re.search(r"\[[\s\d.,]+\]", raw)
        if match:
            vec = json.loads(match.group())
            if isinstance(vec, list) and len(vec) == 7:
                # Clamp to [0, 1]
                return [round(max(0.0, min(1.0, float(v))), 2) for v in vec]
    except (json.JSONDecodeError, ValueError):
        pass

    log.warning(f"Failed to parse LLM response: {raw[:100]}")
    return None


def classify_fact(fact: str, use_llm: bool = True) -> list[float]:
    """Classify a fact, trying LLM first with heuristic fallback."""
    if use_llm:
        vec = llm_classify(fact)
        if vec is not None:
            return vec
    return heuristic_classify(fact)


# ---------------------------------------------------------------------------
# Batch classification
# ---------------------------------------------------------------------------
def pull_all_facts(memoria_url: str) -> list[dict]:
    """Pull all facts from Memoria via search with broad queries."""
    log.info("Pulling facts from Memoria...")
    all_facts = []
    seen_ids = set()

    # Use broad queries to get coverage, then paginate
    broad_queries = [
        "exploit vulnerability attack", "privilege escalation root",
        "command injection bypass", "SQL injection database",
        "reverse shell payload", "web application security",
        "network enumeration scanning", "password cracking brute force",
        "Active Directory domain", "Linux privilege escalation",
        "Windows exploitation", "file upload bypass",
        "container escape docker", "buffer overflow memory",
        "persistence backdoor", "lateral movement pivot",
        "credential theft", "WAF bypass encoding",
        "deserialization", "XML XXE SSRF",
        "Kerberos ticket", "SMB shares",
        "port forwarding tunnel", "binary analysis reverse",
    ]

    for query in broad_queries:
        try:
            payload = json.dumps({"query": query, "top_k": 200}).encode()
            req = Request(
                f"{memoria_url}/search",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())

            results = data.get("results", [])
            for r in results:
                fact_id = r.get("id", "")
                if fact_id and fact_id not in seen_ids:
                    seen_ids.add(fact_id)
                    all_facts.append({
                        "id": fact_id,
                        "text": r.get("text", r.get("fact", "")),
                    })
        except Exception as e:
            log.warning(f"Failed to search Memoria for '{query}': {e}")

    log.info(f"Pulled {len(all_facts)} unique facts from Memoria")
    return all_facts


def batch_classify(
    facts: list[dict],
    output_path: str,
    use_llm: bool = True,
    resume: bool = True,
) -> dict[str, list[float]]:
    """
    Classify all facts and save to JSON file.
    Returns dict of {fact_id: structure_vector}.
    """
    # Load existing progress
    existing = {}
    if resume and os.path.exists(output_path):
        with open(output_path, "r") as f:
            existing = json.load(f)
        log.info(f"Resuming: {len(existing)} facts already classified")

    results = dict(existing)
    classified = 0
    failed = 0
    t0 = time.time()

    for i, fact in enumerate(facts):
        fact_id = fact["id"]
        if fact_id in results:
            continue

        text = fact["text"]
        if not text or len(text) < 10:
            continue

        vec = classify_fact(text, use_llm=use_llm)
        results[fact_id] = vec
        classified += 1

        if classified % 10 == 0:
            elapsed = time.time() - t0
            rate = classified / elapsed if elapsed > 0 else 0
            eta = (len(facts) - i) / rate if rate > 0 else 0
            log.info(f"  [{i+1}/{len(facts)}] classified={classified} "
                     f"rate={rate:.1f}/s ETA={eta/60:.0f}min")

        if classified % SAVE_EVERY == 0:
            with open(output_path, "w") as f:
                json.dump(results, f)
            log.info(f"  Saved checkpoint ({len(results)} total)")

        # Small delay to not overwhelm Ollama
        if use_llm:
            time.sleep(0.1)

    # Final save
    with open(output_path, "w") as f:
        json.dump(results, f)

    elapsed = time.time() - t0
    log.info(f"Classification complete: {classified} new, {len(results)} total, "
             f"{elapsed:.0f}s elapsed")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Classify Memoria facts into structure vectors")
    parser.add_argument("--output", default="data/structure_vectors.json",
                        help="Output JSON file path")
    parser.add_argument("--no-llm", action="store_true",
                        help="Use heuristic classifier only (no Ollama)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, don't resume from existing file")
    parser.add_argument("--test", action="store_true",
                        help="Test with a few facts and print results")
    args = parser.parse_args()

    if args.test:
        test_facts = [
            "Use nmap -sV to detect service versions on open ports",
            "GetUserSPNs.py from Impacket requests TGS tickets for Kerberoasting",
            "find / -perm -4000 2>/dev/null to enumerate SUID binaries",
            "xp_cmdshell allows OS command execution from MSSQL",
            "certutil -urlcache -split -f can download files on Windows",
        ]
        for fact in test_facts:
            h_vec = heuristic_classify(fact)
            print(f"\nFact: {fact[:70]}...")
            print(f"  Heuristic: {h_vec}")
            print(f"  Dims: {dict(zip(STRUCTURE_DIMS, h_vec))}")
        sys.exit(0)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Pull facts and classify
    facts = pull_all_facts(MEMORIA_URL)
    if not facts:
        log.error("No facts retrieved from Memoria")
        sys.exit(1)

    batch_classify(
        facts,
        output_path=args.output,
        use_llm=not args.no_llm,
        resume=not args.no_resume,
    )
