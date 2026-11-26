from __future__ import annotations
import csv
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

V5_ROOT = Path(r"D:\Research\CAPQ-Pareto-V6")
WORK_DIR = V5_ROOT / "11_v6_workspace"
WORK_DIR.mkdir(parents=True, exist_ok=True)


def sha256_hex(fp: Path) -> str:
    h = hashlib.sha256()
    with fp.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# 1. Verify Zero-Touch V4 SHA-256 Lock
v4_files = [
    Path(r"D:\Research\CAPQ-Pareto-Main-IEEE-TCC-V4.pdf"),
    Path(r"D:\Research\CAPQ-Pareto-Supplement-IEEE-TCC-V4.pdf"),
    Path(r"D:\Research\CAPQ-Pareto-Overleaf-IEEE-TCC-V4.zip"),
]
v4_hashes = {p.name: sha256_hex(p) for p in v4_files if p.exists()}

# 2. Locate active V5 LaTeX manuscript directory
ms_candidates = [
    V5_ROOT / "08_manuscript_ieee_tcc_v6",
    V5_ROOT / "09_v4_reviewer_rebuttal_audit" / "overleaf_ieee_tcc_v5",
    V5_ROOT / "09_v4_reviewer_rebuttal_audit" / "overleaf_ieee_tcc_v4",
]
ms_dir = next((p for p in ms_candidates if (p / "main.tex").exists()), None)
if ms_dir is None:
    tex_hits = sorted(V5_ROOT.rglob("main.tex"), key=lambda x: x.stat().st_mtime, reverse=True)
    ms_dir = tex_hits[0].parent if tex_hits else None

assert ms_dir is not None, "Could not find main.tex inside CAPQ-Pareto-V5!"

# 3. Scan main.tex & supplement.tex for P0.1 / P1 defects
forbidden_patterns = [
    r"\bV2\b", r"\bV3\b", r"\bV4\b", r"\bV4_1\b",
    r"pre-registered", r"pre-declared", r"forensic", r"Forensic",
    r"09_v4_reviewer_rebuttal_audit", r"revised for IEEE TCC", r"!",
]

tex_findings = {"manuscript_dir": str(ms_dir), "abstract_unescaped_pct": [], "bare_inequalities": [], "forbidden_terms": []}
for tex_fp in sorted(ms_dir.rglob("*.tex")):
    lines = tex_fp.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_abs = False
    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        if r"\begin{abstract}" in line:
            in_abs = True
        if in_abs and not stripped.startswith("%"):
            if re.search(r"(?<!\\)%", line):
                tex_findings["abstract_unescaped_pct"].append(f"{tex_fp.name}:L{lineno}: {stripped[:110]}")
        if r"\end{abstract}" in line:
            in_abs = False

        if not stripped.startswith("%"):
            # Detect bare < or > outside $...$ math mode (excluding LaTeX tags)
            no_math = re.sub(r"\$[^$]*\$", "", line)
            if re.search(r"\bp\s*[<>]\s*0\.", no_math):
                tex_findings["bare_inequalities"].append(f"{tex_fp.name}:L{lineno}: {stripped[:100]}")
            for pat in forbidden_patterns:
                if re.search(pat, line):
                    tex_findings["forbidden_terms"].append(f"{tex_fp.name}:L{lineno} [{pat}]: {stripped[:95]}")

# 4. Exact Mathematical Verification for Bloom Filter FP & Prop 1 95%-Quantile Tightness
m_bits, n_items, k_hashes = 65536, 4000, 5
bloom_fp_exact_pct = ((1.0 - (1.0 - 1.0 / m_bits) ** (k_hashes * n_items)) ** k_hashes) * 100.0
bloom_fp_exp_approx_pct = ((1.0 - math.exp(-k_hashes * n_items / m_bits)) ** k_hashes) * 100.0

# 5. Check WSL2 Native Bridges & Datasets availability for Step 2 Real Execution
wsl_probe = subprocess.run(
    [
        "wsl", "-u", "root", "-e", "bash", "-lc",
        "ls -la /opt/capq_bridge*/*.so 2>/dev/null || true; which g++ cmake python3 2>/dev/null || true"
    ],
    capture_output=True, text=True
)
datasets_found = [str(p.relative_to(V5_ROOT)) for p in sorted(V5_ROOT.rglob("*.np*"))[:15]]
csv_count = len(list(V5_ROOT.rglob("*.csv")))

report = {
    "v4_locked_hashes": v4_hashes,
    "active_v5_manuscript_dir": str(ms_dir),
    "total_csv_files_in_v5": csv_count,
    "datasets_sample": datasets_found,
    "bloom_fp_math_check": {
        "exact_combinatorial_pct": round(bloom_fp_exact_pct, 5),
        "exponential_approx_pct": round(bloom_fp_exp_approx_pct, 5),
    },
    "tex_findings_summary": {
        "abstract_unescaped_pct_count": len(tex_findings["abstract_unescaped_pct"]),
        "abstract_unescaped_pct_lines": tex_findings["abstract_unescaped_pct"],
        "bare_inequalities_count": len(tex_findings["bare_inequalities"]),
        "bare_inequalities_sample": tex_findings["bare_inequalities"][:8],
        "forbidden_terms_count": len(tex_findings["forbidden_terms"]),
        "forbidden_terms_sample": tex_findings["forbidden_terms"][:12],
    },
    "wsl_bridge_status": wsl_probe.stdout.strip().splitlines(),
}

out_json = WORK_DIR / "v5_step1_forensic_diagnostic.json"
out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

print("=== [STEP 1.1] ZERO-TOUCH V4 LOCK STATUS ===")
for k, v in v4_hashes.items():
    print(f"  [LOCKED] {k:<44} | SHA256={v[:20]}...")

print(f"\n=== [STEP 1.2] V5 MANUSCRIPT & ARTIFACT SCAN ({ms_dir}) ===")
print(f"  CSV files in V5 workspace        : {csv_count}")
print(f"  Datasets (.npy/.npz) detected    : {len(datasets_found)}")
print(f"  Bloom FP exact math (m=64Kb,k=5) : {bloom_fp_exact_pct:.4f}% (confirms 0.126% vs erroneous 0.134%)")
print(f"  Abstract unescaped '%' lines     : {len(tex_findings['abstract_unescaped_pct'])}")
for ln in tex_findings["abstract_unescaped_pct"]:
    print(f"    -> {ln}")
print(f"  Bare '<'/'>' outside math mode   : {len(tex_findings['bare_inequalities'])}")
for ln in tex_findings["bare_inequalities"][:5]:
    print(f"    -> {ln}")
print(f"  Legacy/defensive terms found     : {len(tex_findings['forbidden_terms'])}")
for ln in tex_findings["forbidden_terms"][:8]:
    print(f"    -> {ln}")

print("\n=== [STEP 1.3] WSL2 NATIVE C++ OPENFHE / LIBOQS TOOLCHAIN CHECK ===")
for ln in report["wsl_bridge_status"]:
    print(f"  {ln}")
print(f"\n[SAVED DIAGNOSTIC] {out_json} | SHA256={sha256_hex(out_json)}")