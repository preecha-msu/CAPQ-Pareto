from __future__ import annotations
import csv
import json
import re
from pathlib import Path

REPO = Path(r"D:\Research\CAPQ-Pareto-V3")
AUDIT = REPO / "09_v4_reviewer_rebuttal_audit"

print("==========================================================================")
print("1. EXACT CONTENTS OF ALL 16 V4 VERIFIED ARTIFACTS (GROUND-TRUTH DUMP)")
print("==========================================================================")
for p in sorted(AUDIT.glob("*")):
    if p.suffix == ".json":
        print(f"\n--- [JSON] {p.name} ---")
        obj = json.loads(p.read_text(encoding="utf-8"))
        if p.name == "v4_phase2_real_netem_socket_traces.json":
            print("  keys:", list(obj.keys()), "| net_profiles:", list(obj.get("net_profiles", {}).keys()))
        else:
            print(" ", json.dumps(obj, indent=2)[:900])
    elif p.suffix == ".csv":
        with p.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        print(f"\n--- [CSV] {p.name} (rows={len(rows)}) ---")
        show_n = 6 if len(rows) > 25 else len(rows)
        for r in rows[:show_n]:
            print(" ", dict(r))
        if len(rows) > show_n:
            print(f"  ... ({len(rows) - show_n} more rows)")

print("\n==========================================================================")
print("2. EXISTING PLOTTING & FIG-1 SCRIPTS IN REPO")
print("==========================================================================")
for sp in sorted(REPO.rglob("*.py")):
    if any(k in sp.name.lower() for k in ["fig", "plot", "v3", "manuscript", "upgrade"]):
        print(f"  {sp.relative_to(REPO)} ({sp.stat().st_size:,} B)")

fig1_scripts = list(REPO.rglob("*fig1*.py")) + [p for p in REPO.rglob("*.py") if "SWIMLANE" in p.read_text(encoding="utf-8", errors="ignore").upper() or "GUARANTEES" in p.read_text(encoding="utf-8", errors="ignore")]
print("\n--- Fig 1 Flagship Generator Candidates ---")
for fp in sorted(set(fig1_scripts)):
    print(f"  [FIG1-SRC] {fp.relative_to(REPO)} ({fp.stat().st_size:,} B)")

print("\n==========================================================================")
print("3. TABLE & FIGURE INVENTORY: V2 (11p) vs V3 (Main+Supp) vs V4 (Current)")
print("==========================================================================")
for label, tex_p in [
    ("V2 main.tex", REPO / "08_manuscript_ieee_tcc_v2" / "main.tex"),
    ("V2 supp.tex", REPO / "08_manuscript_ieee_tcc_v2" / "supplement.tex"),
    ("V3 main.tex", REPO / "08_manuscript_ieee_tcc_v3" / "main.tex"),
    ("V3 supp.tex", REPO / "08_manuscript_ieee_tcc_v3" / "supplement.tex"),
    ("V4 main.tex", REPO / "10_manuscript_ieee_tcc_v4" / "main.tex"),
    ("V4 supp.tex", REPO / "10_manuscript_ieee_tcc_v4" / "supplement.tex"),
]:
    if tex_p.is_file():
        t = tex_p.read_text(encoding="utf-8", errors="ignore")
        caps = re.findall(r"\\caption\{([^}]{1,95})", t)
        figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", t)
        secs = re.findall(r"\\section\*?\{([^}]+)\}", t)
        print(f"\n[{label}] size={tex_p.stat().st_size:,} B | sections={len(secs)} | captions={len(caps)} | figs={len(figs)}")
        print("  Sections:", secs)
        print("  Figures :", figs)
        for idx, c in enumerate(caps, 1):
            print(f"  Cap {idx:02d}  : {c.replace(chr(10), ' ')}")

print("\n==========================================================================")
print("4. ALL RAW CSV LOGS ACROSS V1 / V2 / V3 FOR COMPLETE MANIFEST")
print("==========================================================================")
for d_rel in ["06_experiments_results/raw_logs", "06_experiments_results/v2_raw_logs", "06_experiments_results/v3_raw_logs"]:
    dp = REPO / d_rel
    if dp.is_dir():
        c_list = sorted(dp.glob("*.csv"))
        print(f"  [{d_rel}] {len(c_list)} CSVs:", [x.name for x in c_list])