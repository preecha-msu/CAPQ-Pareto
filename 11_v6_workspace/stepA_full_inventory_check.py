from pathlib import Path
import csv, json

ROOT = Path(r"D:\Research\CAPQ-Pareto-V6")

folders = [
    ("V1_raw_logs", ROOT / "06_experiments_results" / "raw_logs"),
    ("V2_raw_logs", ROOT / "06_experiments_results" / "v2_raw_logs"),
    ("V3_raw_logs", ROOT / "06_experiments_results" / "v3_raw_logs"),
    ("V4_audit",    ROOT / "09_v4_reviewer_rebuttal_audit"),
]

total_files = 0
for tag, d in folders:
    print(f"=== [{tag}] ({d}) ===")
    if not d.is_dir():
        print("  [MISSING DIR]")
        continue
    for fp in sorted(list(d.glob("*.csv")) + list(d.glob("*.json"))):
        total_files += 1
        if fp.suffix == ".csv":
            with fp.open("r", encoding="utf-8", errors="ignore") as f:
                rws = list(csv.DictReader(f))
            cols = list(rws[0].keys()) if rws else []
            print(f"  {fp.name:<48} | rows={len(rws):4d} | cols={cols[:6]}")
        else:
            dat = json.loads(fp.read_text(encoding="utf-8"))
            ks = list(dat.keys()) if isinstance(dat, dict) else [f"list[{len(dat)}]"]
            print(f"  {fp.name:<48} | json      | keys={ks[:6]}")

print(f"\n[TOTAL EMPIRICAL ARTIFACT FILES FOUND]: {total_files}")

print("\n=== [EXISTING MANUSCRIPTS & FLAGSHIP FIG-1 SCRIPTS] ===")
for rel in [
    r"scripts\upgrade_fig1_and_build.py",
    r"02_orchestrator\master_v3_publication_rebuild.py",
    r"08_manuscript_ieee_tcc_v2\main.tex",
    r"08_manuscript_ieee_tcc_v3\main.tex",
    r"08_manuscript_ieee_tcc_v3\supplement.tex",
]:
    p = ROOT / rel
    print(f"  {rel:<50} | exists={p.is_file()} | size={p.stat().st_size if p.is_file() else 0:,} B")