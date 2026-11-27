import csv
import hashlib
import json
import pathlib
import re

v6_root = pathlib.Path(r"D:\Research\CAPQ-Pareto-V6")
ms_v2   = v6_root / "08_manuscript_ieee_tcc_v2" / "main.tex"
ms_v6   = v6_root / "08_manuscript_ieee_tcc_v6"
work_v6 = v6_root / "11_v6_workspace"

print("=== [1.1-A] EXTRACTING REAL AUTHOR, AFFILIATION, GRANT & REPO METADATA FROM V2 ===")
if ms_v2.exists():
    v2_txt = ms_v2.read_text(encoding="utf-8", errors="ignore")
    for pat_name, pat in [
        ("markboth", r"\\markboth\{.*?\}\{.*?\}"),
        ("author",   r"\\author\{.*?\}(?=\s*\\IEEEtitleabstractindextext|\s*\\maketitle)"),
        ("ack",      r"\\section\*?\{.*?(?:Acknowledgment|Data Availability).*?\}.*?(?=\\bibliographystyle|\\begin\{IEEEbiography\}|\\end\{document\})"),
        ("bio",      r"\\begin\{IEEEbiography.*?\}.*?\\end\{IEEEbiography.*?\}"),
    ]:
        m = re.search(pat, v2_txt, re.S)
        print(f"-- [{pat_name}] --")
        print(m.group(0).strip()[:900] if m else "  (not found)")
    urls = sorted(set(re.findall(r"https?://[^\s\}\)]+", v2_txt)))
    print(f"-- [V2 URLs] -- {urls}")

print("\n=== [1.1-B] SCANNING V6 MANUSCRIPT FOR '**' MARKDOWN LEAKS & OVERFULL LINES ===")
for fn in ("main.tex", "supplement.tex", "references.bib"):
    p = ms_v6 / fn
    if p.exists():
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        md_hits = [(i + 1, ln.strip()[:100]) for i, ln in enumerate(lines) if "**" in ln]
        print(f"  {fn:<16}: {len(md_hits)} lines containing '**'")
        for ln_no, snippet in md_hits[:8]:
            print(f"    L{ln_no:03d}: {snippet}")

log_p = ms_v6 / "main.log"
if log_p.exists():
    log_txt = log_p.read_text(encoding="utf-8", errors="ignore")
    ov_hits = re.findall(r"(Overfull \\[hv]box \([0-9.]+pt too .*?lines? \d+--?\d+)", log_txt)
    print(f"  main.log Overfull warnings ({len(ov_hits)}):")
    for h in ov_hits:
        print(f"    {h}")

print("\n=== [1.2] INSPECTING EXACT SCHEMAS OF EXISTING REAL CSV/JSON LOGS FOR REPROCESSING ===")
search_dirs = [
    v6_root / "06_experiments_results" / "v6_raw_logs",
    v6_root / "06_experiments_results" / "v2_raw_logs",
    v6_root / "06_experiments_results" / "v3_raw_logs",
    v6_root / "09_v4_reviewer_rebuttal_audit",
]
schema_report = {}
for d in search_dirs:
    if not d.exists():
        continue
    print(f"-- Directory: {d.relative_to(v6_root)} --")
    for f in sorted(d.iterdir()):
        if f.suffix.lower() == ".csv" and not f.name.startswith("v6_"):
            h = hashlib.sha256(f.read_bytes()).hexdigest()
            with f.open("r", encoding="utf-8", errors="ignore", newline="") as fp:
                rd = csv.reader(fp)
                hdr = next(rd, [])
                rows = sum(1 for _ in rd)
            schema_report[str(f.relative_to(v6_root))] = {"rows": rows, "cols": hdr, "sha256": h}
            print(f"  {f.name:<48} | rows={rows:>5} | sha={h[:12]} | cols={hdr[:8]}")
        elif f.suffix.lower() == ".json" and not f.name.startswith("v6_"):
            h = hashlib.sha256(f.read_bytes()).hexdigest()
            print(f"  {f.name:<48} | JSON       | sha={h[:12]}")

print("\n=== [1.2-B] WORKLOADS (.npy/.npz) & V5/V6 WORKSPACE SCRIPTS ===")
for wf in sorted((v6_root / "01_workloads").rglob("*.np[yz]")):
    print(f"  Workload: {wf.relative_to(v6_root)} ({wf.stat().st_size:,} bytes)")
for sf in sorted(work_v6.glob("run_step*.py")):
    print(f"  Script  : {sf.name} ({sf.stat().st_size:,} bytes)")

out_json = work_v6 / "v6_step_a1_schema_and_metadata.json"
out_json.write_text(json.dumps(schema_report, indent=2) + "\n", encoding="utf-8", newline="\n")
print(f"\n[SAVED] {out_json.name} | SHA256={hashlib.sha256(out_json.read_bytes()).hexdigest()}")