import hashlib
import json
import pathlib
import re
import shutil

v6_root = pathlib.Path(r"D:\Research\CAPQ-Pareto-V6")
res_dir = pathlib.Path(r"D:\Research")
ms_v6   = v6_root / "08_manuscript_ieee_tcc_v6"
work_v6 = v6_root / "11_v6_workspace"
v5_logs = v6_root / "06_experiments_results" / "v5_raw_logs"
v6_logs = v6_root / "06_experiments_results" / "v6_raw_logs"
v6_logs.mkdir(parents=True, exist_ok=True)

# 1. Seed v6_raw_logs with verified V5 raw logs (keeping both original and v6-prefixed names for compatibility)
seeded = 0
if v5_logs.exists():
    for src in sorted(v5_logs.iterdir()):
        if src.is_file():
            dst_same = v6_logs / src.name
            if not dst_same.exists():
                shutil.copy2(src, dst_same)
                seeded += 1
            if src.name.startswith("v5_"):
                dst_v6 = v6_logs / ("v6_" + src.name[3:])
                if not dst_v6.exists():
                    shutil.copy2(src, dst_v6)

# Also rebind /opt/capq_bridge_v5 -> /opt/capq_bridge_v6 inside 11_v6_workspace scripts
rebound_so = 0
for py_f in work_v6.glob("*.py"):
    if py_f.name.startswith("v6_step"):
        continue
    txt = py_f.read_text(encoding="utf-8", errors="ignore")
    upd = (
        txt.replace("/opt/capq_bridge_v5/libcapq_openfhe_v5.so", "/opt/capq_bridge_v6/libcapq_openfhe_v6.so")
           .replace("/opt/capq_bridge_v5/libcapq_oqs_v5.so", "/opt/capq_bridge_v6/libcapq_oqs_v6.so")
           .replace("/opt/capq_bridge_v5", "/opt/capq_bridge_v6")
    )
    if upd != txt:
        py_f.write_text(upd, encoding="utf-8", newline="\n")
        rebound_so += 1

print(f"=== [2/4] V6 RAW LOGS & BRIDGE REBINDING ===")
print(f"  Seeded files into v6_raw_logs : {len(list(v6_logs.iterdir()))} total files ({seeded} newly copied)")
print(f"  Scripts rebound to bridge_v6  : {rebound_so} scripts in 11_v6_workspace")

# 2. Deep Forensic Audit of 08_manuscript_ieee_tcc_v6
def inspect_tex(tex_path: pathlib.Path):
    if not tex_path.exists():
        return {"exists": False}
    raw = tex_path.read_text(encoding="utf-8", errors="ignore")
    # Strip comments for accurate structural parsing
    clean_lines = []
    for ln in raw.splitlines():
        ln_nc = re.sub(r"(?<!\\)%.*$", "", ln)
        clean_lines.append(ln_nc)
    clean = "\n".join(clean_lines)

    sections = re.findall(r"\\section\*?\{([^}]*)\}", clean)
    subsections = re.findall(r"\\subsection\*?\{([^}]*)\}", clean)
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", clean)
    inputs = re.findall(r"\\input\{([^}]+)\}", clean)
    cites = set()
    for m in re.findall(r"\\cite\{([^}]+)\}", clean):
        for k in m.split(","):
            if k.strip():
                cites.add(k.strip())
    tables_env = len(re.findall(r"\\begin\{table\*?\}", clean))
    eqs_env = len(re.findall(r"\\begin\{(equation|align|multline)\*?\}", clean))

    # Check log file for exact page count and overfull boxes if present
    log_path = tex_path.with_suffix(".log")
    pages = None
    max_overfull_pt = 0.0
    if log_path.exists():
        log_txt = log_path.read_text(encoding="utf-8", errors="ignore")
        mp = re.search(r"Output written on .*?\((\d+)\s+pages?", log_txt, re.S)
        if mp:
            pages = int(mp.group(1))
        for mo in re.findall(r"Overfull \\[hv]box \(([0-9.]+)pt too", log_txt):
            max_overfull_pt = max(max_overfull_pt, float(mo))

    return {
        "exists": True,
        "lines": len(raw.splitlines()),
        "bytes": tex_path.stat().st_size,
        "compiled_pages": pages,
        "max_overfull_pt": max_overfull_pt,
        "sections": sections,
        "subsection_count": len(subsections),
        "table_envs": tables_env,
        "equation_envs": eqs_env,
        "included_figures": figs,
        "included_inputs": inputs,
        "unique_citations": len(cites),
    }

main_audit = inspect_tex(ms_v6 / "main.tex")
supp_audit = inspect_tex(ms_v6 / "supplement.tex")

bib_entries = 0
bib_path = ms_v6 / "references.bib"
if bib_path.exists():
    bib_txt = bib_path.read_text(encoding="utf-8", errors="ignore")
    bib_entries = len(re.findall(r"^\s*@\w+\s*\{", bib_txt, re.M))

# Identify active vs orphan files in figures/ and tables/
used_figs = set()
for f in main_audit.get("included_figures", []) + supp_audit.get("included_figures", []):
    fname = pathlib.Path(f).name
    used_figs.add(fname)
    if not pathlib.Path(fname).suffix:
        used_figs.add(fname + ".pdf")
        used_figs.add(fname + ".png")

all_figs = sorted(p.name for p in (ms_v6 / "figures").glob("*.*")) if (ms_v6 / "figures").exists() else []
orphan_figs = [f for f in all_figs if f not in used_figs]

used_tbls = set()
for t in main_audit.get("included_inputs", []) + supp_audit.get("included_inputs", []):
    tname = pathlib.Path(t).name
    if not tname.endswith(".tex"):
        tname += ".tex"
    used_tbls.add(tname)

all_tbls = sorted(p.name for p in (ms_v6 / "tables").glob("*.tex")) if (ms_v6 / "tables").exists() else []
orphan_tbls = [t for t in all_tbls if t not in used_tbls]

print("\n=== [3/4] DEEP FORENSIC AUDIT OF 08_manuscript_ieee_tcc_v6 ===")
print(f"  [main.tex]       Pages={main_audit['compiled_pages']} | Lines={main_audit['lines']} | Secs={len(main_audit['sections'])} | Subsecs={main_audit['subsection_count']} | Tables={main_audit['table_envs']} | Eqs={main_audit['equation_envs']} | Figs={len(main_audit['included_figures'])} | Cites={main_audit['unique_citations']}/{bib_entries} | MaxOverfull={main_audit['max_overfull_pt']}pt")
for idx, s in enumerate(main_audit["sections"], 1):
    print(f"    Sec {idx}: {s}")
print(f"    Active Main Figures : {', '.join(pathlib.Path(x).name for x in main_audit['included_figures'])}")
print(f"    Active Main Inputs  : {', '.join(main_audit['included_inputs']) if main_audit['included_inputs'] else '(Inline tables in main.tex)'}")

print(f"\n  [supplement.tex] Pages={supp_audit['compiled_pages']} | Lines={supp_audit['lines']} | Secs={len(supp_audit['sections'])} | Tables={supp_audit['table_envs']} | Eqs={supp_audit['equation_envs']} | Figs={len(supp_audit['included_figures'])} | MaxOverfull={supp_audit['max_overfull_pt']}pt")
for idx, s in enumerate(supp_audit["sections"], 1):
    print(f"    Supp Sec {idx}: {s}")
print(f"    Active Supp Figures : {', '.join(pathlib.Path(x).name for x in supp_audit['included_figures'])}")

print(f"\n  [Orphan/Legacy Cleanup Candidates in 08_manuscript_ieee_tcc_v6]")
print(f"    Unused Figures ({len(orphan_figs)}/{len(all_figs)}) : {', '.join(orphan_figs[:10])}{' ...' if len(orphan_figs) > 10 else ''}")
print(f"    Unused Tables  ({len(orphan_tbls)}/{len(all_tbls)}) : {', '.join(orphan_tbls[:10])}{' ...' if len(orphan_tbls) > 10 else ''}")

report_path = work_v6 / "v6_step2_forensic_manuscript_audit.json"
report_data = {
    "main_tex": main_audit,
    "supplement_tex": supp_audit,
    "bib_total_entries": bib_entries,
    "orphan_figures": orphan_figs,
    "orphan_tables": orphan_tbls,
}
report_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
rh = hashlib.sha256(report_path.read_bytes()).hexdigest()
print(f"\n[FORENSIC REPORT SAVED] {report_path} | SHA256={rh}")