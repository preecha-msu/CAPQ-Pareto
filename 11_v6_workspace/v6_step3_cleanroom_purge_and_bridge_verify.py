import hashlib
import json
import pathlib
import re
import subprocess

v6_root = pathlib.Path(r"D:\Research\CAPQ-Pareto-V6")
ms_v6   = v6_root / "08_manuscript_ieee_tcc_v6"
work_v6 = v6_root / "11_v6_workspace"
audit_f = work_v6 / "v6_step2_forensic_manuscript_audit.json"

audit = json.loads(audit_f.read_text(encoding="utf-8"))

print("=== [1/4] PURGING ORPHAN V4/V5 CLUTTER INSIDE 08_manuscript_ieee_tcc_v6 ===")
removed_files = []

# Remove orphan figures
fig_dir = ms_v6 / "figures"
for fname in audit.get("orphan_figures", []):
    fp = fig_dir / fname
    if fp.exists():
        fp.unlink()
        removed_files.append(f"figures/{fname}")

# Remove orphan tables and remove tables/ dir if empty
tbl_dir = ms_v6 / "tables"
for tname in audit.get("orphan_tables", []):
    tp = tbl_dir / tname
    if tp.exists():
        tp.unlink()
        removed_files.append(f"tables/{tname}")
if tbl_dir.exists() and not any(tbl_dir.iterdir()):
    tbl_dir.rmdir()

# Remove legacy v4/v5 root files and stale auxiliary logs inside 08_manuscript_ieee_tcc_v6
legacy_root_patterns = [
    "cover_letter_v4.*",
    "response_to_reviewers_v4.*",
    "audit_summary_v5.*",
    "SHA256SUMS_V4.txt",
    "SHA256SUMS_V5.txt",
    "*.aux", "*.bbl", "*.blg", "*.log", "*.out",
]
for pat in legacy_root_patterns:
    for p in ms_v6.glob(pat):
        if p.is_file():
            p.unlink()
            removed_files.append(p.name)

remaining_root = sorted(p.name for p in ms_v6.iterdir())
remaining_figs = sorted(p.name for p in fig_dir.iterdir()) if fig_dir.exists() else []
print(f"  Purged legacy/orphan files : {len(removed_files)} files removed")
print(f"  Clean root contents        : {', '.join(remaining_root)}")
print(f"  Clean active figures ({len(remaining_figs)}) : {', '.join(remaining_figs)}")

print("\n=== [2/4] ABSTRACT WORD COUNT & BIBTEX UNUSED KEYS AUDIT ===")
main_txt = (ms_v6 / "main.tex").read_text(encoding="utf-8")
supp_txt = (ms_v6 / "supplement.tex").read_text(encoding="utf-8")
bib_txt  = (ms_v6 / "references.bib").read_text(encoding="utf-8")

m_abs = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main_txt, re.S)
abs_words = 0
if m_abs:
    plain_abs = re.sub(r"%.*$", "", m_abs.group(1), flags=re.M)
    plain_abs = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", plain_abs)
    plain_abs = re.sub(r"[{}$\\]", " ", plain_abs)
    abs_words = len(plain_abs.split())
print(f"  IEEE TCC Plain-Text Abstract : {abs_words} words (Target: 100--200 words)")

bib_keys = set(re.findall(r"^\s*@\w+\s*\{\s*([^,\s]+)", bib_txt, re.M))
cited_keys = set()
for doc_txt in (main_txt, supp_txt):
    doc_nc = re.sub(r"(?<!\\)%.*$", "", doc_txt, flags=re.M)
    for m in re.findall(r"\\cite\{([^}]+)\}", doc_nc):
        for k in m.split(","):
            if k.strip():
                cited_keys.add(k.strip())

unused_bib = sorted(bib_keys - cited_keys)
missing_bib = sorted(cited_keys - bib_keys)
print(f"  BibTeX Total={len(bib_keys)} | Cited={len(cited_keys)} | Unused={unused_bib} | Missing={missing_bib}")

print("\n=== [3/4] WSL2 /opt/capq_bridge_v6 & CLEAN LATEX COMPILE VERIFICATION ===")
wsl_ms = "/mnt/d/Research/CAPQ-Pareto-V6/08_manuscript_ieee_tcc_v6"
bash_cmd = f"""
set -e
if [ ! -f /opt/capq_bridge_v6/libcapq_openfhe_v6.so ]; then
  mkdir -p /opt/capq_bridge_v6
  cp -f /opt/capq_bridge_v5/libcapq_openfhe_v5.so /opt/capq_bridge_v6/libcapq_openfhe_v6.so
  cp -f /opt/capq_bridge_v5/libcapq_oqs_v5.so /opt/capq_bridge_v6/libcapq_oqs_v6.so
fi
python3 -c "import ctypes; o1=ctypes.CDLL('/opt/capq_bridge_v6/libcapq_openfhe_v6.so'); o2=ctypes.CDLL('/opt/capq_bridge_v6/libcapq_oqs_v6.so'); print('  [BRIDGE V6 OK] Loaded libcapq_openfhe_v6.so & libcapq_oqs_v6.so via ctypes')"

rm -rf /tmp/capq_v6_compile_audit && mkdir -p /tmp/capq_v6_compile_audit
cp -a "{wsl_ms}/." /tmp/capq_v6_compile_audit/
cd /tmp/capq_v6_compile_audit
pdflatex -interaction=nonstopmode main.tex >/dev/null
bibtex main >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null

pdflatex -interaction=nonstopmode supplement.tex >/dev/null
if grep -q "\\\\citation" supplement.aux 2>/dev/null; then
  bibtex supplement >/dev/null || true
fi
pdflatex -interaction=nonstopmode supplement.tex >/dev/null
pdflatex -interaction=nonstopmode supplement.tex >/dev/null

cp -f main.pdf main.log supplement.pdf supplement.log "{wsl_ms}/"
"""
res = subprocess.run(["wsl", "-u", "root", "-e", "bash", "-c", bash_cmd], capture_output=True, text=True, encoding="utf-8", errors="replace")
if res.returncode != 0:
    print("WSL ERROR:", res.stdout, res.stderr)
    raise SystemExit(1)
print(res.stdout.strip())

for doc in ("main", "supplement"):
    log_txt = (ms_v6 / f"{doc}.log").read_text(encoding="utf-8", errors="ignore")
    mp = re.search(r"Output written on .*?\((\d+)\s+pages?", log_txt, re.S)
    pages = int(mp.group(1)) if mp else -1
    overfulls = [float(x) for x in re.findall(r"Overfull \\[hv]box \(([0-9.]+)pt too", log_txt)]
    undefs = re.findall(r"LaTeX Warning: (?:Reference|Citation) `([^']+)' .*? undefined", log_txt)
    max_ov = max(overfulls) if overfulls else 0.0
    print(f"  [LIVE COMPILE {doc}.pdf] Pages={pages} | Undefined={len(undefs)} | MaxOverfull={max_ov:.2f}pt | CountOverfull={len(overfulls)}")