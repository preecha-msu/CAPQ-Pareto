from pathlib import Path
import json, re, subprocess, sys, zipfile

REPO = Path(r"D:\Research\CAPQ-Pareto")
sys.path.insert(0, str(REPO / "05_telemetry_audit"))
import sha256_manifest as sm


def balance_cell_braces(cell: str) -> str:
    out = []
    depth = 0
    i = 0
    while i < len(cell):
        if cell[i] == "\\" and i + 1 < len(cell) and cell[i + 1] in "{}":
            out.append(cell[i : i + 2])
            i += 2
            continue
        if cell[i] == "{":
            depth += 1
            out.append("{")
        elif cell[i] == "}":
            if depth > 0:
                depth -= 1
                out.append("}")
        else:
            out.append(cell[i])
        i += 1
    if depth > 0:
        out.append("}" * depth)
    return "".join(out)


def fix_tabular_rows(tex: str) -> str:
    lines = tex.splitlines()
    in_tab = False
    fixed = []
    for ln in lines:
        s = ln.strip()
        if r"\begin{tabular}" in s:
            in_tab = True
            fixed.append(ln)
            continue
        if r"\end{tabular}" in s:
            in_tab = False
            fixed.append(ln)
            continue
        if in_tab and s.endswith(r"\\") and not s.startswith("%"):
            body = ln[: ln.rfind(r"\\")]
            tail = ln[ln.rfind(r"\\") :]
            # Split on unescaped &
            cells = re.split(r"(?<!\\)&", body)
            cells = [balance_cell_braces(c) for c in cells]
            fixed.append("&".join(cells) + tail)
        else:
            fixed.append(ln)
    return "\n".join(fixed) + "\n"


bib_keys = set(re.findall(r"@\w+\{([^,\s]+),", (REPO / "references.bib").read_text(encoding="utf-8")))
key_map = {"badawi2022openfhe": "albadawi2022openfhe", "iyengar2021quic": "iyengar2021rfc9000", "tpcc": "tpcc_spec"}

# 1. Repair braces/citations in every table & unit-test each table individually in pdflatex
for tf in sorted((REPO / "00_manuscript/tables").glob("*.tex")):
    t = tf.read_text(encoding="utf-8")
    for k_old, k_new in key_map.items():
        t = re.sub(rf"\b{k_old}\b", k_new, t)
    def _clean_cite(m):
        kept = [k.strip() for k in m.group(1).split(",") if k.strip() in bib_keys]
        return f"\\cite{{{','.join(kept)}}}" if kept else ""
    t = re.sub(r"\\cite\{([^}]+)\}", _clean_cite, t)
    t = fix_tabular_rows(t)
    tf.write_text(t, encoding="utf-8", newline="\n")

    test_doc = (
        "\\documentclass[10pt,journal,compsoc]{IEEEtran}\n"
        "\\usepackage{amsmath,amssymb,graphicx,adjustbox,booktabs,multirow,array,xcolor,url}\n"
        "\\begin{document}\n"
        f"\\input{{/mnt/d/Research/CAPQ-Pareto/00_manuscript/tables/{tf.name}}}\n"
        "\\end{document}\n"
    )
    chk = subprocess.run(
        ["wsl", "-u", "root", "-e", "bash", "-c", "mkdir -p /tmp/capq_tchk && cd /tmp/capq_tchk && cat > t.tex && pdflatex -interaction=nonstopmode -halt-on-error t.tex > t.log 2>&1 || (tail -n 20 t.log && exit 1)"],
        input=test_doc, text=True, capture_output=True,
    )
    if chk.returncode != 0:
        print(f"[TABLE ERROR in {tf.name}]\n{chk.stdout}")
        raise SystemExit(1)
    print(f"[TABLE OK] {tf.name}")

# 2. Re-sync SHA-256 manifest & update Table S7 hashes in supplement.tex
m_path = REPO / "06_experiments_results/sha256_manifest.json"
m_data = json.loads(m_path.read_text(encoding="utf-8"))
rows_s7 = []
for rec in m_data.get("verified_artifacts", []):
    tp = Path(rec["latex_table_path"])
    if not tp.is_absolute():
        tp = REPO / tp
    if tp.exists():
        rec["latex_table_sha256"] = sm.sha256_digest(tp)
    eid = str(rec["experiment_id"]).replace("_", r"\_")
    tnm = tp.name.replace("_", r"\_")
    cs, ts = str(rec["raw_csv_sha256"]), str(rec["latex_table_sha256"])
    rows_s7.append(f"\\texttt{{{eid}}} & \\texttt{{{tnm}}} & ${int(rec['measured_rows']):,}$ & \\texttt{{{cs[:16]}...{cs[-12:]}}} & \\texttt{{{ts[:16]}...{ts[-12:]}}} \\\\")
m_path.write_text(json.dumps(m_data, indent=2), encoding="utf-8", newline="\n")

supp_p = REPO / "supplement.tex"
supp_txt = supp_p.read_text(encoding="utf-8")
s7_pat = r"(\\label\{tab:supp_sha256\}.*?\\midrule\n)(.*?)(\n\\bottomrule)"
supp_txt = re.sub(s7_pat, lambda m: m.group(1) + "\n".join(rows_s7) + m.group(3), supp_txt, flags=re.DOTALL)
supp_p.write_text(supp_txt, encoding="utf-8", newline="\n")

# 3. Compile both supplement.pdf and main.pdf cleanly
for stem in ["supplement", "main"]:
    cap = stem.capitalize()
    sh = f"""
set -e
rm -rf /tmp/capq_final_{stem} && mkdir -p /tmp/capq_final_{stem}
cp /mnt/d/Research/CAPQ-Pareto/{stem}.tex /mnt/d/Research/CAPQ-Pareto/references.bib /tmp/capq_final_{stem}/
cp -r /mnt/d/Research/CAPQ-Pareto/00_manuscript /tmp/capq_final_{stem}/
cd /tmp/capq_final_{stem}
pdflatex -interaction=nonstopmode -halt-on-error {stem}.tex > build.log 2>&1 || (tail -n 30 build.log && exit 1)
if grep -q "\\\\citation" {stem}.aux; then
    bibtex {stem} >> build.log 2>&1 || (cat {stem}.blg && exit 1)
fi
pdflatex -interaction=nonstopmode -halt-on-error {stem}.tex >> build.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error {stem}.tex >> build.log 2>&1
PAGES=$(pdfinfo {stem}.pdf | awk '/^Pages:/ {{print $2}}')
pdftotext {stem}.pdf rendered.txt
cp {stem}.pdf /mnt/d/Research/CAPQ-Pareto-{cap}-IEEE-TCC.pdf
echo "[PDF OK] D:\\Research\\CAPQ-Pareto-{cap}-IEEE-TCC.pdf -> $PAGES pages"
"""
    r = subprocess.run(["wsl", "-u", "root", "-e", "bash", "-c", sh], text=True, capture_output=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit(1)
    print(r.stdout.strip())

s_txt = subprocess.check_output(["wsl", "-u", "root", "-e", "cat", "/tmp/capq_final_supplement/rendered.txt"], text=True, errors="replace")
m_txt = subprocess.check_output(["wsl", "-u", "root", "-e", "cat", "/tmp/capq_final_main/rendered.txt"], text=True, errors="replace")
s_tabs = sorted({" ".join(x.split()) for x in re.findall(r"TABLE\s+S\d+", s_txt)}, key=lambda v: int(v.split("S")[1]))
m_tabs = sorted({" ".join(x.split()) for x in re.findall(r"TABLE\s+\d+", m_txt)}, key=lambda v: int(re.search(r"\d+", v).group()))
print(f"[VERIFIED] Main Tables: {m_tabs} | Supplement Tables: {s_tabs}")

# 4. Overwrite Overleaf ZIP
zip_p = Path(r"D:\Research\CAPQ-Pareto-Overleaf-IEEE-TCC.zip")
rels = ["main.tex", "supplement.tex", "references.bib", "README.md", "WORKFLOW_CHECKLIST.md", "06_experiments_results/sha256_manifest.json"]
for d in ["00_manuscript/figures", "00_manuscript/tables"]:
    rels.extend(f.relative_to(REPO).as_posix() for f in sorted((REPO / d).rglob("*")) if f.is_file())
with zipfile.ZipFile(zip_p, "w", zipfile.ZIP_DEFLATED) as zf:
    for rel in rels:
        zf.write(REPO / rel, arcname=rel)
print(f"[ZIP OK] {zip_p} ({zip_p.stat().st_size:,} bytes | SHA-256: {sm.sha256_digest(zip_p)})")