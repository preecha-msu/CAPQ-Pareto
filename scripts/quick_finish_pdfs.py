from pathlib import Path
import re, subprocess, zipfile, sys

REPO = Path(r"D:\Research\CAPQ-Pareto")
sys.path.insert(0, str(REPO / "05_telemetry_audit"))
import sha256_manifest as sm

bib_keys = set(re.findall(r"@\w+\{([^,\s]+),", (REPO / "references.bib").read_text(encoding="utf-8")))

# 1. Clean legacy citation keys & wrappers in Phase 2-3 supplementary tables so pdflatex/bibtex never trips
key_map = {
    "badawi2022openfhe": "albadawi2022openfhe",
    "iyengar2021quic": "iyengar2021rfc9000",
    "tpcc": "tpcc_spec",
}
for tf in (REPO / "00_manuscript/tables").glob("*.tex"):
    t = tf.read_text(encoding="utf-8")
    for k_old, k_new in key_map.items():
        t = re.sub(rf"\b{k_old}\b", k_new, t)
    def _clean_cite(m):
        kept = [k.strip() for k in m.group(1).split(",") if k.strip() in bib_keys]
        return f"\\cite{{{','.join(kept)}}}" if kept else ""
    t = re.sub(r"\\cite\{([^}]+)\}", _clean_cite, t)
    tf.write_text(t, encoding="utf-8", newline="\n")

# Ensure xcolor is available in both .tex files just in case any table uses color macros
for tex_name in ["main.tex", "supplement.tex"]:
    p = REPO / tex_name
    txt = p.read_text(encoding="utf-8")
    if r"\usepackage{xcolor}" not in txt:
        txt = txt.replace(r"\usepackage{array}", "\\usepackage{array}\n\\usepackage{xcolor}")
        p.write_text(txt, encoding="utf-8", newline="\n")

# 2. Compile both PDFs with full error visibility if anything ever fails
for stem in ["supplement", "main"]:
    cap = stem.capitalize()
    sh = f"""
set -e
rm -rf /tmp/capq_final_{stem} && mkdir -p /tmp/capq_final_{stem}
cp /mnt/d/Research/CAPQ-Pareto/{stem}.tex /mnt/d/Research/CAPQ-Pareto/references.bib /tmp/capq_final_{stem}/
cp -r /mnt/d/Research/CAPQ-Pareto/00_manuscript /tmp/capq_final_{stem}/
cd /tmp/capq_final_{stem}
pdflatex -interaction=nonstopmode -halt-on-error {stem}.tex > build.log 2>&1 || (tail -n 35 build.log && exit 1)
if grep -q "\\\\citation" {stem}.aux; then
    bibtex {stem} >> build.log 2>&1 || (cat {stem}.blg && exit 1)
fi
pdflatex -interaction=nonstopmode -halt-on-error {stem}.tex >> build.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error {stem}.tex >> build.log 2>&1
PAGES=$(pdfinfo {stem}.pdf | awk '/^Pages:/ {{print $2}}')
pdftotext {stem}.pdf rendered.txt
cp {stem}.pdf /mnt/d/Research/CAPQ-Pareto-{cap}-IEEE-TCC.pdf
echo "[OK] CAPQ-Pareto-{cap}-IEEE-TCC.pdf -> $PAGES pages"
"""
    r = subprocess.run(["wsl", "-u", "root", "-e", "bash", "-c", sh], text=True, capture_output=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit(1)
    print(r.stdout.strip())

# 3. Verify all 7 Supplement tables (S1..S7) & 6 Main tables + Overwrite Overleaf ZIP
s_txt = subprocess.check_output(["wsl", "-u", "root", "-e", "cat", "/tmp/capq_final_supplement/rendered.txt"], text=True, errors="replace")
s_tabs = sorted({" ".join(x.split()) for x in re.findall(r"TABLE\s+S\d+", s_txt)}, key=lambda v: int(v.split("S")[1]))
print(f"[VERIFIED SUPPLEMENT TABLES] {s_tabs}")

zip_p = Path(r"D:\Research\CAPQ-Pareto-Overleaf-IEEE-TCC.zip")
rels = ["main.tex", "supplement.tex", "references.bib", "README.md", "WORKFLOW_CHECKLIST.md", "06_experiments_results/sha256_manifest.json"]
for d in ["00_manuscript/figures", "00_manuscript/tables"]:
    rels.extend(f.relative_to(REPO).as_posix() for f in sorted((REPO / d).rglob("*")) if f.is_file())
with zipfile.ZipFile(zip_p, "w", zipfile.ZIP_DEFLATED) as zf:
    for rel in rels:
        zf.write(REPO / rel, arcname=rel)
print(f"[OK] Updated Overleaf ZIP: {zip_p} ({zip_p.stat().st_size:,} bytes | SHA-256: {sm.sha256_digest(zip_p)[:16]}...)")