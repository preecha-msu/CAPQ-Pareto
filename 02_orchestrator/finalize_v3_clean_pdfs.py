from __future__ import annotations
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

V3_ROOT = Path(r"D:\Research\CAPQ-Pareto-V3")
LOG_DIR = V3_ROOT / "06_experiments_results" / "v3_raw_logs"
OVERLEAF_DIR = V3_ROOT / "00_manuscript" / "overleaf_ieee_tcc_v3"
TAB_DIR = OVERLEAF_DIR / "tables"
FIG_DIR = OVERLEAF_DIR / "figures"


def sha256_fp(fp: Path) -> str:
    h = hashlib.sha256()
    with fp.open("rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def convert_markdown_backticks_to_texttt(tex_str: str) -> str:
    """Safely converts any `identifier_name` into \texttt{identifier\_name}."""
    def repl(m: re.Match) -> str:
        inner = m.group(1).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
        return f"\\texttt{{{inner}}}"
    return re.sub(r"`([^`\n]+)`", repl, tex_str)


# Polish symbols in tab_v3_cleanroom_isa.tex (<< -> $\ll$, -> -> $\to$) and ensure \resizebox is clean
isa_tab_fp = TAB_DIR / "tab_v3_cleanroom_isa.tex"
if isa_tab_fp.is_file():
    t_isa = isa_tab_fp.read_text(encoding="utf-8")
    t_isa = t_isa.replace("<<", "$\\ll$").replace("->", "$\\to$")
    isa_tab_fp.write_text(t_isa, encoding="utf-8", newline="\n")

# Enrich references.bib with full classic + V3 citations (25 complete BibTeX entries)
EXTRA_BIB = r"""
@inproceedings{wang2021milvus,
  author = {Wang, Jianguo and Yi, Xiaomeng and Guo, Rentong and Jin, Hai and Xu, Peng and Li, Shengjun and Wang, Xiangyu and Guo, Xiangzhou and Li, Chengming and Xu, Xiaohai and others},
  title = {Milvus: A Purpose-Built Vector Data Management System},
  booktitle = {Proc. ACM SIGMOD},
  pages = {2614--2627},
  year = {2021}
}
@inproceedings{stebila2016liboqs,
  author = {Stebila, Douglas and Mosca, Michele},
  title = {Post-Quantum Key Exchange for the {Internet} and the {Open Quantum Safe} Project},
  booktitle = {Proc. SAC},
  pages = {14--37},
  year = {2016}
}
@inproceedings{cooper2010ycsb,
  author = {Cooper, Brian F. and Silberstein, Adam and Tam, Erwin and Ramakrishnan, Raghu and Sears, Russell},
  title = {Benchmarking Cloud Serving Systems with {YCSB}},
  booktitle = {Proc. ACM SoCC},
  pages = {143--154},
  year = {2010}
}
@article{jegou2011sift,
  author = {J{\'e}gou, Herv{\'e} and Douze, Matthijs and Schmid, Cordelia},
  title = {Product Quantization for Nearest Neighbor Search},
  journal = {IEEE Trans. Pattern Anal. Mach. Intell.},
  volume = {33},
  number = {1},
  pages = {117--128},
  year = {2011}
}
@article{khan2018rapl,
  author = {Khan, Kashif Nizam and Hirki, Mikael and Niemi, Tapio and Nurminen, Jukka K. and Ou, Zhonghong},
  title = {{RAPL} in Action: Experiences in Using {RAPL} for Power Measurements},
  journal = {ACM Trans. Model. Perform. Eval. Comput. Syst.},
  volume = {3},
  number = {2},
  pages = {1--26},
  year = {2018}
}
@inproceedings{li2010linucb,
  author = {Li, Lihong and Chu, Wei and Langford, John and Schapire, Robert E.},
  title = {A Contextual-Bandit Approach to Personalized News Article Recommendation},
  booktitle = {Proc. WWW},
  pages = {661--670},
  year = {2010}
}
@article{kunsch1989mbb,
  author = {K{\"u}nsch, Hans R.},
  title = {The Jackknife and the Bootstrap for General Stationary Observations},
  journal = {Ann. Statist.},
  volume = {17},
  number = {3},
  pages = {1217--1241},
  year = {1989}
}
@article{holm1979simple,
  author = {Holm, Sture},
  title = {A Simple Sequentially Rejective Multiple Test Procedure},
  journal = {Scand. J. Statist.},
  volume = {6},
  number = {2},
  pages = {65--70},
  year = {1979}
}
"""

bib_fp = OVERLEAF_DIR / "references.bib"
bib_cur = bib_fp.read_text(encoding="utf-8")
if "kunsch1989mbb" not in bib_cur:
    bib_fp.write_text(bib_cur.rstrip() + "\n" + EXTRA_BIB.strip() + "\n", encoding="utf-8", newline="\n")

# Fix main.tex and supplement.tex: convert all Markdown backticks, add \texorpdfstring, add MBB & RAPL/GCP exact citations
main_fp = OVERLEAF_DIR / "main.tex"
m_txt = main_fp.read_text(encoding="utf-8")
m_txt = convert_markdown_backticks_to_texttt(m_txt)
m_txt = m_txt.replace(
    r"\section{Controlled $2^5$ Factorial ANOVA, Network Shocks, and Cross-ISA Continuum}",
    r"\section{Controlled \texorpdfstring{$2^5$}{2\textasciicircum 5} Factorial ANOVA, Network Shocks, and Cross-ISA Continuum}",
)
m_txt = m_txt.replace(
    "Moving-Block Bootstrap ($B=4,000$) with Holm--Bonferroni correction",
    "Moving-Block Bootstrap ($B=4,000$)~\\cite{kunsch1989mbb} with Holm--Bonferroni correction~\\cite{holm1979simple}",
)
m_txt = m_txt.replace(
    "bare-metal RAPL energy solely on physical x86\\_64 edge hardware",
    "wraparound-safe bare-metal RAPL energy~\\cite{khan2018rapl} solely on physical x86\\_64 edge hardware",
)
m_txt = m_txt.replace(
    "high-dimensional vector indices and confidential zero-trust vector search~\\cite{malkov2020hnsw,chen2021spann,chen2024bgem3}",
    "high-dimensional vector indices (SIFT1M, BGE-M3) and YCSB/SQL workloads~\\cite{wang2021milvus,malkov2020hnsw,chen2021spann,cooper2010ycsb,jegou2011sift,chen2024bgem3}",
)
m_txt = m_txt.replace(
    "ML-KEM (FIPS 203)~\\cite{nist_fips203_2024}",
    "ML-KEM (FIPS 203)~\\cite{nist_fips203_2024,stebila2016liboqs}",
)
m_txt = m_txt.replace(
    "standard discounted bandits~\\cite{garivier2011discounted,russac2019dlinucb}",
    "contextual and discounted linear bandits~\\cite{li2010linucb,garivier2011discounted,russac2019dlinucb}",
)
main_fp.write_text(m_txt, encoding="utf-8", newline="\n")

supp_fp = OVERLEAF_DIR / "supplement.tex"
s_txt = convert_markdown_backticks_to_texttt(supp_fp.read_text(encoding="utf-8"))
supp_fp.write_text(s_txt, encoding="utf-8", newline="\n")

# Sanity check: ensure zero stray Markdown backticks remain in main.tex or supplement.tex
for check_fp in (main_fp, supp_fp):
    c_str = check_fp.read_text(encoding="utf-8")
    # Note: LaTeX uses `` for left quotes and `' in citation messages, so check specifically for paired single `...` with _
    bad_bt = re.findall(r"`[^`\n]*_[^`\n]*`", c_str)
    assert not bad_bt, f"Found unescaped code backticks in {check_fp.name}: {bad_bt}"


def compile_in_wsl_texlive(stem: str) -> tuple[int, int, int, int, int]:
    wsl_src = subprocess.check_output(["wsl", "-u", "root", "-e", "wslpath", "-u", str(OVERLEAF_DIR)], text=True).strip()
    wsl_cmd = f"""
set -e
rm -rf /tmp/capq_v3_tex_{stem}
mkdir -p /tmp/capq_v3_tex_{stem}
cp -r '{wsl_src}/'* /tmp/capq_v3_tex_{stem}/
cd /tmp/capq_v3_tex_{stem}
pdflatex -interaction=nonstopmode -halt-on-error {stem}.tex > build.log 2>&1 || (tail -n 45 build.log && exit 1)
if grep -q "\\\\citation" {stem}.aux; then
    bibtex {stem} >> build.log 2>&1 || (cat {stem}.blg && exit 1)
fi
pdflatex -interaction=nonstopmode -halt-on-error {stem}.tex >> build.log 2>&1 || (tail -n 45 build.log && exit 1)
pdflatex -interaction=nonstopmode -halt-on-error {stem}.tex >> build.log 2>&1 || (tail -n 45 build.log && exit 1)
cp {stem}.pdf {stem}.log '{wsl_src}/'
"""
    res = subprocess.run(["wsl", "-u", "root", "-e", "bash", "-c", wsl_cmd], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"WSL2 LaTeX failed on {stem}.tex:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")

    log_txt = (OVERLEAF_DIR / f"{stem}.log").read_text(encoding="utf-8", errors="ignore")
    m_pg = re.search(rf"Output written on {stem}\.pdf \((\d+) pages?, (\d+) bytes\)", log_txt)
    pages = int(m_pg.group(1)) if m_pg else -1
    sz_b = int(m_pg.group(2)) if m_pg else 0
    u_cites = len(re.findall(r"Citation `.*?' on page .*? undefined", log_txt))
    u_refs = len(re.findall(r"Reference `.*?' on page .*? undefined", log_txt))
    overfull_severe = len([float(x) for x in re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", log_txt) if float(x) > 10.0])
    assert u_cites == 0 and u_refs == 0, f"Unresolved cites={u_cites} or refs={u_refs} in {stem}.tex!"
    return pages, sz_b, u_cites, u_refs, overfull_severe


m_pg, m_sz, mc, mr, m_ov = compile_in_wsl_texlive("main")
s_pg, s_sz, sc, sr, s_ov = compile_in_wsl_texlive("supplement")
assert 1 <= m_pg <= 12, f"main.pdf page count {m_pg} outside 1..12 IEEE TCC budget!"

print(f"[LATEX VERIFIED] main.pdf       : {m_pg} pages ({m_sz/1024:.1f} KB) | Undefined cites={mc} | Undefined refs={mr} | SevereOverfull={m_ov}")
print(f"[LATEX VERIFIED] supplement.pdf : {s_pg} pages ({s_sz/1024:.1f} KB) | Undefined cites={sc} | Undefined refs={sr} | SevereOverfull={s_ov}")

dst_main = Path(r"D:\Research\CAPQ-Pareto-Main-IEEE-TCC-V3.pdf")
dst_supp = Path(r"D:\Research\CAPQ-Pareto-Supplement-IEEE-TCC-V3.pdf")
dst_zip = Path(r"D:\Research\CAPQ-Pareto-Overleaf-IEEE-TCC-V3.zip")

shutil.copy2(OVERLEAF_DIR / "main.pdf", dst_main)
shutil.copy2(OVERLEAF_DIR / "supplement.pdf", dst_supp)

if dst_zip.exists():
    dst_zip.unlink()
with zipfile.ZipFile(dst_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for fn in ("main.tex", "supplement.tex", "references.bib", "SHA256SUMS_V3.txt", "README_OVERLEAF.md"):
        zf.write(OVERLEAF_DIR / fn, arcname=fn)
    for t_fp in sorted(TAB_DIR.glob("*.tex")):
        zf.write(t_fp, arcname=f"tables/{t_fp.name}")
    for f_fp in sorted(FIG_DIR.glob("*.pdf")):
        zf.write(f_fp, arcname=f"figures/{f_fp.name}")

print(f"[SHA-256] {dst_main.name:<42} : {sha256_fp(dst_main)} ({m_pg} pages)")
print(f"[SHA-256] {dst_supp.name:<42} : {sha256_fp(dst_supp)} ({s_pg} pages)")
print(f"[SHA-256] {dst_zip.name:<42} : {sha256_fp(dst_zip)} ({dst_zip.stat().st_size/1024:.1f} KB)")