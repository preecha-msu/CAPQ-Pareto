import hashlib
import html
import pathlib
import re
import shutil
import subprocess
import zipfile

v6_root = pathlib.Path(r"D:\Research\CAPQ-Pareto-V6")
res_dir = pathlib.Path(r"D:\Research")
ms_v6   = v6_root / "08_manuscript_ieee_tcc_v6"
fig_dir = ms_v6 / "figures"
tbl_dir = ms_v6 / "tables"
prev_dir = v6_root / "11_v6_workspace" / "pdf_page_previews"
prev_dir.mkdir(parents=True, exist_ok=True)

def sha256_64(p: pathlib.Path) -> str:
    d = hashlib.sha256(p.read_bytes()).hexdigest()
    assert len(d) == 64
    return d

v5_locked = {p.name: sha256_64(p) for p in sorted(res_dir.glob("*V5*")) if p.is_file()}

# 1. Ensure Section 3 in supplement.tex sits directly above its SHA-256 table with zero vertical gap
supp_p = ms_v6 / "supplement.tex"
s_txt = supp_p.read_text(encoding="utf-8")
s_txt = s_txt.replace(
    "\\FloatBarrier\n\\clearpage\n\\section{Full 64-Character SHA-256 Provenance Manifest}",
    "\\FloatBarrier\n\\onecolumn\n\\section{Full 64-Character SHA-256 Provenance Manifest}",
)
s_txt = re.sub(
    r"(\\section\{Full 64-Character SHA-256 Provenance Manifest\}.*?)\\begin\{table\*\}\[!b\]",
    r"\1\\begin{table}[!ht]",
    s_txt,
    count=1,
    flags=re.S,
)
# Match closing \end{table*} at the very end of Section 3 if converted to \begin{table}[!ht]
if "\\begin{table}[!ht]" in s_txt:
    idx_sec3 = s_txt.index("\\section{Full 64-Character SHA-256 Provenance Manifest}")
    s_txt = s_txt[:idx_sec3] + s_txt[idx_sec3:].replace("\\end{table*}", "\\end{table}", 1)
supp_p.write_text(s_txt, encoding="utf-8", newline="\n")

print("=== [1/3] COMPILING ALL 4 V6 PDFs & MEASURING EXACT SECTION SPACING (REGEX BBOX PARSER) ===")
wsl_ms = "/mnt/d/Research/CAPQ-Pareto-V6/08_manuscript_ieee_tcc_v6"
wsl_prev = "/mnt/d/Research/CAPQ-Pareto-V6/11_v6_workspace/pdf_page_previews"

def compile_and_measure(stem: str):
    bib_cmd = f"bibtex {stem} >> build.log 2>&1 || (cat {stem}.blg && exit 1)" if stem == "main" else ""
    ppm_cmd = f"rm -f '{wsl_prev}/{stem}_page'*.png && pdftoppm -png -r 150 {stem}.pdf '{wsl_prev}/{stem}_page'" if stem in ("main", "supplement") else ""
    sh_cmd = f"""
set -e
rm -rf /tmp/capq_v6_sp_{stem} && mkdir -p /tmp/capq_v6_sp_{stem}
cp -a "{wsl_ms}/." /tmp/capq_v6_sp_{stem}/
cd /tmp/capq_v6_sp_{stem}
pdflatex -interaction=nonstopmode -halt-on-error {stem}.tex > build.log 2>&1 || (tail -n 50 build.log && exit 1)
{bib_cmd}
pdflatex -interaction=nonstopmode -halt-on-error {stem}.tex >> build.log 2>&1 || (tail -n 50 build.log && exit 1)
pdflatex -interaction=nonstopmode -halt-on-error {stem}.tex >> build.log 2>&1 || (tail -n 50 build.log && exit 1)
pdftotext -bbox-layout {stem}.pdf {stem}_bbox.html
{ppm_cmd}
cp -f {stem}.pdf {stem}.log {stem}_bbox.html "{wsl_ms}/"
"""
    r = subprocess.run(["wsl", "-u", "root", "-e", "bash", "-c", sh_cmd], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(f"COMPILE ERROR ({stem}):\n{r.stdout}\n{r.stderr}")
        raise SystemExit(1)

    log_txt = (ms_v6 / f"{stem}.log").read_text(encoding="utf-8", errors="ignore")
    pgs = int(re.search(r"Output written on .*?\((\d+)\s+pages?", log_txt, re.S).group(1))
    ov_h = [float(x) for x in re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", log_txt)]
    ov_v = [float(x) for x in re.findall(r"Overfull \\vbox \(([0-9.]+)pt too high\)", log_txt)]
    undefs = re.findall(r"LaTeX Warning: (?:Reference|Citation) `([^']+)' .*? undefined", log_txt)
    assert not undefs, f"Undefined refs in {stem}: {undefs}"
    assert (max(ov_h) if ov_h else 0.0) == 0.0, f"Overfull hbox in {stem}: {ov_h}"
    assert (max(ov_v) if ov_v else 0.0) == 0.0, f"Overfull vbox in {stem}: {ov_v}"

    bbox_p = ms_v6 / f"{stem}_bbox.html"
    headings_out = []
    if bbox_p.exists() and stem in ("main", "supplement"):
        raw_h = bbox_p.read_text(encoding="utf-8", errors="ignore")
        for p_idx, pm in enumerate(re.finditer(r'<page\s+width="([\d.]+)"\s+height="([\d.]+)"[^>]*>(.*?)</page>', raw_h, re.S | re.I), 1):
            p_height = float(pm.group(2))
            p_body = pm.group(3)
            lines_info = []
            for lm in re.finditer(r'<line\s+xMin="([\d.]+)"\s+yMin="([\d.]+)"\s+xMax="([\d.]+)"\s+yMax="([\d.]+)"[^>]*>(.*?)</line>', p_body, re.S | re.I):
                xmin, ymin, xmax, ymax = float(lm.group(1)), float(lm.group(2)), float(lm.group(3)), float(lm.group(4))
                if ymin < 48.0 or ymax > p_height - 32.0:
                    continue  # Exclude top running header & bottom folio
                words = [html.unescape(re.sub(r"<[^>]+>", "", w)) for w in re.findall(r"<word[^>]*>(.*?)</word>", lm.group(5), re.S | re.I)]
                t_str = " ".join(w for w in words if w).strip()
                if t_str:
                    lines_info.append((xmin, ymin, xmax, ymax, t_str))

            for idx_l, (xmin, ymin, xmax, ymax, t_str) in enumerate(lines_info):
                is_sec = bool(re.match(r"^(?:[1-9](?:\.[1-9])?\s+(?:Introduction|Related|System|Workload|Transport|Explicit|Public|True|Coordinate|Plaintext|Online|Bandit|Silicon|Factorial|2|Provenance|Limitations|Conclusion|Complete|Extended|Full)|Acknowledgment,\s+Data)", t_str, re.I))
                if is_sec:
                    col_above = [l for l in lines_info[:idx_l] if abs(l[0] - xmin) < 130.0 and l[3] <= ymin + 0.5]
                    col_below = [l for l in lines_info[idx_l + 1:] if abs(l[0] - xmin) < 130.0 and l[1] >= ymax - 0.5]
                    gap_up = round(ymin - max(l[3] for l in col_above), 1) if col_above else -1.0
                    gap_dn = round(min(l[1] for l in col_below) - ymax, 1) if col_below else -1.0
                    dist_bot = round(p_height - ymax, 1)
                    assert dist_bot > 45.0, f"Orphan heading at bottom of {stem} p.{p_idx}: {t_str}"
                    headings_out.append((p_idx, t_str[:46], gap_up, gap_dn))
        bbox_p.unlink()
    elif bbox_p.exists():
        bbox_p.unlink()

    return pgs, headings_out

for st in ("main", "supplement", "response_to_reviewers", "cover_letter"):
    pg_cnt, h_list = compile_and_measure(st)
    print(f"  [VERIFIED {st+'.pdf':<26}] Pages={pg_cnt:2d} | OverfullHbox=0.00pt | OverfullVbox=0.00pt | Undefined=0")
    for p_no, h_txt, g_up, g_dn in h_list:
        up_str = f"{g_up:5.1f}pt" if g_up >= 0 else "col-top"
        dn_str = f"{g_dn:5.1f}pt" if g_dn >= 0 else "next-ln"
        print(f"    p.{p_no:02d} | {h_txt:<46} | gap_above={up_str} | gap_below={dn_str}")

png_previews = sorted(p.name for p in prev_dir.glob("*.png"))
print(f"\n=== [2/3] SAVED {len(png_previews)} PAGE PREVIEW PNGs IN 11_v6_workspace/pdf_page_previews ===")

print("\n=== [3/3] EXPORTING ALL 7 VERIFIED V6 DELIVERABLES TO D:\\Research\\ ===")
shutil.copy2(ms_v6 / "main.pdf",                  res_dir / "CAPQ-Pareto-Main-IEEE-TCC-V6.pdf")
shutil.copy2(ms_v6 / "supplement.pdf",            res_dir / "CAPQ-Pareto-Supplement-IEEE-TCC-V6.pdf")
shutil.copy2(ms_v6 / "response_to_reviewers.pdf", res_dir / "CAPQ-Pareto-Response-to-Reviewers-V6.pdf")
shutil.copy2(ms_v6 / "cover_letter.pdf",          res_dir / "CAPQ-Pareto-Cover-Letter-IEEE-TCC-V6.pdf")
shutil.copy2(fig_dir / "fig1_architecture.png",   res_dir / "CAPQ-Pareto-Fig1-Preview-V6.png")

zip_v6 = res_dir / "CAPQ-Pareto-Overleaf-IEEE-TCC-V6.zip"
with zipfile.ZipFile(zip_v6, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for fn in ("main.tex", "supplement.tex", "references.bib", "response_to_reviewers.tex", "cover_letter.tex"):
        zf.write(ms_v6 / fn, arcname=fn)
    for tf in sorted(tbl_dir.glob("*.tex")):
        zf.write(tf, arcname=f"tables/{tf.name}")
    for ff in sorted(fig_dir.glob("*.*")):
        zf.write(ff, arcname=f"figures/{ff.name}")

v6_delivs = [
    "CAPQ-Pareto-Main-IEEE-TCC-V6.pdf",
    "CAPQ-Pareto-Supplement-IEEE-TCC-V6.pdf",
    "CAPQ-Pareto-Response-to-Reviewers-V6.pdf",
    "CAPQ-Pareto-Cover-Letter-IEEE-TCC-V6.pdf",
    "CAPQ-Pareto-Fig1-Preview-V6.png",
    "CAPQ-Pareto-Overleaf-IEEE-TCC-V6.zip",
]
manifest_p = res_dir / "CAPQ-Pareto-SHA256-Manifest-V6.txt"
m_lines = ["# CAPQ-Pareto V6 Final Verified Deliverables SHA-256 Manifest (Full 64-Hex)"]
for dn in v6_delivs:
    dp = res_dir / dn
    h64 = sha256_64(dp)
    m_lines.append(f"{h64}  {dn}")
    print(f"  {dn:<44} | {dp.stat().st_size:>9,} B | SHA256={h64}")
manifest_p.write_text("\n".join(m_lines) + "\n", encoding="utf-8", newline="\n")
print(f"  {manifest_p.name:<44} | {manifest_p.stat().st_size:>9,} B | SHA256={sha256_64(manifest_p)}")

for fn_v5, h_pre in v5_locked.items():
    assert sha256_64(res_dir / fn_v5) == h_pre, f"V5 Zero-Touch violated on {fn_v5}!"
print("\n[ZERO-TOUCH V5 VERIFIED 100%] All V5 files in D:\\Research\\ remain untouched byte-for-byte!")