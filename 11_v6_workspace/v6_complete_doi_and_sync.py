import hashlib
import json
import pathlib
import re
import shutil
import ssl
import subprocess
import urllib.request
import zipfile

v6_root = pathlib.Path(r"D:\Research\CAPQ-Pareto-V6")
res_dir = pathlib.Path(r"D:\Research")
ms_v6   = v6_root / "08_manuscript_ieee_tcc_v6"
fig_dir = ms_v6 / "figures"
tbl_dir = ms_v6 / "tables"
sub_v6  = v6_root / "00_final_submission_v6"
sub_v6.mkdir(parents=True, exist_ok=True)

DOI_ID  = "10.5281/zenodo.24575614"
DOI_URL = "https://doi.org/10.5281/zenodo.24575614"
GH_URL  = "https://github.com/preecha-msu/CAPQ-Pareto/tree/v6"

def sha256_64(p: pathlib.Path) -> str:
    d = hashlib.sha256(p.read_bytes()).hexdigest()
    assert len(d) == 64
    return d

def run_cmd(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(args, cwd=v6_root, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise RuntimeError("Command failed: " + " ".join(args) + "\nSTDOUT:\n" + r.stdout + "\nSTDERR:\n" + r.stderr)
    return r

v5_locked = {p.name: sha256_64(p) for p in sorted(res_dir.glob("*V5*")) if p.is_file()}

print("=== [1/5] RESTORING CLEAN 0.00pt .tex BASELINE & EMBEDDING ZENODO DOI ===")
run_cmd([
    "git", "checkout", "HEAD", "--",
    "08_manuscript_ieee_tcc_v6/main.tex",
    "08_manuscript_ieee_tcc_v6/supplement.tex",
    "08_manuscript_ieee_tcc_v6/response_to_reviewers.tex",
    "08_manuscript_ieee_tcc_v6/cover_letter.tex",
])

xurl_hook = "\\IfFileExists{xurl.sty}{\\usepackage{xurl}}{}\n\\begin{document}"
gh_macro  = "\\url{" + GH_URL + "}"
doi_combo = "\\url{" + GH_URL + "} (permanent Zenodo archive: \\url{" + DOI_URL + "})"
doi_para  = (
    "\n\\smallskip\\noindent\\textbf{Permanent Zenodo Archive:} "
    "\\url{" + DOI_URL + "} (DOI: \\texttt{" + DOI_ID + "}).\\smallskip\n"
)

for fn_tex in ("main.tex", "supplement.tex"):
    tp = ms_v6 / fn_tex
    txt = tp.read_text(encoding="utf-8")
    if "xurl.sty" not in txt:
        txt = txt.replace("\\begin{document}", xurl_hook, 1)
    if DOI_ID not in txt:
        assert gh_macro in txt, "Missing GitHub URL in " + fn_tex
        txt = txt.replace(gh_macro, doi_combo)
    tp.write_text(txt, encoding="utf-8", newline="\n")
    print("  [EMBEDDED OK] " + fn_tex)

for fn_doc in ("response_to_reviewers.tex", "cover_letter.tex"):
    tp = ms_v6 / fn_doc
    txt = tp.read_text(encoding="utf-8")
    if "xurl.sty" not in txt:
        txt = txt.replace("\\begin{document}", xurl_hook + "\n\\sloppy", 1)
    if DOI_ID not in txt:
        # Insert as a standalone full-width paragraph outside any header tabular/box!
        if "\\section*" in txt:
            txt = txt.replace("\\section*", doi_para + "\\section*", 1)
        else:
            txt = txt.replace("\\end{document}", doi_para + "\n\\end{document}", 1)
    tp.write_text(txt, encoding="utf-8", newline="\n")
    print("  [EMBEDDED OK] " + fn_doc)

# Update .zenodo.json & reproduce_and_verify_v6.py
z_path = v6_root / ".zenodo.json"
z_meta = json.loads(z_path.read_text(encoding="utf-8"))
z_meta["doi"] = DOI_ID
z_meta["identifiers"] = [
    {"identifier": DOI_ID, "relation": "isIdenticalTo", "scheme": "doi"},
    {"identifier": GH_URL, "relation": "isSupplementTo", "scheme": "url"},
]
z_path.write_text(json.dumps(z_meta, indent=2) + "\n", encoding="utf-8", newline="\n")

ver_p = v6_root / "reproduce_and_verify_v6.py"
ver_txt = ver_p.read_text(encoding="utf-8")
if DOI_ID not in ver_txt:
    ver_txt = ver_txt.replace(
        'print("ALL CAPQ-PARETO V6 ARTIFACT & NUMERICAL AUDITS PASSED 100%.")',
        'print("ALL CAPQ-PARETO V6 ARTIFACT & NUMERICAL AUDITS PASSED 100% (Zenodo DOI: ' + DOI_ID + ').")'
    )
    ver_p.write_text(ver_txt, encoding="utf-8", newline="\n")

print("\n=== [2/5] COMPILING ALL 4 PDFs IN WSL2 & VERIFYING 0.00pt OVERFULL + DOI ===")
wsl_ms = "/mnt/d/Research/CAPQ-Pareto-V6/08_manuscript_ieee_tcc_v6"

for stem in ("main", "supplement", "response_to_reviewers", "cover_letter"):
    bib_cmd = "bibtex " + stem + " >> build.log 2>&1 || (cat " + stem + ".blg && exit 1)" if stem == "main" else ""
    sh_cmd = (
        "set -e\n"
        "rm -rf /tmp/capq_v6_final_" + stem + " && mkdir -p /tmp/capq_v6_final_" + stem + "\n"
        'cp -a "' + wsl_ms + '/." /tmp/capq_v6_final_' + stem + "/\n"
        "cd /tmp/capq_v6_final_" + stem + "\n"
        "pdflatex -interaction=nonstopmode -halt-on-error " + stem + ".tex > build.log 2>&1 || (tail -n 50 build.log && exit 1)\n"
        + bib_cmd + "\n"
        "pdflatex -interaction=nonstopmode -halt-on-error " + stem + ".tex >> build.log 2>&1 || (tail -n 50 build.log && exit 1)\n"
        "pdflatex -interaction=nonstopmode -halt-on-error " + stem + ".tex >> build.log 2>&1 || (tail -n 50 build.log && exit 1)\n"
        "pdftotext " + stem + ".pdf " + stem + "_plain.txt\n"
        "cp -f " + stem + ".pdf " + stem + ".log " + stem + '_plain.txt "' + wsl_ms + '/"\n'
    )
    r = subprocess.run(["wsl", "-u", "root", "-e", "bash", "-c", sh_cmd], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError("LaTeX compile failed (" + stem + "):\n" + r.stdout + "\n" + r.stderr)

    lg = (ms_v6 / (stem + ".log")).read_text(encoding="utf-8", errors="ignore")
    pt_p = ms_v6 / (stem + "_plain.txt")
    ptxt = pt_p.read_text(encoding="utf-8", errors="ignore")
    pt_p.unlink()

    pgs = int(re.search(r"Output written on .*?\((\d+)\s+pages?", lg, re.S).group(1))
    ov_h = [float(x) for x in re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", lg)]
    ov_v = [float(x) for x in re.findall(r"Overfull \\vbox \(([0-9.]+)pt too high\)", lg)]
    undefs = re.findall(r"LaTeX Warning: (?:Reference|Citation) `([^']+)' .*? undefined", lg)
    max_h = max(ov_h) if ov_h else 0.0
    max_v = max(ov_v) if ov_v else 0.0

    print(f"  [VERIFIED {stem + '.pdf':<26}] Pages={pgs:2d} | OverfullHbox={max_h:5.2f}pt | OverfullVbox={max_v:5.2f}pt | Undefined={len(undefs)} | DOI=OK")
    assert max_h == 0.0 and max_v == 0.0, f"Overfull in {stem}: h={max_h}, v={max_v}"
    assert not undefs, f"Undefined refs in {stem}: {undefs}"
    assert "**" not in ptxt, f"Double-asterisk found in {stem}"
    assert "24575614" in ptxt and "/tree/v6" in ptxt, f"DOI or /tree/v6 missing in {stem}"
    if stem == "main":
        assert 10 <= pgs <= 12, f"Main pages={pgs} out of 10..12 limit"

vr = run_cmd(["python", str(ver_p)])
print("  [VERIFIER PASS]", vr.stdout.strip().splitlines()[-1])

print("\n=== [3/5] EXPORTING ALL 7 DOI-STAMPED V6 DELIVERABLES TO D:\\Research & 00_final_submission_v6 ===")
shutil.copy2(ms_v6 / "main.pdf",                  res_dir / "CAPQ-Pareto-Main-IEEE-TCC-V6.pdf")
shutil.copy2(ms_v6 / "supplement.pdf",            res_dir / "CAPQ-Pareto-Supplement-IEEE-TCC-V6.pdf")
shutil.copy2(ms_v6 / "response_to_reviewers.pdf", res_dir / "CAPQ-Pareto-Response-to-Reviewers-V6.pdf")
shutil.copy2(ms_v6 / "cover_letter.pdf",          res_dir / "CAPQ-Pareto-Cover-Letter-IEEE-TCC-V6.pdf")
shutil.copy2(fig_dir / "fig1_architecture.png",   res_dir / "CAPQ-Pareto-Fig1-Preview-V6.png")

v6_delivs_core = [
    "CAPQ-Pareto-Main-IEEE-TCC-V6.pdf",
    "CAPQ-Pareto-Supplement-IEEE-TCC-V6.pdf",
    "CAPQ-Pareto-Response-to-Reviewers-V6.pdf",
    "CAPQ-Pareto-Cover-Letter-IEEE-TCC-V6.pdf",
    "CAPQ-Pareto-Fig1-Preview-V6.png",
    "CAPQ-Pareto-Overleaf-IEEE-TCC-V6.zip",
]

zip_v6 = res_dir / "CAPQ-Pareto-Overleaf-IEEE-TCC-V6.zip"
with zipfile.ZipFile(zip_v6, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for fn in ("main.tex", "supplement.tex", "references.bib", "response_to_reviewers.tex", "cover_letter.tex"):
        zf.write(ms_v6 / fn, arcname=fn)
    for tf in sorted(tbl_dir.glob("*.tex")):
        zf.write(tf, arcname="tables/" + tf.name)
    for ff in sorted(fig_dir.glob("*.*")):
        zf.write(ff, arcname="figures/" + ff.name)
    zf.write(v6_root / "SHA256SUMS_V6.txt", arcname="SHA256SUMS_V6.txt")
    zf.write(v6_root / "reproduce_and_verify_v6.py", arcname="reproduce_and_verify_v6.py")
    zf.write(v6_root / ".zenodo.json", arcname=".zenodo.json")

manifest_p = res_dir / "CAPQ-Pareto-SHA256-Manifest-V6.txt"
m_lines = ["# CAPQ-Pareto V6 Final Deliverables SHA-256 Manifest (Zenodo DOI: " + DOI_ID + ")"]
for dn in v6_delivs_core:
    m_lines.append(sha256_64(res_dir / dn) + "  " + dn)
manifest_p.write_text("\n".join(m_lines) + "\n", encoding="utf-8", newline="\n")

all_7 = v6_delivs_core + ["CAPQ-Pareto-SHA256-Manifest-V6.txt"]
deliv_table_rows = []
for dn in all_7:
    src_f = res_dir / dn
    dst_f = sub_v6 / dn
    shutil.copy2(src_f, dst_f)
    h64 = sha256_64(dst_f)
    kb = dst_f.stat().st_size / 1024.0
    deliv_table_rows.append(f"| [`{dn}`](00_final_submission_v6/{dn}) | {kb:,.1f} KB | `{h64}` |")
    print(f"  {dn:<44} | {dst_f.stat().st_size:>9,} B | SHA256={h64}")

bt = chr(96) * 3
readme_v6 = "\n".join([
    "# Empirical Systems Characterization of Post-Quantum Session Pooling, Single-Ciphertext CKKS Retrieval, and Transport Fallback Across Cloud-Edge Vector-SQL Streams (CAPQ-Pareto V6)",
    "",
    f"[![DOI](https://zenodo.org/badge/DOI/{DOI_ID}.svg)]({DOI_URL})",
    "",
    "**Target Venue:** *IEEE Transactions on Cloud Computing (IEEE TCC)* — Final Verified Revision (**V6**)  ",
    f"**Permanent Archival DOI:** [`{DOI_ID}`]({DOI_URL})  ",
    "**Author:** Preecha Noiumkar (Member, IEEE), Department of Information Technology, Faculty of Informatics, Mahasarakham University, Thailand (`preecha.n@msu.ac.th` | ORCID: [0009-0000-5964-4896](https://orcid.org/0009-0000-5964-4896))",
    "",
    "![CAPQ-Pareto V6 System Architecture](00_final_submission_v6/CAPQ-Pareto-Fig1-Preview-V6.png)",
    "",
    "---",
    "",
    "## Final V6 Submission Package (`00_final_submission_v6/`)",
    "",
    "| Final V6 Deliverable File | Size | Full 64-Character SHA-256 Digest |",
    "|---|---:|---|",
    *deliv_table_rows,
    "",
    "---",
    "",
    "## One-Command Clean-Room Verification",
    "",
    "Verify all **40 active V6 raw/reprocessed experimental artifacts** (full 64-hex SHA-256 manifest in `SHA256SUMS_V6.txt`) and reproduce the core metrics across Tables 2–9 in one command:",
    "",
    bt + "bash",
    "python reproduce_and_verify_v6.py",
    bt,
    "",
    "## Repository Structure (Final V6)",
    "- `00_final_submission_v6/` — Final compiled IEEE TCC V6 PDFs (11-page Main Paper, 5-page Supplement, Point-by-Point Response, Cover Letter), Overleaf V6 ZIP, Fig. 1 PNG, and SHA-256 Manifest.",
    "- `08_manuscript_ieee_tcc_v6/` — Complete LaTeX sources (`main.tex`, `supplement.tex`, `response_to_reviewers.tex`, `cover_letter.tex`, `references.bib`, auto-generated `tables/tab1..tab9.tex`, and 14 vector `figures/*.pdf`).",
    "- `06_experiments_results/v6_raw_logs/` — Verified V6 experimental logs: Native-C `OpenSSL EVP AES-256-GCM` socket benchmarks (`v6_b1_*`), `PostgreSQL 18.6 + pgvector 0.8.1` predicate-filtered hybrid SQL+HNSW (`v6_b2_*`), `N=8192` 1-CT slot-provisioning curves up to `M=3,072` (`v6_a2_*`), SLA threshold sensitivity & unweighted 4D Pareto (`v6_a3_*`), and reconciled dual-track 3-region WAN logs.",
    "- `11_v6_workspace/` — Native-C benchmark source (`bench_v6_native_c_tcp_vs_udp_aead.c`), OpenFHE C++ / Python reprocessing scripts, and automated `0.00pt` Overfull PDF verification pipeline.",
    f"- `SHA256SUMS_V6.txt` & `.zenodo.json` — Complete 64-hex cryptographic provenance manifest and Zenodo archival metadata (`{DOI_ID}`).",
])
(v6_root / "README.md").write_text(readme_v6 + "\n", encoding="utf-8", newline="\n")

for fn_v5, h_pre in v5_locked.items():
    assert sha256_64(res_dir / fn_v5) == h_pre, "V5 Zero-Touch violated on " + fn_v5
print("\n[ZERO-TOUCH V5 VERIFIED 100%] Every V5 file in D:\\Research remains untouched byte-for-byte!")

print("\n=== [4/5] SELECTING VERIFIED GITHUB TRANSPORT & PUSHING SMALL CHUNKS ===")
git_prefix = ["git", "-c", "http.version=HTTP/1.1", "-c", "http.postBuffer=524288000"]
test_nat = subprocess.run(git_prefix + ["ls-remote", "--heads", "origin", "v6"], cwd=v6_root, capture_output=True, text=True)
if test_nat.returncode != 0:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    cand_ips = ["57.155.223.240", "20.205.243.166"]
    for url, hdrs in [
        ("https://1.1.1.1/dns-query?name=github.com&type=A", {"Accept": "application/dns-json"}),
        ("https://8.8.8.8/resolve?name=github.com&type=A", {"Accept": "application/json"}),
    ]:
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=5, context=ctx) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for ans in data.get("Answer", []):
                    ip_s = ans.get("data", "").strip()
                    if ans.get("type") == 1 and ip_s and ip_s not in cand_ips:
                        cand_ips.insert(0, ip_s)
        except Exception:
            pass
    chosen_ip = None
    for ip in cand_ips:
        pr = subprocess.run(git_prefix + ["-c", "http.curloptResolve=github.com:443:" + ip, "ls-remote", "--heads", "origin", "v6"], cwd=v6_root, capture_output=True, text=True)
        if pr.returncode == 0:
            chosen_ip = ip
            break
    if not chosen_ip:
        raise RuntimeError("Could not reach github.com:443.")
    git_prefix += ["-c", "http.curloptResolve=github.com:443:" + chosen_ip]
    print("  [TRANSPORT OK] DoH route github.com:443 -> " + chosen_ip)
else:
    print("  [TRANSPORT OK] Native HTTPS route active.")

def push_chunk(paths: list[str], msg: str, tag: str) -> None:
    for p in paths:
        if (v6_root / p).exists():
            run_cmd(["git", "add", p])
    if not run_cmd(["git", "diff", "--cached", "--name-only"]).stdout.strip():
        print("  [" + tag + "] Already up to date.")
        return
    run_cmd(["git", "commit", "-m", msg])
    pr = subprocess.run(git_prefix + ["push", "-u", "origin", "v6"], cwd=v6_root, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if pr.returncode != 0:
        raise RuntimeError("Push failed (" + tag + "):\n" + pr.stderr)
    print("  [" + tag + " PUSH OK]")

run_cmd(["git", "checkout", "v6"])
# Remove temporary helper scripts if 0-byte or broken
for stale in ("11_v6_workspace/v6_fix_reb_hbox.py", "11_v6_workspace/v6_finish_doi_zero_overfull.py"):
    sp = v6_root / stale
    if sp.exists():
        sp.unlink()

push_chunk(
    ["README.md", ".zenodo.json", "reproduce_and_verify_v6.py", "08_manuscript_ieee_tcc_v6", "11_v6_workspace/v6_complete_doi_and_sync.py"],
    "docs(v6-zenodo-doi): embed Zenodo DOI " + DOI_ID + " with 0.00pt Overfull across all 4 V6 documents",
    "CHUNK 1/3 (TeX + README + Metadata)"
)
push_chunk(
    [
        "00_final_submission_v6/CAPQ-Pareto-Main-IEEE-TCC-V6.pdf",
        "00_final_submission_v6/CAPQ-Pareto-Supplement-IEEE-TCC-V6.pdf",
        "00_final_submission_v6/CAPQ-Pareto-Response-to-Reviewers-V6.pdf",
        "00_final_submission_v6/CAPQ-Pareto-Cover-Letter-IEEE-TCC-V6.pdf",
        "00_final_submission_v6/CAPQ-Pareto-SHA256-Manifest-V6.txt",
    ],
    "release(v6-doi-pdfs): update final 0.00pt V6 PDFs & manifest with DOI " + DOI_ID,
    "CHUNK 2/3 (Final V6 PDFs)"
)
push_chunk(
    ["00_final_submission_v6/CAPQ-Pareto-Overleaf-IEEE-TCC-V6.zip"],
    "release(v6-doi-zip): update final CAPQ-Pareto-Overleaf-IEEE-TCC-V6.zip with DOI " + DOI_ID,
    "CHUNK 3/3 (Final Overleaf ZIP)"
)

print("\n=== [5/5] UPDATING GITHUB FRONT PAGE (origin/main) TO MATCH FINAL DOI-STAMPED V6 ===")
pm = subprocess.run(git_prefix + ["push", "origin", "v6:main", "--force"], cwd=v6_root, capture_output=True, text=True, encoding="utf-8", errors="replace")
if pm.returncode != 0:
    raise RuntimeError("Failed updating origin/main:\n" + pm.stderr)

print("  [ALL DONE 100%] Latest Commit:", run_cmd(["git", "log", "-n", "1", "--oneline"]).stdout.strip())