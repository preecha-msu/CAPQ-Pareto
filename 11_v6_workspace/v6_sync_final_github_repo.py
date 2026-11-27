import hashlib
import pathlib
import shutil
import subprocess

v6_root = pathlib.Path(r"D:\Research\CAPQ-Pareto-V6")
res_dir = pathlib.Path(r"D:\Research")

def sha256_64(p: pathlib.Path) -> str:
    d = hashlib.sha256(p.read_bytes()).hexdigest()
    assert len(d) == 64
    return d

v5_locked = {p.name: sha256_64(p) for p in sorted(res_dir.glob("*V5*")) if p.is_file()}

protected_rels = set()
for ln in (v6_root / "SHA256SUMS_V6.txt").read_text(encoding="utf-8").splitlines():
    s = ln.strip()
    if s and not s.startswith("#"):
        _, rel_p = s.split(None, 1)
        protected_rels.add(rel_p.strip().replace("\\", "/"))

legacy_candidates = [
    "main.tex",
    "supplement.tex",
    "references.bib",
    "README_OVERLEAF.md",
    "CHANGELOG_INTERNAL_V5.md",
    "SHA256SUMS_V3.txt",
    "SHA256SUMS_V4.txt",
    "SHA256SUMS_V5.txt",
    "00_manuscript",
    "08_manuscript_ieee_tcc_v2",
    "08_manuscript_ieee_tcc_v3",
    "08_manuscript_ieee_tcc_v4",
    "08_manuscript_ieee_tcc_v5",
    "09_submission_docs_v3",
    "10_manuscript_ieee_tcc_v4",
    "10_submission_docs_v4",
    "10_v5_workspace",
]
for rel_old in legacy_candidates:
    if any(p == rel_old or p.startswith(rel_old + "/") for p in protected_rels):
        continue
    target = v6_root / rel_old
    if target.exists():
        subprocess.run(["git", "rm", "-rf", "--ignore-unmatch", rel_old], cwd=v6_root, check=False, capture_output=True)
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.is_file():
            target.unlink()
        print(f"  [CLEANED LEGACY] Removed obsolete path: {rel_old}")

sub_v6 = v6_root / "00_final_submission_v6"
sub_v6.mkdir(parents=True, exist_ok=True)

v6_delivs = [
    "CAPQ-Pareto-Main-IEEE-TCC-V6.pdf",
    "CAPQ-Pareto-Supplement-IEEE-TCC-V6.pdf",
    "CAPQ-Pareto-Response-to-Reviewers-V6.pdf",
    "CAPQ-Pareto-Cover-Letter-IEEE-TCC-V6.pdf",
    "CAPQ-Pareto-Fig1-Preview-V6.png",
    "CAPQ-Pareto-Overleaf-IEEE-TCC-V6.zip",
    "CAPQ-Pareto-SHA256-Manifest-V6.txt",
]

deliv_table_rows = []
for fn in v6_delivs:
    src_f = res_dir / fn
    assert src_f.exists(), f"Missing final V6 deliverable in D:\\Research: {fn}"
    dst_f = sub_v6 / fn
    shutil.copy2(src_f, dst_f)
    h64 = sha256_64(dst_f)
    kb = dst_f.stat().st_size / 1024.0
    deliv_table_rows.append(f"| [`{fn}`](00_final_submission_v6/{fn}) | {kb:,.1f} KB | `{h64}` |")

bt = chr(96) * 3
readme_v6 = "\n".join([
    "# Empirical Systems Characterization of Post-Quantum Session Pooling, Single-Ciphertext CKKS Retrieval, and Transport Fallback Across Cloud-Edge Vector-SQL Streams (CAPQ-Pareto V6)",
    "",
    "**Target Venue:** *IEEE Transactions on Cloud Computing (IEEE TCC)* — Final Verified Revision (**V6**)  ",
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
    "- `SHA256SUMS_V6.txt` & `.zenodo.json` — Complete 64-hex cryptographic provenance manifest and Zenodo archival metadata.",
])
(v6_root / "README.md").write_text(readme_v6 + "\n", encoding="utf-8", newline="\n")

vr = subprocess.run(["python", str(v6_root / "reproduce_and_verify_v6.py")], capture_output=True, text=True, encoding="utf-8")
print(vr.stdout.strip())
if vr.returncode != 0:
    print(vr.stderr)
    raise SystemExit(1)

for fn_v5, h_pre in v5_locked.items():
    assert sha256_64(res_dir / fn_v5) == h_pre, f"V5 file modified: {fn_v5}"
print("[LOCAL V6 REPO READY & ZERO-TOUCH V5 VERIFIED]")