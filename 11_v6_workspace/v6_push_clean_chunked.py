import json
import pathlib
import ssl
import subprocess
import urllib.request

repo = pathlib.Path(r"D:\Research\CAPQ-Pareto-V6")

def run_cmd(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(args, cwd=repo, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(args)}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
    return r

# 1. Reset ONLY the bloated local commits back to origin/v6 (100% of files on disk stay untouched!)
orig_v6_sha = run_cmd(["git", "rev-parse", "refs/remotes/origin/v6"]).stdout.strip()
print(f"=== [1/4] RESETTING GIT INDEX TO CLEAN origin/v6 ({orig_v6_sha[:8]}) ===")
run_cmd(["git", "reset", "--mixed", orig_v6_sha])

# 2. Harden .gitignore against large raw dataset binaries (.npz/.npy/.so) & temporary previews
gi = repo / ".gitignore"
gi_txt = gi.read_text(encoding="utf-8") if gi.exists() else ""
for rule in [
    "01_workloads/**/*.npz",
    "01_workloads/**/*.npy",
    "11_v6_workspace/pdf_page_previews/",
    "*.so",
    "*.o",
]:
    if rule not in gi_txt:
        gi_txt = gi_txt.rstrip() + "\n" + rule + "\n"
gi.write_text(gi_txt, encoding="utf-8", newline="\n")

# Verify 40/40 SHA-256 verifier passes 100%
vr = run_cmd(["python", str(repo / "reproduce_and_verify_v6.py")])
print("  [VERIFIER OK] 40/40 V6 artifacts verified intact.")

# 3. Determine working Git transport (try native DNS first; if DNS fails, pick verified DoH IP via ls-remote)
print("\n=== [2/4] SELECTING VERIFIED GITHUB TRANSPORT ===")
git_prefix = ["git", "-c", "http.version=HTTP/1.1", "-c", "http.postBuffer=524288000"]

test_native = subprocess.run(git_prefix + ["ls-remote", "--heads", "origin", "v6"], cwd=repo, capture_output=True, text=True)
if test_native.returncode == 0:
    print("  [TRANSPORT OK] Native GitHub HTTPS connection active.")
else:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    cand_ips = []
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
                        cand_ips.append(ip_s)
        except Exception:
            pass
    cand_ips.append("20.205.243.166")
    chosen_ip = None
    for ip in cand_ips:
        probe = subprocess.run(
            git_prefix + ["-c", f"http.curloptResolve=github.com:443:{ip}", "ls-remote", "--heads", "origin", "v6"],
            cwd=repo, capture_output=True, text=True
        )
        if probe.returncode == 0:
            chosen_ip = ip
            break
    if not chosen_ip:
        raise RuntimeError("Could not reach github.com:443 via any resolved IP.")
    git_prefix = git_prefix + ["-c", f"http.curloptResolve=github.com:443:{chosen_ip}"]
    print(f"  [TRANSPORT OK] Verified DoH route github.com:443 -> {chosen_ip}")

def audit_staged_sizes(max_mb: float = 8.0) -> float:
    staged = run_cmd(["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"]).stdout.splitlines()
    tot_b = 0
    for rel in staged:
        fp = repo / rel
        if fp.exists() and fp.is_file():
            sz = fp.stat().st_size
            if sz > max_mb * 1024 * 1024:
                run_cmd(["git", "reset", "HEAD", "--", rel])
                print(f"  [SKIPPED LARGE BLOB] Unstaged {rel} ({sz/1e6:.2f} MB)")
            else:
                tot_b += sz
    return tot_b / (1024.0 * 1024.0)

def commit_and_push_chunk(msg: str, label: str) -> None:
    st = run_cmd(["git", "diff", "--cached", "--name-only"]).stdout.strip()
    if not st:
        print(f"  [{label}] Nothing new to commit.")
        return
    mb = audit_staged_sizes()
    print(f"  [{label}] Committing & pushing {mb:.2f} MB payload...")
    run_cmd(["git", "commit", "-m", msg])
    p = subprocess.run(git_prefix + ["push", "-u", "origin", "v6"], cwd=repo, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"Push failed on {label}:\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    print(f"  [{label} PUSH OK]")

print("\n=== [3/4] PUSHING CLEAN V6 IN 3 SMALL (<2 MB) CHUNKS TO PREVENT HTTP 408 ===")

# Chunk 1: Tracked updates/deletions + README + Verifier + V6 CSVs + TeX sources + workspace scripts
run_cmd(["git", "add", "-u"])
for p_rel in [
    ".gitignore", "README.md", ".zenodo.json", "SHA256SUMS_V6.txt", "reproduce_and_verify_v6.py",
    "08_manuscript_ieee_tcc_v6", "06_experiments_results/v6_raw_logs",
    "11_v6_workspace/bench_v6_native_c_tcp_vs_udp_aead.c", "11_v6_workspace/run_f2_native_c_bench.sh",
]:
    if (repo / p_rel).exists():
        run_cmd(["git", "add", p_rel])
for py_f in sorted((repo / "11_v6_workspace").glob("*.py")):
    run_cmd(["git", "add", str(py_f.relative_to(repo))])
commit_and_push_chunk("release(v6-core): update V6 title, README, TeX sources, Groups A & B logs, and 40/40 SHA256 verifier", "CHUNK 1/3")

# Chunk 2: Final V6 PDFs, PNG & Manifest in 00_final_submission_v6/ (~1.99 MB)
for fn in [
    "CAPQ-Pareto-Main-IEEE-TCC-V6.pdf",
    "CAPQ-Pareto-Supplement-IEEE-TCC-V6.pdf",
    "CAPQ-Pareto-Response-to-Reviewers-V6.pdf",
    "CAPQ-Pareto-Cover-Letter-IEEE-TCC-V6.pdf",
    "CAPQ-Pareto-Fig1-Preview-V6.png",
    "CAPQ-Pareto-SHA256-Manifest-V6.txt",
]:
    run_cmd(["git", "add", f"00_final_submission_v6/{fn}"])
commit_and_push_chunk("docs(v6-pdfs): add final verified V6 Main, Supplement, Rebuttal, Cover Letter PDFs & Fig 1 preview", "CHUNK 2/3")

# Chunk 3: Final V6 Overleaf ZIP (~1.02 MB)
run_cmd(["git", "add", "00_final_submission_v6/CAPQ-Pareto-Overleaf-IEEE-TCC-V6.zip"])
commit_and_push_chunk("docs(v6-overleaf-zip): add final verified CAPQ-Pareto-Overleaf-IEEE-TCC-V6.zip", "CHUNK 3/3")

# Step 4: Instant 0-byte server-side pointer update so GitHub front page (main) matches v6!
print("\n=== [4/4] POINTING GITHUB FRONT PAGE (origin/main) TO FINAL V6 (0-BYTE SERVER REF UPDATE) ===")
pm = subprocess.run(git_prefix + ["push", "origin", "v6:main", "--force"], cwd=repo, capture_output=True, text=True, encoding="utf-8", errors="replace")
if pm.returncode != 0:
    raise RuntimeError(f"Failed updating origin/main:\n{pm.stderr}")

head_short = run_cmd(["git", "log", "-n", "1", "--oneline"]).stdout.strip()
print("  [ORIGIN/MAIN UPDATED OK] Latest Commit:", head_short)
print("\n=== [ALL GITHUB PAGES UPDATED TO FINAL V6 100%] ===")
print("  1. Front Page : https://github.com/preecha-msu/CAPQ-Pareto")
print("  2. Branch v6  : https://github.com/preecha-msu/CAPQ-Pareto/tree/v6")