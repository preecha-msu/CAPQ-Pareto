import hashlib
import json
import os
import pathlib
import re
import subprocess

res_dir = pathlib.Path(r"D:\Research")
v5_root = pathlib.Path(r"D:\Research\CAPQ-Pareto-V5")
v6_root = pathlib.Path(r"D:\Research\CAPQ-Pareto-V6")
ms_v6   = v6_root / "08_manuscript_ieee_tcc_v6"
work_v6 = v6_root / "11_v6_workspace"

def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

print("=== [1/5] ZERO-TOUCH V5 DELIVERABLES INTEGRITY CHECK ===")
expected_v5 = {
    "CAPQ-Pareto-Main-IEEE-TCC-V5.pdf":       "7fc5dd4792e925e4a0df7d687c857b31df8610eda5af218776d14ed1a100965d",
    "CAPQ-Pareto-Supplement-IEEE-TCC-V5.pdf": "b309df60b4f22ef401cbe68d0b1cf5b4aae7f4830ef60c042ee35bfca20db8ce",
    "CAPQ-Pareto-Overleaf-IEEE-TCC-V5.zip":   "5ae8be25ab8598366c878384a79ad899f7d9c1f34ee6c7bff474f69f6e9fcfc5",
}
v5_hashes = {}
for p in sorted(res_dir.glob("*V5*")):
    if p.is_file():
        digest = sha256_file(p)
        v5_hashes[p.name] = digest
        exp = expected_v5.get(p.name)
        tag = "EXACT MATCH" if (exp and digest == exp) else ("LOCKED" if not exp else "MISMATCH!")
        print(f"  [{tag:<11}] {p.name:<44} | {p.stat().st_size:>9,} B | {digest[:16]}...")

print("\n=== [2/5] GIT BRANCH 'v6' & REPOSITORY STATUS ===")
git_dir = v6_root / ".git"
git_info = {"has_git": git_dir.exists()}
if git_dir.exists():
    def run_git(args):
        r = subprocess.run(["git", "-C", str(v6_root)] + args, capture_output=True, text=True, encoding="utf-8", errors="replace")
        return r.stdout.strip()
    branch = run_git(["branch", "--show-current"])
    remotes = run_git(["remote", "-v"])
    commits = run_git(["log", "-n", "5", "--oneline"])
    status_lines = [ln for ln in run_git(["status", "-s"]).splitlines() if ln.strip()]
    branches_all = run_git(["branch", "-a"])
    git_info.update({
        "active_branch": branch,
        "dirty_file_count": len(status_lines),
        "dirty_sample": status_lines[:15],
        "recent_commits": commits.splitlines(),
    })
    print(f"  Active Branch      : {branch}")
    print(f"  All Branches       : {', '.join(b.strip() for b in branches_all.splitlines())}")
    print(f"  Modified/Untracked : {len(status_lines)} files")
    for ln in status_lines[:12]:
        print(f"    {ln}")
    if len(status_lines) > 12:
        print(f"    ... (+{len(status_lines) - 12} more files)")
    print("  Recent Commits     :")
    for c in commits.splitlines():
        print(f"    {c}")
else:
    print("  [WARNING] .git folder not found inside D:\\Research\\CAPQ-Pareto-V6!")

print("\n=== [3/5] V6 WORKSPACE, DATASETS & EXPERIMENT LOGS INVENTORY ===")
top_dirs = [d.name for d in sorted(v6_root.iterdir()) if d.is_dir() and d.name != ".git"]
print(f"  Top-Level Folders ({len(top_dirs)}): {', '.join(top_dirs)}")

npy_files = list((v6_root / "01_workloads").rglob("*.np[yz]")) if (v6_root / "01_workloads").exists() else []
print(f"  Datasets (.npy/.npz) in 01_workloads : {len(npy_files)} files")

exp_root = v6_root / "06_experiments_results"
csv_summary = {}
if exp_root.exists():
    for sub in sorted(exp_root.iterdir()):
        if sub.is_dir():
            csvs = list(sub.glob("*.csv"))
            jsons = list(sub.glob("*.json"))
            if csvs or jsons:
                csv_summary[sub.name] = {"csv": len(csvs), "json": len(jsons)}
                print(f"  Logs in 06_experiments_results/{sub.name:<22}: {len(csvs):>3} CSVs | {len(jsons):>2} JSONs")

v6_py = sorted(p.name for p in work_v6.glob("*.py"))
print(f"  Scripts in 11_v6_workspace ({len(v6_py)})       : {', '.join(v6_py[:8])}{' ...' if len(v6_py) > 8 else ''}")

print("\n=== [4/5] V6 MANUSCRIPT (08_manuscript_ieee_tcc_v6) & D:\\Research\\*V6* SCAN ===")
ms_info = {"exists": ms_v6.exists()}
if ms_v6.exists():
    tex_files = sorted(p.name for p in ms_v6.glob("*.tex"))
    fig_dir = ms_v6 / "figures"
    figs = sorted(p.name for p in fig_dir.glob("*.*")) if fig_dir.exists() else []
    tbl_dir = ms_v6 / "tables"
    tbls = sorted(p.name for p in tbl_dir.glob("*.tex")) if tbl_dir.exists() else []
    v5_token_hits = {}
    for tf in ms_v6.glob("*.tex"):
        txt = tf.read_text(encoding="utf-8", errors="ignore")
        hits = len(re.findall(r"\bV5\b|_v5\b|-V5\b", txt))
        if hits:
            v5_token_hits[tf.name] = hits
    ms_info.update({
        "tex_files": tex_files,
        "figure_count": len(figs),
        "figures": figs,
        "table_files_count": len(tbls),
        "residual_v5_tokens": v5_token_hits,
    })
    print(f"  TeX Root Files     : {', '.join(tex_files)}")
    print(f"  Figures in figures/: {len(figs)} files -> {', '.join(figs[:6])}{' ...' if len(figs) > 6 else ''}")
    print(f"  External tables/   : {len(tbls)} .tex files")
    print(f"  Residual V5 tokens : {v5_token_hits if v5_token_hits else '0 hits (Clean)'}")
else:
    print("  [WARNING] 08_manuscript_ieee_tcc_v6 does not exist yet!")

v6_deliv = {}
for p in sorted(res_dir.glob("*V6*")):
    if p.is_file():
        digest = sha256_file(p)
        v6_deliv[p.name] = {"bytes": p.stat().st_size, "sha256": digest}
        print(f"  [D:\\Research V6] {p.name:<44} | {p.stat().st_size:>9,} B | {digest[:16]}...")

inv_out = work_v6 / "v6_step1_baseline_inventory.json"
payload = {
    "v5_locked_hashes": v5_hashes,
    "git_info": git_info,
    "datasets_count": len(npy_files),
    "experiment_logs": csv_summary,
    "v6_workspace_scripts": v6_py,
    "manuscript_v6": ms_info,
    "v6_deliverables": v6_deliv,
}
inv_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
print(f"\n[INVENTORY SAVED] {inv_out} | SHA256={sha256_file(inv_out)}")