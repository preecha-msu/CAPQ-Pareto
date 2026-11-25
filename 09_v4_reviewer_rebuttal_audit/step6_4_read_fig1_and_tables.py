from __future__ import annotations
import csv
from pathlib import Path

repo = Path(r"D:\Research\CAPQ-Pareto-V3")

print("=== [A] ALL 35 ROWS OF v4_phase2_bandit_10seeds_summary.csv ===")
with (repo / "09_v4_reviewer_rebuttal_audit" / "v4_phase2_bandit_10seeds_summary.csv").open("r", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        print(
            f"{r['scenario']:<28} | {r['controller']:<28} | "
            f"reg={r['mean_regret']:>7} [{r['regret_ci95_lo']},{r['regret_ci95_hi']}] | "
            f"lat={r['mean_lat_ms']:>6} | acc={r['mean_acc']} | sla={r['sla_pass_pct']:>6}% | "
            f"p={r['paired_wilcoxon_p_vs_capq']}"
        )

print("\n=== [B] HEADERS & SAMPLE ROWS OF KEY V2/V3 CSVs ===")
for rel in [
    "06_experiments_results/v2_raw_logs/v2_million_ef_sweep.csv",
    "06_experiments_results/v2_raw_logs/v2_mixed_rw_concurrency.csv",
    "06_experiments_results/v2_raw_logs/v2_ckks_unified_scaling.csv",
    "06_experiments_results/v2_raw_logs/v2_boundary_replicated_ckks_solution.csv",
    "06_experiments_results/v2_raw_logs/v2_mldsa_auth_0rtt_audit.csv",
    "06_experiments_results/v2_raw_logs/v2_arch_aead_payload_sweep_native_c.csv",
    "06_experiments_results/v2_raw_logs/v2_controlled_10mode_ablation.csv",
    "06_experiments_results/v3_raw_logs/v3_adversarial_leakage_attacks.csv",
    "06_experiments_results/v3_raw_logs/v3_operational_failure_abuse_audit.csv",
    "06_experiments_results/v3_raw_logs/v3_cloud_budget_economics_audit.csv",
    "06_experiments_results/v3_raw_logs/v3_cleanroom_reexecution_matrix.csv",
    "06_experiments_results/v3_raw_logs/v3_wan_mbb_paired_holm_stats.csv",
]:
    p = repo / rel
    if p.is_file():
        with p.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        print(f"[{p.name}] rows={len(rows)} cols={list(rows[0].keys())}")
        print("  sample:", dict(rows[0]))

print("\n=== [C] EXACT SOURCE OF scripts/upgrade_fig1_and_build.py ===")
print((repo / "scripts" / "upgrade_fig1_and_build.py").read_text(encoding="utf-8"))