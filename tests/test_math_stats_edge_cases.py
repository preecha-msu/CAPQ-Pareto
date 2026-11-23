from __future__ import annotations
import math
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

REPO_ROOT = Path(r"D:\Research\CAPQ-Pareto")
sys.path.insert(0, str(REPO_ROOT / "05_telemetry_audit"))
sys.path.insert(0, str(REPO_ROOT / "02_orchestrator" / "pareto_math_model"))

import sha256_manifest as sm
import math_stats_core as msc


class TestCAPQMathAndEdgeCases(unittest.TestCase):
    def test_normalization_zero_span_and_bounds(self):
        # Degenerate x_max == x_min must not divide by zero
        self.assertEqual(msc.normalize_metric(50.0, 50.0, 50.0), 0.0)
        self.assertAlmostEqual(msc.normalize_metric(55.0, 50.0, 60.0), 0.5, places=12)
        with self.assertRaises(ValueError):
            msc.normalize_metric(10.0, 20.0, 10.0)
        with self.assertRaises(ValueError):
            msc.normalize_metric(float("nan"), 0.0, 1.0)

    def test_rapl_wraparound_and_non_negative_clamp(self):
        max_uj = 262143999  # Typical Intel RAPL wrap ceiling (~262.14 J)
        # Wraparound case: t0 near max, t1 wrapped past 0 -> delta = 1000 + 4000 = 5000 uJ = 5.0 mJ
        net_mj = msc.compute_rapl_net_energy_mj(
            e_raw_t0=max_uj - 1000,
            e_raw_t1=4000,
            max_energy_counter=max_uj,
            p_idle_mw=1000.0,
            delta_t_sec=0.002,  # idle = 2.0 mJ -> net = 3.0 mJ
        )
        self.assertAlmostEqual(net_mj, 3.0, places=9)

        # Sub-ms idle jitter where idle > raw delta must clamp strictly at 0.0 mJ
        clamped = msc.compute_rapl_net_energy_mj(1000, 1200, max_uj, p_idle_mw=5000.0, delta_t_sec=0.001)
        self.assertEqual(clamped, 0.0)
        # Windows Kernel EMI pWh verification (exact Intel Core Ultra 9 185H reading)
        emi_mj = msc.compute_rapl_net_energy_mj(
            e_raw_t0=20978215098888,
            e_raw_t1=20980773388055,
            max_energy_counter=msc.UINT64_MAX_COUNTER,
            p_idle_mw=0.0,
            delta_t_sec=2.026,
            kappa_mj_per_unit=msc.KAPPA_WINDOWS_PWH_TO_MJ,
        )
        self.assertAlmostEqual(emi_mj, 2558289167 * 3.6e-6, places=6)

    def test_gcp_unit_conversion_exactness(self):
        # 1000 ms (1 sec), 1 GiB outbound (2^30 bytes), 4 vCPU, 16 GB RAM
        cost = msc.compute_gcp_query_cost_usd(
            t_cloud_ms=1000.0,
            b_out_bytes=(1 << 30),
            n_vcpu=4.0,
            mem_gb=16.0,
            p_vcpu_per_sec=0.00000745,
            p_mem_gb_per_sec=0.00000100,
            p_egress_per_gib=0.12,
        )
        expected = (4.0 * 0.00000745 + 16.0 * 0.00000100) * 1.0 + 0.12 * 1.0
        self.assertAlmostEqual(cost, expected, places=12)

    def test_linucb_sla_fallback_and_spd_update(self):
        sched = msc.RegularizedLinUCBScheduler(["reg_a", "reg_b", "reg_c"], d_c=10, lambda_reg=1.0, nu=0.2)
        x = np.array([1.0, 0.0, 0.5, 0.0, 0.2, 0.8, 0.1, 0.01, 0.34, 0.33], dtype=np.float64)
        # Empty M_sla must deterministically fallback to min p95 latency in M_sec
        chosen = sched.select_action(
            x_t=x,
            sec_valid_actions=["reg_a", "reg_c"],
            sla_feasible_actions=[],
            est_p95_latency={"reg_a": 42.0, "reg_b": 19.0, "reg_c": 135.0},
        )
        self.assertEqual(chosen, "reg_a")
        sched.update("reg_a", x, loss_j=0.31)
        eigvals = np.linalg.eigvalsh(sched.A["reg_a"])
        self.assertTrue(np.all(eigvals >= 1.0 - 1e-12))

    def test_wilcoxon_mcnemar_holm_edge_cases(self):
        # Identical paired trials -> stat=0.0, p=1.0 (no crash)
        arr = np.full(30, 12.34)
        stat, p_w = msc.paired_wilcoxon_safe(arr, arr)
        self.assertEqual((stat, p_w), (0.0, 1.0))

        # Zero discordant SLA pairs -> p=1.0 (no division by zero)
        sla = np.ones(30, dtype=bool)
        n01, n10, p_mc0 = msc.mcnemar_sla_safe(sla, sla)
        self.assertEqual((n01, n10, p_mc0), (0, 0, 1.0))

        # Small discordant (<25) -> exact binomial; Large (>=25) -> continuity-corrected chi2
        a_s = np.array([True] * 10 + [True] * 20)
        b_s = np.array([False] * 10 + [True] * 20)
        _, _, p_exact = msc.mcnemar_sla_safe(a_s, b_s)
        self.assertAlmostEqual(p_exact, 2.0 * (0.5 ** 10), places=9)

        a_l = np.array([True] * 26 + [True] * 4)
        b_l = np.array([False] * 26 + [True] * 4)
        _, _, p_chi2 = msc.mcnemar_sla_safe(a_l, b_l)
        self.assertTrue(0.0 < p_chi2 < 1e-5)

        # Holm-Bonferroni monotonicity & original index preservation
        raw_p = [0.04, 0.01, 0.60]
        adj_p = msc.holm_bonferroni(raw_p)
        # Sorted: 0.01*3=0.03, max(0.03, 0.04*2)=0.08, max(0.08, min(1, 0.60*1))=0.60 -> restored [0.08, 0.03, 0.60]
        self.assertAlmostEqual(adj_p[0], 0.08, places=12)
        self.assertAlmostEqual(adj_p[1], 0.03, places=12)
        self.assertAlmostEqual(adj_p[2], 0.60, places=12)

    def test_sha256_zero_estimation_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            csv_f = tdir / "raw.csv"
            tex_f = tdir / "tab.tex"
            man_f = tdir / "manifest.json"
            csv_f.write_text("trial,lat_ms\n1,12.4\n", encoding="utf-8")
            tex_f.write_text("\\begin{tabular}{c}12.4\\end{tabular}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                sm.register_verified_table(man_f, csv_f, tex_f, measured_rows=0, experiment_id="exp0")
            rec = sm.register_verified_table(man_f, csv_f, tex_f, measured_rows=1, experiment_id="exp1")
            self.assertEqual(len(rec["raw_csv_sha256"]), 64)
            self.assertEqual(len(rec["latex_table_sha256"]), 64)


if __name__ == "__main__":
    print(f"[SHA-256] sha256_manifest.py : {sm.sha256_digest(REPO_ROOT / '05_telemetry_audit' / 'sha256_manifest.py')}")
    print(f"[SHA-256] math_stats_core.py : {sm.sha256_digest(REPO_ROOT / '02_orchestrator' / 'pareto_math_model' / 'math_stats_core.py')}")
    unittest.main(verbosity=2)