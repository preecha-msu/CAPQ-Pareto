import csv
import hashlib
import json
import math
import pathlib
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

v6_root = pathlib.Path(r"D:\Research\CAPQ-Pareto-V6")
res_dir = pathlib.Path(r"D:\Research")
logs_v6 = v6_root / "06_experiments_results" / "v6_raw_logs"
logs_v2 = v6_root / "06_experiments_results" / "v2_raw_logs"
logs_v3 = v6_root / "06_experiments_results" / "v3_raw_logs"
ms_v6   = v6_root / "08_manuscript_ieee_tcc_v6"
fig_dir = ms_v6 / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

def sha256_file(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save_csv(path: pathlib.Path, rows: list[dict]) -> str:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return sha256_file(path)

# ==============================================================================
# 1. FIX TRACK 1 CONFIDENTIAL CLASS ('zt_ckks') & COPY SUPPORTING REAL CSVs
# ==============================================================================
print("=== [2.3-FIX] FIXING TRACK 1 CONFIDENTIAL CLASS ('zt_ckks') & CONSOLIDATING V6 LOGS ===")
t1_raw = list(csv.DictReader((logs_v2 / "v2_wan_3regions_interleaved_raw.csv").open("r", encoding="utf-8")))
v2_sum_map = {(r["region"], r["mode"]): r for r in csv.DictReader((logs_v2 / "v2_wan_3regions_summary_mbb_ci.csv").open("r", encoding="utf-8"))}
v5_t1_map  = {(r["region"], r["mode"]): r for r in csv.DictReader((logs_v6 / "v5_live_gcp_3regions_pure_m1_m5_verified.csv").open("r", encoding="utf-8"))}

GCP_PRICING = {
    "Bangkok":    {"vcpu_hr": 0.2264, "egress_gib": 0.120},
    "Singapore":  {"vcpu_hr": 0.2216, "egress_gib": 0.120},
    "US-Central": {"vcpu_hr": 0.1896, "egress_gib": 0.085},
}

def unified_cost_per_1m_usd(reg_canon: str, srv_ms: float, resp_wire_bytes: float) -> float:
    pr = GCP_PRICING[reg_canon]
    return (pr["vcpu_hr"] / 3600.0) * (srv_ms / 1000.0) * 1e6 + pr["egress_gib"] * (resp_wire_bytes / (1024.0 ** 3)) * 1e6

def t1_full_rns_resp_bytes(mode_name: str, qt: str) -> float:
    if qt == "sql":
        return 512.0
    if qt in ("json", "vec_1024d"):
        return 1024.0
    # Exact qt == "zt_ckks"
    if "Classical" in mode_name:
        return 1024.0
    if "M1_" in mode_name or "M3_" in mode_name:
        return 931154.0
    return 132106.0

t1_v6_rows = []
for reg in ("Bangkok", "Singapore", "US-Central"):
    for m in ["M1_Fresh_TLS13_PQC", "M2_Pooled_TLS13_Classical", "M3_Pooled_TLS13_Hybrid768", "M4_Rule_Heuristic_Pooled_INT8", "M5_CAPQ_V2_Adaptive"]:
        rr = [x for x in t1_raw if x["region"] == reg and x["mode"] == m]
        v5r = v5_t1_map[(reg, m)]
        v2r = v2_sum_map[(reg, m)]
        lats = np.array([float(x["user_lat_ms"]) for x in rr], dtype=np.float64)
        accs = np.array([float(x["eff_acc"]) for x in rr], dtype=np.float64)

        pub_rr  = [x for x in rr if x["qtype"] != "zt_ckks"]
        conf_rr = [x for x in rr if x["qtype"] == "zt_ckks"]
        assert len(pub_rr) == 72 and len(conf_rr) == 24, f"Unexpected split: {len(pub_rr)} / {len(conf_rr)}"

        pub_sla150 = float(np.mean([1.0 if (float(x["user_lat_ms"]) <= 150.0 and float(x["eff_acc"]) >= 0.92) else 0.0 for x in pub_rr]) * 100.0)
        conf_mean_rec = float(np.mean([float(x["eff_acc"]) for x in conf_rr]))
        conf_lat250_priv = float(np.mean([1.0 if (float(x["user_lat_ms"]) <= 250.0 and float(x["eff_acc"]) > 0.0) else 0.0 for x in conf_rr]) * 100.0)
        conf_strict092_250 = float(np.mean([1.0 if (float(x["user_lat_ms"]) <= 250.0 and float(x["eff_acc"]) >= 0.92) else 0.0 for x in conf_rr]) * 100.0)

        stream_lat_priv_250 = float(np.mean([
            1.0 if (
                (float(x["user_lat_ms"]) <= (250.0 if x["qtype"] == "zt_ckks" else 150.0))
                and ((float(x["eff_acc"]) > 0.0) if x["qtype"] == "zt_ckks" else (float(x["eff_acc"]) >= 0.92))
            ) else 0.0
            for x in rr
        ]) * 100.0)

        stream_strict092_250 = float(np.mean([
            1.0 if (
                (float(x["user_lat_ms"]) <= (250.0 if x["qtype"] == "zt_ckks" else 150.0))
                and (float(x["eff_acc"]) >= 0.92)
            ) else 0.0
            for x in rr
        ]) * 100.0)

        u_costs = [unified_cost_per_1m_usd(reg, float(x["srv_compute_ms"]), t1_full_rns_resp_bytes(m, x["qtype"])) for x in rr]
        mean_u_cost = float(np.mean(u_costs))
        cpu_only_cost = float(np.mean([float(x["cost_1m_usd"]) for x in rr]))

        t1_v6_rows.append({
            "region": reg,
            "zone": v5r["zone"],
            "phys_1rtt_ms": float(v5r["phys_1rtt_ms"]),
            "mode": m,
            "n_queries": len(rr),
            "mean_lat_ms": round(float(np.mean(lats)), 2),
            "mbb_ci95_lo_ms": float(v5r["mbb_ci95_lo_ms"]),
            "mbb_ci95_hi_ms": float(v5r["mbb_ci95_hi_ms"]),
            "p50_lat_ms": round(float(np.percentile(lats, 50)), 2),
            "p95_lat_ms": round(float(v5r["p95_lat_ms"]), 2),
            "mean_stream_acc": round(float(np.mean(accs)), 4),
            "conf_zt_ckks_recall10": round(conf_mean_rec, 4),
            "pub_150ms_acc092_sla_pct": round(pub_sla150, 2),
            "conf_lat250_priv_sla_pct": round(conf_lat250_priv, 2),
            "conf_strict_acc092_250ms_sla_pct": round(conf_strict092_250, 2),
            "stream_lat_priv_250ms_sla_pct": round(stream_lat_priv_250, 2),
            "stream_strict_acc092_250ms_sla_pct": round(stream_strict092_250, 2),
            "client_rapl_net_edge_mj": float(v2r["net_edge_mj"]),
            "legacy_v2_compact_wire_cost_1m_usd": round(cpu_only_cost, 3),
            "unified_full_rns_cloud_cost_1m_usd": round(mean_u_cost, 2),
        })
        print(f"  [FIXED T1 {reg:<10} | {m:<29}] Mean={np.mean(lats):6.2f} | zt_ckks_Rec={conf_mean_rec:.4f} | ConfStrict092={conf_strict092_250:5.1f}% | StreamStrict092={stream_strict092_250:5.1f}% | UnifCost=${mean_u_cost:5.2f}")

sha_t1 = save_csv(logs_v6 / "v6_track1_live_gcp_m1_m5_reconciled.csv", t1_v6_rows)
print(f"  [SAVED] v6_track1_live_gcp_m1_m5_reconciled.csv | SHA256={sha_t1}")

# Copy verified real logs for Leakage A1-A3, 1M HNSW, Mixed RW, ML-DSA, and Cross-ISA into v6_raw_logs for self-contained V6 provenance
for src_p, dst_name in [
    (logs_v3 / "v3_adversarial_leakage_attacks.csv",       "v6_ref_adversarial_leakage_attacks.csv"),
    (logs_v3 / "v3_operational_failure_abuse_audit.csv",   "v6_ref_operational_failure_abuse_audit.csv"),
    (logs_v2 / "v2_million_ef_sweep.csv",                  "v6_ref_million_hnsw_ef_sweep.csv"),
    (logs_v2 / "v2_mixed_rw_concurrency.csv",              "v6_ref_mixed_rw_concurrency.csv"),
    (logs_v2 / "v2_mldsa_auth_0rtt_audit.csv",             "v6_ref_mldsa_auth_chains.csv"),
    (logs_v2 / "v2_arch_aead_payload_sweep_native_c.csv",  "v6_ref_cross_isa_native_c.csv"),
    (logs_v3 / "v3_cross_isa_e2e_continuum.csv",           "v6_ref_cross_isa_e2e.csv"),
]:
    shutil.copy2(src_p, logs_v6 / dst_name)

# ==============================================================================
# 2. STEP 3.1: GENERATE CLEAN PUBLICATION-GRADE V6 VECTOR FIGURES (ZERO OVERLAP)
# ==============================================================================
print("\n=== [3.1] GENERATING ALL V6 VECTOR FIGURES DIRECTLY FROM VERIFIED V6 CSVs ===")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8.5,
    "axes.titlesize": 9.2,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# --- FIGURE 1: Clean 4-Layer System Architecture (Zero Overlap, No Equal-Probe Claim, Honest Transport Stack) ---
fig1, ax1 = plt.subplots(figsize=(13.2, 5.9))
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 56)
ax1.axis("off")

def draw_box(ax, x, y, w, h, title, lines, fc="#F8FAFC", ec="#334155", tc="#0F172A", title_fs=8.6, body_fs=7.5, lw=1.25):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=0.8", facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(box)
    ax.text(x + w / 2.0, y + h - 1.85, title, ha="center", va="top", fontsize=title_fs, fontweight="bold", color=tc)
    if lines:
        body_txt = "\n".join(lines)
        ax.text(x + 1.1, y + h - 4.2, body_txt, ha="left", va="top", fontsize=body_fs, color="#1E293B", linespacing=1.28)

def draw_arrow(ax, x1, y1, x2, y2, color="#334155", lw=1.3):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>,head_length=4.5,head_width=2.6", color=color, linewidth=lw)
    ax.add_patch(arr)

ax1.text(50, 54.6, "CAPQ-Pareto System Architecture: Session-Pooled Hybrid PQC Transport, Single-MTU INT8, and 1-CT RNS-CKKS Search",
         ha="center", va="center", fontsize=10.2, fontweight="bold", color="#0F172A")

# Swimlane 1 (Left): Workload & Lagged Context Plane
draw_box(ax1, 1.0, 31.0, 21.5, 21.2, "1. Edge Query & Context Plane", [
    "Interleaved Stream Classes:",
    " - Relational SQL / JSON point ops",
    " - Public semantic ANN (128d/1024d)",
    " - Confidential vector search (CKKS)",
    "Lagged 5D Context x_{t-} (no leak):",
    " [1, I_sql, I_conf, RTT_ewma, Loss_ewma]"
], fc="#EFF6FF", ec="#1D4ED8", tc="#1E3A8A")

draw_box(ax1, 1.0, 3.0, 21.5, 25.0, "2. Online Transport Controller", [
    "Per-Class Feasibility Gate:",
    " - Prunes non-FHE arms on conf queries",
    " - Enforces per-class accuracy target",
    "Transport Selection Policies:",
    " - Static Pooled UDP+INT8+1CT (M8)",
    " - RFC 9308 Timeout->TCP Fallback Rule",
    " - Online 5D LinUCB / D-LinUCB Bandits",
    "   (beneficial under UDP policing C3)"
], fc="#FEF3C7", ec="#B45309", tc="#78350F")

# Swimlane 2 (Center-Left): Cryptographic Session & Transport Framing
draw_box(ax1, 25.5, 31.0, 24.5, 21.2, "3. Hybrid PQC Key Establishment", [
    "OpenSSL 3.5.5 + liboqs (1-RTT Init):",
    " - KEM: X25519 + ML-KEM-768 (FIPS 203)",
    " - Auth: ML-DSA-65 chain (14.67 KB cold)",
    "Session Amortization & Replay Guard:",
    " - Persistent TLS 1.3 connection pool",
    " - Exporter-derived AEAD session keys",
    " - Partitioned Bloom filter (0-RTT idemp.)"
], fc="#F0FDF4", ec="#15803D", tc="#14532D")

draw_box(ax1, 25.5, 3.0, 24.5, 25.0, "4. Dual-Path Data Transport", [
    "Path A: Kernel Pooled TLS 1.3 / TCP",
    " - Zero middlebox policing; HOL under loss",
    "Path B: User-Space Paced UDP Framing",
    " - Selective NACK frame retransmission",
    " - Single-MTU INT8 request (<= 1,500 B)",
    " - RFC 9308 timeout fallback to Path A",
    "   when middleboxes throttle UDP :443"
], fc="#ECFEFF", ec="#0E7490", tc="#164E63")

# Swimlane 3 (Center-Right): Dual-Tier Cloud Execution Engine
draw_box(ax1, 53.0, 31.0, 24.5, 21.2, "5A. Public Relational & ANN Tier", [
    "PostgreSQL 18.6 + pgvector 0.8.1:",
    " - Asymmetric INT8 query quantization",
    " - Prop. 1: Cond. Var <= Delta_q^2 / 4,",
    "   Marginal E[Var] = Delta_q^2 / 6",
    " - 5.39x (SIFT) & 22.76x (BGE-M3) tighter",
    "   95% quantile vs. Holder l1 bound",
    " - HNSW index (up to 1M SIFT / 100k BGE)"
], fc="#F5F3FF", ec="#6D28D9", tc="#4C1D95")

draw_box(ax1, 53.0, 3.0, 24.5, 25.0, "5B. Coordinate-Confidential Tier", [
    "OpenFHE C++ RNS-CKKS (N=8192, L=2):",
    " - log2 Q = 140 <= 218 bits (128-bit classic)",
    " - Constant-probe r=2 padded buckets (B_max)",
    " - Server SIMD slot-mask + rotate + sum",
    " - 1-CT response: 909.3 -> 129.0 KiB (7.05x)",
    " - Client decrypt: 3.28 -> 0.59 ms",
    " - Downlink crossover B* = 423-464 Mbps"
], fc="#FFF1F2", ec="#BE123C", tc="#881337")

# Swimlane 4 (Right Sidebar): Empirical Scope & Honest Guarantees
draw_box(ax1, 80.2, 3.0, 18.8, 49.2, "6. Verified Empirical Scope", [
    "Primary Latency Drivers:",
    " - PQC session pooling (+52.2 pp",
    "   factorial SLA; p_MC = 0.0001)",
    " - 1-MTU INT8 over lossy WAN",
    "",
    "Leakage & Security Scope:",
    " - Pad |B_k|=B_max -> A1 MI=0 bits",
    " - IND-CPA^D safe (client-only dec.)",
    " - Zipf query-frequency A3 remains",
    "   exposed without ORAM/dummy cover",
    "",
    "Corpus Scaling Trade-off:",
    " - CKKS noise loss |dR| <= 0.005",
    " - Fixed M=224 Recall@10: 0.788 (5k)",
    "   -> 0.636 (25k) -> 0.550 (100k)",
    " - Reaching 0.922 at 5k needs M=512"
], fc="#FFFBEB", ec="#D97706", tc="#92400E", body_fs=7.2)

draw_arrow(ax1, 11.7, 31.0, 11.7, 28.2, "#1D4ED8")
draw_arrow(ax1, 22.5, 41.5, 25.5, 41.5, "#15803D")
draw_arrow(ax1, 22.5, 15.5, 25.5, 15.5, "#B45309")
draw_arrow(ax1, 37.7, 31.0, 37.7, 28.2, "#0E7490")
draw_arrow(ax1, 50.0, 20.0, 53.0, 40.0, "#6D28D9")
draw_arrow(ax1, 50.0, 13.0, 53.0, 15.5, "#BE123C")
draw_arrow(ax1, 77.5, 41.5, 80.2, 41.5, "#D97706")
draw_arrow(ax1, 77.5, 15.5, 80.2, 15.5, "#D97706")

fig1.tight_layout(pad=0.3)
fig1.savefig(fig_dir / "fig1_architecture.pdf", bbox_inches="tight")
fig1.savefig(fig_dir / "fig1_architecture.png", dpi=300, bbox_inches="tight")
shutil.copy2(fig_dir / "fig1_architecture.png", res_dir / "CAPQ-Pareto-Fig1-Preview-V6.png")
plt.close(fig1)

# --- FIGURE 2: Proposition 1 Same-Quantile 95% Bound + Conditional Variance <= Delta_q^2/4 Audit ---
p1_rows = list(csv.DictReader((logs_v6 / "v6_prop1_conditional_vs_marginal_10k.csv").open("r", encoding="utf-8")))
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(7.2, 2.75))
labels_p1 = ["SIFT-128d\nDeterm.", "SIFT-128d\nStoch.", "BGE-1024d\nDeterm.", "BGE-1024d\nStoch."]
emp_q95 = [float(r["emp_abs_q95"]) * 1e3 for r in p1_rows]
clt_q95 = [float(r["clt95_halfwidth_196sigma"]) * 1e3 for r in p1_rows]
hol_bnd = [float(r["worst_case_holder_bound"]) * 1e3 for r in p1_rows]
x2 = np.arange(len(labels_p1))
w2 = 0.26
ax2a.bar(x2 - w2, emp_q95, w2, label="Empirical 95% |Error|", color="#2563EB")
ax2a.bar(x2, clt_q95, w2, label=r"Analytical $1.96\sigma$ (Marginal)", color="#10B981")
ax2a.bar(x2 + w2, hol_bnd, w2, label=r"Worst-Case Hölder $\ell_1$", color="#EF4444", alpha=0.82)
ax2a.set_yscale("log")
ax2a.set_xticks(x2)
ax2a.set_xticklabels(labels_p1)
ax2a.set_ylabel(r"Inner-Product Error ($\times 10^{-3}$)")
ax2a.set_title("(a) Same-Quantile (95%) Error vs. Hölder")
ax2a.legend(loc="upper left", frameon=True)
ax2a.grid(axis="y", linestyle="--", alpha=0.35)

# Panel (b): Tightening factor & conditional variance verification
tight_x = [float(r["same_quantile_95_tighter_vs_holder"]) for r in p1_rows]
bars2b = ax2b.bar(x2, tight_x, width=0.48, color=["#1D4ED8", "#3B82F6", "#047857", "#10B981"])
for b_i, v_i in zip(bars2b, tight_x):
    ax2b.text(b_i.get_x() + b_i.get_width()/2, v_i + 0.7, f"{v_i:.2f}x", ha="center", va="bottom", fontsize=7.6, fontweight="bold")
ax2b.set_xticks(x2)
ax2b.set_xticklabels(labels_p1)
ax2b.set_ylim(0, 27.5)
ax2b.set_ylabel("95%-Quantile Tightening Factor (x)")
ax2b.set_title(r"(b) Tightening ($\mathrm{Var}_{\xi|\mathbf{q},\mathbf{x}} \leq \Delta_q^2/4$ Verified)")
ax2b.grid(axis="y", linestyle="--", alpha=0.35)
fig2.tight_layout(pad=0.5)
fig2.savefig(fig_dir / "fig2_prop1_same_quantile95.pdf", bbox_inches="tight")
plt.close(fig2)

# --- FIGURE 3 (Ordered cleanly as Fig. 3 in Sec. 4): PG18.6 HNSW Million-Scale Recall & Concurrency ---
m1m_rows = list(csv.DictReader((logs_v6 / "v6_ref_million_hnsw_ef_sweep.csv").open("r", encoding="utf-8")))
conc_rows = list(csv.DictReader((logs_v6 / "v5_pg18_hnsw_concurrency_2000ops_per_level.csv").open("r", encoding="utf-8")))
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(7.2, 2.75))
s1m_pg = [r for r in m1m_rows if "SIFT1M" in r["dataset"] and "PG18" in r["engine"]]
bge_pg = [r for r in m1m_rows if "BGE" in r["dataset"] and "PG18" in r["engine"]]
ax3a.plot([float(r["p95_ms"]) for r in s1m_pg], [float(r["recall_at_10"]) for r in s1m_pg], "o-", color="#1D4ED8", lw=1.6, label="SIFT1M (1,000,000 vec, 128d)")
ax3a.plot([float(r["p95_ms"]) for r in bge_pg], [float(r["recall_at_10"]) for r in bge_pg], "s--", color="#047857", lw=1.6, label="BGE-M3 (100,000 vec, 1024d)")
ax3a.axhline(0.92, color="#DC2626", linestyle=":", lw=1.1, label="Target Recall@10 = 0.92")
ax3a.set_xlabel("PostgreSQL 18.6 HNSW P95 Latency (ms)")
ax3a.set_ylabel("Measured Recall@10")
ax3a.set_title("(a) 1M-Vector & 100k-RAG HNSW Sweep")
ax3a.legend(loc="lower right", frameon=True)
ax3a.grid(True, linestyle="--", alpha=0.35)

c_lvls = [int(r["concurrency_c"]) for r in conc_rows]
tput_c = [float(r["throughput_ops_s"]) for r in conc_rows]
p95_c  = [float(r["p95_ms"]) for r in conc_rows]
ax3b_t = ax3b.twinx()
l1 = ax3b.plot(c_lvls, tput_c, "o-", color="#2563EB", lw=1.6, label="Throughput (ops/s)")
l2 = ax3b_t.plot(c_lvls, p95_c, "^--", color="#D97706", lw=1.6, label="P95 Latency (ms)")
ax3b.set_xlabel("Concurrent SQL+HNSW Clients (2,000 ops/level)")
ax3b.set_ylabel("Throughput (queries/s)", color="#2563EB")
ax3b_t.set_ylabel("P95 Latency (ms)", color="#D97706")
ax3b.set_title("(b) Multi-Client HNSW Concurrency")
lines3 = l1 + l2
ax3b.legend(lines3, [x.get_label() for x in lines3], loc="center right", frameon=True)
ax3b.grid(True, linestyle="--", alpha=0.35)
fig3.tight_layout(pad=0.5)
fig3.savefig(fig_dir / "fig3_pg18_hnsw_million_and_concurrency.pdf", bbox_inches="tight")
plt.close(fig3)

# --- FIGURE 4: Confidential CKKS Plaintext-vs-CKKS Scaling & 1-CT Bandwidth Crossover ---
scl_rows = list(csv.DictReader((logs_v6 / "v6_ckks_scaling_and_breakeven_500q.csv").open("r", encoding="utf-8")))
xov_rows = [r for r in csv.DictReader((logs_v6 / "v6_ckks_1ct_bandwidth_crossover_curve.csv").open("r", encoding="utf-8")) if int(r["enclave_vectors_N"]) == 5000]
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(7.2, 2.8))

ns_x = ["5k", "25k", "100k"]
x4 = np.arange(3)
m224_plain = [float(r["plaintext_fp64_recall10"]) for r in scl_rows if r["profile"] == "SPANN_r2_M224_True1CT_N8192"]
m224_ckks  = [float(r["ckks_openfhe_recall10"]) for r in scl_rows if r["profile"] == "SPANN_r2_M224_True1CT_N8192"]
m512_ckks  = [float(r["ckks_openfhe_recall10"]) for r in scl_rows if r["profile"] == "SPANN_r4_M512_True1CT_N8192"]
ax4a.plot(x4, m224_plain, "o--", color="#64748B", lw=1.5, label="Plaintext FP64 (M=224)")
ax4a.plot(x4, m224_ckks, "s-", color="#2563EB", lw=1.6, label="CKKS 1-CT (r=2, M=224)")
ax4a.plot(x4, m512_ckks, "^-", color="#059669", lw=1.6, label="CKKS 1-CT (r=4, M=512)")
ax4a.axhline(0.92, color="#DC2626", linestyle=":", lw=1.1, label="0.92 Recall Target")
ax4a.set_xticks(x4)
ax4a.set_xticklabels(ns_x)
ax4a.set_ylim(0.45, 1.00)
ax4a.set_xlabel("Confidential Enclave Corpus Size (N)")
ax4a.set_ylabel("Recall@10 (500 SIFT Queries)")
ax4a.set_title(r"(a) Plaintext vs. CKKS Recall ($|\Delta R_{\mathrm{CKKS}}|\leq 0.005$)")
ax4a.legend(loc="lower left", fontsize=6.9, frameon=True)
ax4a.grid(True, linestyle="--", alpha=0.35)

bws = [float(r["link_goodput_mbps"]) for r in xov_rows]
t_mct = [float(r["multict_comp_plus_downlink_ms"]) for r in xov_rows]
t_1ct = [float(r["true1ct_comp_plus_downlink_ms"]) for r in xov_rows]
b_star = float(xov_rows[0]["downlink_breakeven_mbps"])
ax4b.plot(bws, t_mct, "o--", color="#DC2626", lw=1.6, label="Multi-CT (909.3 KiB, N=4096)")
ax4b.plot(bws, t_1ct, "s-", color="#1D4ED8", lw=1.6, label="True 1-CT (129.0 KiB, N=8192)")
ax4b.axvline(b_star, color="#7C3AED", linestyle=":", lw=1.3, label=f"Crossover B* = {b_star:.0f} Mbps")
ax4b.set_xscale("log")
ax4b.set_yscale("log")
ax4b.set_xlabel("Downlink Bandwidth Goodput (Mbps)")
ax4b.set_ylabel("Compute + Downlink Time (ms)")
ax4b.set_title("(b) 1-CT vs. Multi-CT Bandwidth Crossover")
ax4b.legend(loc="upper right", fontsize=6.9, frameon=True)
ax4b.grid(True, which="both", linestyle="--", alpha=0.35)
fig4.tight_layout(pad=0.5)
fig4.savefig(fig_dir / "fig4_ckks_1ct_scaling_and_crossover.pdf", bbox_inches="tight")
plt.close(fig4)

# --- FIGURE 5: Bandit C3 (with RFC 9308 Rule Fallback) & RAPL Silicon Energy Uncertainty ---
b6_rows = list(csv.DictReader((logs_v6 / "v6_bandit_5d_with_rfc9308_rule_and_tails.csv").open("r", encoding="utf-8")))
c3_sub = {r["controller"]: r for r in b6_rows if "Bidirectional_RFC9308" in r["campaign"]}
rapl_rows = [r for r in csv.DictReader((logs_v6 / "v5_steady_rapl_omp_energy_gum_u95.csv").open("r", encoding="utf-8")) if r["workload_mode"] == "single_1ct_m224"]

fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(7.2, 2.85))
c3_order = [
    ("M8\n(No Fall.)", "M8_Static_QUIC_INT8_SPANN_1CT", "#EF4444"),
    ("TCP\n1-CT", "Static_TCP_FP32_SPANN_1CT", "#64748B"),
    ("RFC9308\nRule(H32)", "Rule_QUIC_RFC9308_TCP_Fallback_H32", "#F59E0B"),
    ("Stat.\nLinUCB", "Stationary_LinUCB_5D_Gated", "#10B981"),
    ("CAPQ\nD-LinUCB", "CAPQ_DLinUCB_5D_Gated", "#2563EB"),
]
x5 = np.arange(len(c3_order))
c3_sla = [float(c3_sub[k]["lat_priv_sla_pct"]) for _, k, _ in c3_order]
c3_cols = [col for _, _, col in c3_order]
bars5a = ax5a.bar(x5, c3_sla, width=0.54, color=c3_cols)
for bi, vi in zip(bars5a, c3_sla):
    ax5a.text(bi.get_x() + bi.get_width()/2, vi + 1.2, f"{vi:.1f}%", ha="center", va="bottom", fontsize=7.2, fontweight="bold")
ax5a.set_xticks(x5)
ax5a.set_xticklabels([lbl for lbl, _, _ in c3_order], fontsize=7.2)
ax5a.set_ylim(0, 106)
ax5a.set_ylabel("Latency+Privacy SLA Pass (%)")
ax5a.set_title("(a) C3 UDP Policing: Rule Fallback vs. Bandits")
ax5a.grid(axis="y", linestyle="--", alpha=0.35)

omp_t = [int(r["omp_threads"]) for r in rapl_rows]
gross_e = [float(r["gross_energy_mj"]) for r in rapl_rows]
gross_u = [float(r["gross_u95_mj"]) for r in rapl_rows]
net_e   = [float(r["net_energy_mj"]) for r in rapl_rows]
net_u   = [float(r["net_u95_mj"]) for r in rapl_rows]
ax5b.errorbar(omp_t, gross_e, yerr=gross_u, fmt="o-", color="#DC2626", capsize=3, lw=1.5, label=r"Gross Package Energy $\pm U_{95}$")
ax5b.errorbar(omp_t, net_e, yerr=net_u, fmt="s--", color="#1D4ED8", capsize=3, lw=1.5, label=r"Idle-Subtracted Net $\pm U_{95}$")
ax5b.set_xticks(omp_t)
ax5b.set_xlabel("OpenMP Threads (1-CT M=224 Steady-State)")
ax5b.set_ylabel("Energy per Query (mJ)")
ax5b.set_title("(b) Silicon RAPL Energy & GUM Uncertainty")
ax5b.legend(loc="upper right", fontsize=6.9, frameon=True)
ax5b.grid(True, linestyle="--", alpha=0.35)
fig5.tight_layout(pad=0.5)
fig5.savefig(fig_dir / "fig5_bandit_rfc9308_and_rapl_energy.pdf", bbox_inches="tight")
plt.close(fig5)

# --- FIGURE 6: 2^5 Factorial Monte Carlo Block Randomization + Adversarial Leakage Attacks A1-A3 ---
glm_rows = list(csv.DictReader((logs_v6 / "v5_factorial_2pow5_glm_and_10k_block_permutation.csv").open("r", encoding="utf-8")))[:5]
leak_rows = list(csv.DictReader((logs_v6 / "v6_ref_adversarial_leakage_attacks.csv").open("r", encoding="utf-8")))

fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(7.2, 2.8))
f_names = ["A: PQC Pool", "B: SPANN Rep.", "C: SIMD Enc.", "D: D-Bandit", "E: SLA Gate"]
f_eff   = [float(r["sla_diff_pp"]) for r in glm_rows]
f_lo    = [float(r["sla_diff_pp"]) - float(r["sla_ci95_lo"]) for r in glm_rows]
f_hi    = [float(r["sla_ci95_hi"]) - float(r["sla_diff_pp"]) for r in glm_rows]
y6 = np.arange(len(f_names))[::-1]
cols6 = ["#1D4ED8" if float(r["block_perm10k_p_holm5"]) < 0.05 else "#94A3B8" for r in glm_rows]
ax6a.barh(y6, f_eff, xerr=[f_lo, f_hi], color=cols6, height=0.52, capsize=3)
ax6a.axvline(0, color="#0F172A", lw=0.9)
ax6a.set_yticks(y6)
ax6a.set_yticklabels(f_names)
ax6a.set_xlabel("SLA Effect Size (pp, 95% CI)")
ax6a.set_title("(a) 2^5 Factorial (Monte Carlo Block Perm.)")
ax6a.grid(axis="x", linestyle="--", alpha=0.35)

lk_lbls = ["A1 Unpad.\nSize", "A1 Padded\nB_max", "A2 Timing\nSide-Ch.", "A3 Zipf\nFreq.(Top5)"]
lk_succ = [float(r["adversary_success_pct"]) for r in leak_rows]
lk_cols = ["#EF4444", "#10B981", "#10B981", "#F59E0B"]
x6b = np.arange(len(lk_lbls))
bars6b = ax6b.bar(x6b, lk_succ, width=0.52, color=lk_cols)
for bi, vi in zip(bars6b, lk_succ):
    ax6b.text(bi.get_x() + bi.get_width()/2, vi + 1.8, f"{vi:.1f}%", ha="center", va="bottom", fontsize=7.2, fontweight="bold")
ax6b.set_xticks(x6b)
ax6b.set_xticklabels(lk_lbls, fontsize=7.1)
ax6b.set_ylim(0, 115)
ax6b.set_ylabel("Adversary Identification (%)")
ax6b.set_title("(b) Empirical Leakage Audit (A1-A3)")
ax6b.grid(axis="y", linestyle="--", alpha=0.35)
fig6.tight_layout(pad=0.5)
fig6.savefig(fig_dir / "fig6_factorial_and_leakage_audit.pdf", bbox_inches="tight")
plt.close(fig6)

# --- FIGURE 7: Dual-Track 3-Region WAN Mean & P95 Latency (Including Honest US-Central Exploration Penalty) ---
t2_rows = list(csv.DictReader((logs_v6 / "v6_track2_controlled_m1_m8_reconciled.csv").open("r", encoding="utf-8")))
fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(7.2, 2.85))
regs = ["Bangkok", "Singapore", "US-Central"]
x7 = np.arange(3)
w7 = 0.25
t1_map_p = {(r["region"], r["mode"]): r for r in t1_v6_rows}
ax7a.bar(x7 - w7, [float(t1_map_p[(rg, "M1_Fresh_TLS13_PQC")]["mean_lat_ms"]) for rg in regs], w7, label="M1 Fresh PQC", color="#EF4444")
ax7a.bar(x7,      [float(t1_map_p[(rg, "M3_Pooled_TLS13_Hybrid768")]["mean_lat_ms"]) for rg in regs], w7, label="M3 Pooled Hybrid", color="#64748B")
ax7a.bar(x7 + w7, [float(t1_map_p[(rg, "M5_CAPQ_V2_Adaptive")]["mean_lat_ms"]) for rg in regs], w7, label="M5 Adaptive (V2 Campaign)", color="#2563EB")
ax7a.set_yscale("log")
ax7a.set_xticks(x7)
ax7a.set_xticklabels(regs)
ax7a.set_ylabel("Mean User Latency (ms, log)")
ax7a.set_title("(a) Track 1: Live Cellular/Wireless WAN")
ax7a.legend(loc="upper left", fontsize=6.8, frameon=True)
ax7a.grid(axis="y", which="both", linestyle="--", alpha=0.35)

t2_p_map = {(r["region_profile"], r["mode"]): r for r in t2_rows}
r_profs = ["Bangkok_Calibrated_ASE3", "Singapore_Calibrated_ASE1", "US_Central_Calibrated_USC1"]
ax7b.bar(x7 - w7, [float(t2_p_map[(rp, "M7_Pooled_PQC_TCP_FP32_SPANN_MultiCT")]["mean_lat_ms"]) for rp in r_profs], w7, label="M7 Pooled TCP Multi-CT", color="#94A3B8")
ax7b.bar(x7,      [float(t2_p_map[(rp, "M8_Pooled_PQC_QUIC_INT8_SPANN_1CT")]["mean_lat_ms"]) for rp in r_profs], w7, label="M8 Static UDP+INT8+1CT", color="#10B981")
ax7b.bar(x7 + w7, [float(t2_p_map[(rp, "M5_CAPQ_Pareto_V5_Adaptive_1CT")]["mean_lat_ms"]) for rp in r_profs], w7, label="M5 Bandit (TCP explore tail)", color="#2563EB")
ax7b.set_yscale("log")
ax7b.set_xticks(x7)
ax7b.set_xticklabels(regs)
ax7b.set_ylabel("Mean User Latency (ms, log)")
ax7b.set_title("(b) Track 2: Calibrated Multi-Region WAN")
ax7b.legend(loc="upper left", fontsize=6.8, frameon=True)
ax7b.grid(axis="y", which="both", linestyle="--", alpha=0.35)
fig7.tight_layout(pad=0.5)
fig7.savefig(fig_dir / "fig7_dual_track_wan_latency.pdf", bbox_inches="tight")
plt.close(fig7)

# --- FIGURE 8: Unified Cloud Cost vs. Latency & Per-Class Confidential SLA Trade-off (Singapore Track 2) ---
sin_t2 = [r for r in t2_rows if r["region_profile"] == "Singapore_Calibrated_ASE1"]
fig8, (ax8a, ax8b) = plt.subplots(1, 2, figsize=(7.2, 2.85))
short_m = {
    "M1_Fresh_PQC_TCP_SPANN_MultiCT": "M1",
    "M2_Fresh_Classical_TLS_Plain": "M2*",
    "M3_Pooled_Classical_TLS_Plain": "M3*",
    "M4_Pooled_PQC_TCP_Disjoint_r1_1CT": "M4",
    "M6_Pooled_PQC_TCP_Disjoint_r2_1CT": "M6",
    "M7_Pooled_PQC_TCP_FP32_SPANN_MultiCT": "M7",
    "M8_Pooled_PQC_QUIC_INT8_SPANN_1CT": "M8",
    "M5_CAPQ_Pareto_V5_Adaptive_1CT": "M5",
}
for r in sin_t2:
    sm = short_m[r["mode"]]
    c_val = float(r["unified_cloud_cost_1m_usd"])
    l_val = float(r["mean_lat_ms"])
    col = "#DC2626" if "*" in sm else ("#10B981" if sm == "M8" else ("#2563EB" if sm == "M5" else "#475569"))
    ax8a.scatter(c_val, l_val, s=48, color=col, zorder=3)
    ax8a.annotate(sm, (c_val, l_val), textcoords="offset points", xytext=(4, 3), fontsize=7.3, fontweight="bold")
ax8a.set_yscale("log")
ax8a.set_xlabel("Unified Cloud CPU+Egress Cost ($ / 1M Queries)")
ax8a.set_ylabel("Singapore Mean Latency (ms, log)")
ax8a.set_title("(a) Unified Cloud Cost vs. Latency (*No FHE)")
ax8a.grid(True, which="both", linestyle="--", alpha=0.35)

conf_modes = [r for r in sin_t2 if float(r["conf_mean_recall10"]) > 0.0]
x8 = np.arange(len(conf_modes))
w8 = 0.36
ax8b.bar(x8 - w8/2, [float(r["conf_lat250_priv_pct"]) for r in conf_modes], w8, label=r"Conf. Latency $\leq 250$ms (%)", color="#2563EB")
ax8b.bar(x8 + w8/2, [float(r["conf_mean_recall10"]) * 100.0 for r in conf_modes], w8, label="Conf. Recall@10 (%)", color="#10B981")
ax8b.axhline(92.0, color="#DC2626", linestyle=":", lw=1.1, label="92% Strict Target")
ax8b.set_xticks(x8)
ax8b.set_xticklabels([short_m[r["mode"]] for r in conf_modes])
ax8b.set_ylim(0, 108)
ax8b.set_ylabel("Percentage (%)")
ax8b.set_title("(b) Per-Class Conf. Latency vs. Recall@10 (M=224)")
ax8b.legend(loc="upper left", fontsize=6.7, frameon=True)
ax8b.grid(axis="y", linestyle="--", alpha=0.35)
fig8.tight_layout(pad=0.5)
fig8.savefig(fig_dir / "fig8_unified_cost_and_per_class_sla.pdf", bbox_inches="tight")
plt.close(fig8)

# Remove old out-of-order figure files if present
for old_f in ("fig3_ckks_1ct_and_scaling.pdf", "fig4_pg18_hnsw_concurrency.pdf", "fig5_bandit_and_rapl_energy.pdf",
              "fig6_factorial_glm_permutation.pdf", "fig8_pareto_4d_cost_conf_sla.pdf"):
    op = fig_dir / old_f
    if op.exists():
        op.unlink()

active_figs = sorted(p.name for p in fig_dir.iterdir())
print(f"  Generated clean V6 figure suite ({len(active_figs)} files): {', '.join(active_figs)}")
print(f"  Preview PNG SHA256: {sha256_file(res_dir / 'CAPQ-Pareto-Fig1-Preview-V6.png')}")