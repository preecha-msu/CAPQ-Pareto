from __future__ import annotations
from pathlib import Path
import re
import subprocess
import sys
import zipfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO_ROOT = Path(r"D:\Research\CAPQ-Pareto-V2")
sys.path.insert(0, str(REPO_ROOT / "05_telemetry_audit"))
import sha256_manifest as sm


def build_reference_style_fig1(out_pdf: Path, out_png: Path) -> None:
    """Render Figure 1 matching the exact 4-swimlane + Right Guarantees Sidebar layout."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(13.8, 6.55), dpi=300)
    ax.set_xlim(0, 138)
    ax.set_ylim(0, 65.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def rbox(x: float, y: float, w: float, h: float, fc: str, ec: str = "none", lw: float = 1.4, rad: float = 0.9, pad: float = 0.25) -> None:
        patch = FancyBboxPatch(
            (x + pad, y + pad), w - 2 * pad, h - 2 * pad,
            boxstyle=f"round,pad={pad},rounding_size={rad}",
            facecolor=fc, edgecolor=ec, linewidth=lw,
        )
        ax.add_patch(patch)

    def v_arrow(x: float, y0: float, y1: float, col: str = "#2C3E50", lw: float = 1.7) -> None:
        arr = FancyArrowPatch(
            (x, y0), (x, y1),
            arrowstyle="-|>,head_length=5.0,head_width=3.2",
            linewidth=lw, color=col, shrinkA=0, shrinkB=0,
        )
        ax.add_patch(arr)

    def h_bi_arrow(x0: float, x1: float, y: float, col: str = "#34495E", lw: float = 1.4) -> None:
        arr = FancyArrowPatch(
            (x0, y), (x1, y),
            arrowstyle="<->,head_length=4.2,head_width=2.6",
            linewidth=lw, color=col, shrinkA=0, shrinkB=0,
        )
        ax.add_patch(arr)

    # --- TOP HEADER TITLE ---
    ax.text(
        69.0, 63.3,
        "CAPQ-PARETO MULTI-REGIME CRYPTOGRAPHIC ROUTING & CLOUD-EDGE WORKFLOW",
        ha="center", va="center", fontsize=13.4, fontweight="bold", color="#111111",
    )

    # =========================================================================
    # BAND 1: WORKLOAD & TELEMETRY LAYER (y = 48.2 .. 60.8)
    # =========================================================================
    rbox(1.0, 48.2, 16.4, 12.6, fc="#EAECEE", ec="none", rad=1.0)
    ax.text(9.2, 54.5, "WORKLOAD &\nTELEMETRY\nLAYER", ha="center", va="center", fontsize=9.3, fontweight="bold", color="#1B2631")

    rbox(18.2, 48.2, 90.4, 12.6, fc="#F4F6F6", ec="#7F8C8D", lw=1.6, rad=1.0)
    ax.text(
        63.4, 58.6,
        "HYBRID DB STREAM & TELEMETRY SUITE (TPC-C | YCSB-A | SIFT-128d | BGE-M3 1024d)",
        ha="center", va="center", fontsize=9.7, fontweight="bold", color="#111111",
    )

    b1_cards = [
        (19.8, "ACID SQL & Vector Preprocessor\n(256B-4KB SQL | L2-Norm RAG)"),
        (49.4, "Live WAN RTT & Loss Profiler\n(MTU 1500B & 10D Context x_t)"),
        (79.0, "Hardware RAPL & Cloud Meter\n(pdh.dll mJ & $/s GCP Billing)"),
    ]
    for bx, btxt in b1_cards:
        rbox(bx, 49.3, 28.0, 7.1, fc="#E5E8E8", ec="#707B7C", lw=1.35, rad=0.8)
        ax.text(bx + 14.0, 52.85, btxt, ha="center", va="center", fontsize=8.1, color="#17202A", linespacing=1.25)

    # Downward arrows from Band 1 to Band 2
    for ax_x in [33.8, 63.4, 93.0]:
        v_arrow(ax_x, 48.2, 44.5, col="#34495E")

    # =========================================================================
    # BAND 2: SLA-GATED ROUTING FABRIC (y = 35.4 .. 45.2)
    # =========================================================================
    rbox(1.0, 35.4, 107.6, 9.8, fc="#EBF5FB", ec="none", rad=1.0)
    ax.text(9.2, 40.3, "SLA-GATED\nROUTING\nFABRIC", ha="center", va="center", fontsize=9.3, fontweight="bold", color="#1B2631")

    rbox(18.2, 36.4, 90.4, 7.8, fc="#99A8B8", ec="#5D6D7E", lw=1.6, rad=0.9)
    ax.text(
        63.4, 41.6,
        "CAPQ-PARETO SLA-GATED REGULARIZED LINUCB AI ROUTER",
        ha="center", va="center", fontsize=10.2, fontweight="bold", color="#0B131B",
    )
    ax.text(
        63.4, 38.4,
        "(Sliding-Window p95 <= 150/350 ms & Recall@10 >= 0.92 Gate | < 45 us Decision)",
        ha="center", va="center", fontsize=8.6, color="#111111",
    )

    # Bi-directional arrows (Dispatch downward + 4D Telemetry Feedback upward) between Band 2 & Band 3
    for cx in [32.4, 63.4, 94.4]:
        v_arrow(cx - 2.2, 36.4, 31.8, col="#2C3E50")
        v_arrow(cx + 2.2, 31.8, 36.4, col="#1E8449")
    ax.text(47.9, 34.0, "Action m* / 4D Feedback", ha="center", va="center", fontsize=7.3, fontweight="bold", color="#1A5276")
    ax.text(78.9, 34.0, "SLA Gate Pruning", ha="center", va="center", fontsize=7.3, fontweight="bold", color="#196F3D")

    # =========================================================================
    # BAND 3: TRI-REGIME CRYPTO & QUANTIZATION LAYER (y = 18.0 .. 32.2)
    # =========================================================================
    rbox(1.0, 18.0, 107.6, 14.2, fc="#F2F4F4", ec="none", rad=1.0)
    ax.text(
        9.2, 25.1,
        "TRI-REGIME\nCRYPTO &\nQUANT LAYER\n(10 Actions)",
        ha="center", va="center", fontsize=9.0, fontweight="bold", color="#1B2631", linespacing=1.18,
    )

    regimes = [
        (
            18.2, "#D4E6F1", "#2874A6",
            "REGIME A: LOSS-FREE SQL\nFAST-PATH",
            "(ACID 100% | Batch b = 1..16)",
            "QUIC 0-RTT + X25519_ML-KEM-768\nAES-256-GCM + Nonce Replay Guard",
        ),
        (
            49.2, "#D1F2EB", "#148F77",
            "REGIME B: SINGLE-MTU\nQUANTIZED RAG PATH",
            "(1,032B <= 1 MTU | >=0.92 Recall)",
            "QUIC 0-RTT + ML-KEM-512 + INT8-SQ\nISA-Aware AES-GCM / ChaCha20",
        ),
        (
            80.2, "#FCF3CF", "#D68910",
            "REGIME C: ZERO-TRUST\nBUCKETED FHE PATH",
            "(z_q = 1 Untrusted | N = 4096..8192)",
            "Spherical k-Means Bucket (K_c = 64)\nOpenFHE 1.2.3 CKKS EvalInnerProduct",
        ),
    ]
    for rx, rfc, rec_c, rtitle, rsub, rfoot in regimes:
        rbox(rx, 18.9, 28.4, 12.6, fc=rfc, ec=rec_c, lw=1.6, rad=0.9)
        ax.text(rx + 14.2, 29.0, rtitle, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#111111", linespacing=1.12)
        ax.text(rx + 14.2, 25.3, rsub, ha="center", va="center", fontsize=7.6, fontweight="bold", color="#1F2D3D")
        ax.text(rx + 14.2, 21.6, rfoot, ha="center", va="center", fontsize=7.2, color="#17202A", linespacing=1.22)

    # Inter-regime dynamic switching arrows
    h_bi_arrow(46.6, 49.2, 25.1, col="#2C3E50")
    h_bi_arrow(77.6, 80.2, 25.1, col="#2C3E50")

    # Downward arrows from Band 3 to Band 4
    v_arrow(40.0, 18.9, 15.1, col="#2C3E50")
    v_arrow(86.8, 18.9, 15.1, col="#148F77")

    # =========================================================================
    # BAND 4: MULTI-TIER HARDWARE & CLOUD TESTBED (y = 1.0 .. 15.6)
    # =========================================================================
    rbox(1.0, 1.0, 107.6, 14.6, fc="#EAECEE", ec="none", rad=1.0)
    ax.text(
        9.2, 8.3,
        "MULTI-TIER\nHARDWARE &\nCLOUD TIERS\n(Edge & 3 GCP)",
        ha="center", va="center", fontsize=8.9, fontweight="bold", color="#1B2631", linespacing=1.18,
    )

    # Left Testbed Box: Physical Bare-Metal Edge Tier
    rbox(18.2, 1.8, 43.8, 13.1, fc="#D6DBDF", ec="#5D6D7E", lw=1.6, rad=0.9)
    ax.text(40.1, 13.2, "BARE-METAL EDGE & KERNEL WAN TIER", ha="center", va="center", fontsize=8.9, fontweight="bold", color="#111111")
    ax.text(40.1, 11.1, "(Physical Silicon MSR & Controlled Netem Nodes)", ha="center", va="center", fontsize=7.8, color="#1B2631")

    edge_pills = [
        (19.4, "Core Ultra 9\n185H (x86_64)"),
        (33.6, "Kernel RAPL\n(pdh.dll mJ)"),
        (47.8, "tc/netem WAN\n(MTU 1500B)"),
    ]
    for px, ptxt in edge_pills:
        rbox(px, 5.2, 13.0, 4.3, fc="#EAECEE", ec="#5D6D7E", lw=1.1, rad=0.6)
        ax.text(px + 6.5, 7.35, ptxt, ha="center", va="center", fontsize=7.3, fontweight="bold", color="#17202A", linespacing=1.15)
    ax.text(40.1, 3.2, "AVX2/VAES, liboqs 0.12.0, In-Situ Quiescent Idle Subtracted", ha="center", va="center", fontsize=7.2, color="#17202A")

    # WAN link arrow between Edge Tier and Cloud Tier
    h_bi_arrow(62.0, 64.8, 8.3, col="#1A5276", lw=1.6)
    ax.text(63.4, 10.2, "WAN", ha="center", va="center", fontsize=6.8, fontweight="bold", color="#1A5276")

    # Right Testbed Box: Live 3-Region GCP & ARM64 Tier
    rbox(64.8, 1.8, 43.8, 13.1, fc="#D1F2EB", ec="#148F77", lw=1.6, rad=0.9)
    ax.text(86.7, 13.2, "LIVE 3-REGION GOOGLE CLOUD & ARM64 TIER", ha="center", va="center", fontsize=8.9, fontweight="bold", color="#111111")
    ax.text(86.7, 11.1, "(Geo-Distributed Hybrid DB & Cross-ISA Cloud Nodes)", ha="center", va="center", fontsize=7.8, color="#0E6251")

    cloud_pills = [
        (66.0, "BKK / SG / US\n(n4 & e2 VMs)"),
        (80.2, "Tau T2A ARM64\n(Neoverse-N1)"),
        (94.4, "PG18.6 pgvec\n& Qdrant HNSW"),
    ]
    for px, ptxt in cloud_pills:
        rbox(px, 5.2, 13.0, 4.3, fc="#E8F8F5", ec="#148F77", lw=1.1, rad=0.6)
        ax.text(px + 6.5, 7.35, ptxt, ha="center", va="center", fontsize=7.3, fontweight="bold", color="#0B5345", linespacing=1.15)
    ax.text(86.7, 3.2, "11.5 / 37.3 / 229.6 ms Live UDP:443, vCPU/s + $0.12/GiB Egress", ha="center", va="center", fontsize=7.2, color="#0B5345")

    # =========================================================================
    # RIGHT SIDEBAR: CAPQ-PARETO SYSTEM GUARANTEES (x = 110.0 .. 137.0)
    # =========================================================================
    rbox(110.0, 1.0, 27.0, 59.8, fc="#F2F4F4", ec="#7F8C8D", lw=1.6, rad=1.1)
    ax.text(
        123.5, 55.6,
        "CAPQ-PARETO\n4D SLA & PQC\nGUARANTEES",
        ha="center", va="center", fontsize=9.8, fontweight="bold", color="#111111", linespacing=1.2,
    )

    # Inner Warm Orange Guarantees Card
    rbox(111.4, 9.8, 24.2, 41.2, fc="#FEF5E7", ec="#E67E22", lw=2.1, rad=1.0)
    ax.text(123.5, 48.3, "GUARANTEES", ha="center", va="center", fontsize=9.8, fontweight="bold", color="#111111")

    guarantees = [
        (41.8, "1. Zero HNDL & Bounded\n   Zero-Trust Leakage\n   (FIPS 203 + <=6b CKKS)"),
        (32.5, "2. Strict 150/350 ms SLA\n   & Recall@10 >= 0.92\n   (98.3% US-Central Pass)"),
        (23.0, "3. Single-MTU 4D Pareto\n   (1,032B INT8 | 6.0x\n   Lower Live p99 Latency)"),
        (13.8, "4. Cross-ISA & Silicon\n   RAPL/Billing Audit\n   (x86 vs ARM64 + $/1M)"),
    ]
    for gy, gtxt in guarantees:
        ax.text(112.4, gy, gtxt, ha="left", va="center", fontsize=7.9, color="#17202A", linespacing=1.24)

    ax.text(
        123.5, 5.2,
        "1,980 Real Trials ->\n8 Vector Figures ->\nSHA-256 Verified",
        ha="center", va="center", fontsize=8.4, fontweight="bold", color="#111111", linespacing=1.25,
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.08)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig1_pdf = REPO_ROOT / "00_manuscript/figures/fig1_capq_architecture.pdf"
    fig1_png_repo = REPO_ROOT / "00_manuscript/figures/workflow/capq_pareto_workflow.png"
    fig1_preview = Path(r"D:\Research\CAPQ-Pareto-Fig1-Preview.png")

    build_reference_style_fig1(fig1_pdf, fig1_png_repo)
    fig1_preview.write_bytes(fig1_png_repo.read_bytes())
    print(f"[FIG1 UPGRADED] Saved vector {fig1_pdf.name} & preview {fig1_preview}")

    # Run the clean PDF builder & verifier (ensuring adjustbox + <=12 pages + S1..S7 tables)
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import audit_and_fix_formatting as aff
    aff.main()


if __name__ == "__main__":
    main()