import hashlib
import pathlib
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

v6_root = pathlib.Path(r"D:\Research\CAPQ-Pareto-V6")
res_dir = pathlib.Path(r"D:\Research")
ms_v6   = v6_root / "08_manuscript_ieee_tcc_v6"
fig_dir = ms_v6 / "figures"
tbl_dir = ms_v6 / "tables"

def sha256_64(p: pathlib.Path) -> str:
    d = hashlib.sha256(p.read_bytes()).hexdigest()
    assert len(d) == 64
    return d

# Lock V5 hashes before running any operations
v5_locked = {p.name: sha256_64(p) for p in sorted(res_dir.glob("*V5*")) if p.is_file()}

# ==============================================================================
# 1. RENDER FLAGSHIP 4-SWIMLANE + RIGHT WARM-ORANGE SIDEBAR FIGURE 1 (V6 DATA)
# ==============================================================================
print("=== [1/4] BUILDING FLAGSHIP 4-SWIMLANE + WARM-ORANGE SIDEBAR FIGURE 1 ===")

def build_flagship_v6_fig1(out_pdf: pathlib.Path, out_png: pathlib.Path) -> int:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(13.8, 6.65), dpi=300)
    ax.set_xlim(0, 138)
    ax.set_ylim(0, 66.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    text_nodes = []

    def add_txt(x, y, s, **kwargs):
        t = ax.text(x, y, s, **kwargs)
        text_nodes.append(t)
        return t

    def rbox(x: float, y: float, w: float, h: float, fc: str, ec: str = "none", lw: float = 1.4, rad: float = 0.9, pad: float = 0.22) -> None:
        patch = FancyBboxPatch(
            (x + pad, y + pad), w - 2 * pad, h - 2 * pad,
            boxstyle=f"round,pad={pad},rounding_size={rad}",
            facecolor=fc, edgecolor=ec, linewidth=lw,
        )
        ax.add_patch(patch)

    def v_arrow(x: float, y0: float, y1: float, col: str = "#2C3E50", lw: float = 1.65) -> None:
        arr = FancyArrowPatch(
            (x, y0), (x, y1),
            arrowstyle="-|>,head_length=4.8,head_width=3.0",
            linewidth=lw, color=col, shrinkA=0, shrinkB=0,
        )
        ax.add_patch(arr)

    def h_bi_arrow(x0: float, x1: float, y: float, col: str = "#34495E", lw: float = 1.4) -> None:
        arr = FancyArrowPatch(
            (x0, y), (x1, y),
            arrowstyle="<->,head_length=4.0,head_width=2.5",
            linewidth=lw, color=col, shrinkA=0, shrinkB=0,
        )
        ax.add_patch(arr)

    # --- TOP HEADER TITLE ---
    add_txt(
        69.0, 64.3,
        "CAPQ-PARETO SYSTEM ARCHITECTURE: HYBRID PQC POOLING, 1-MTU INT8, AND 1-CT RNS-CKKS WORKFLOW",
        ha="center", va="center", fontsize=11.6, fontweight="bold", color="#111111",
    )

    # =========================================================================
    # BAND 1: WORKLOAD & TELEMETRY LAYER (y = 49.0 .. 61.6)
    # =========================================================================
    rbox(1.0, 49.0, 16.4, 12.6, fc="#EAECEE", ec="#BDC3C7", lw=1.1, rad=1.0)
    add_txt(9.2, 55.3, "WORKLOAD &\nTELEMETRY\nLAYER", ha="center", va="center", fontsize=9.0, fontweight="bold", color="#1B2631", linespacing=1.2)

    rbox(18.2, 49.0, 90.4, 12.6, fc="#F4F6F6", ec="#7F8C8D", lw=1.55, rad=1.0)
    add_txt(
        63.4, 59.5,
        "MIXED VECTOR-SQL STREAM & LAGGED TELEMETRY SUITE (SQL | JSON | SIFT1M 128d | BGE-M3 1024d)",
        ha="center", va="center", fontsize=9.1, fontweight="bold", color="#111111",
    )

    b1_cards = [
        (19.8, "Relational & Public ANN Stream\n(SQL/JSON Ops | 1M PG18.6 HNSW)"),
        (49.4, "Lagged 5D Pre-Context x_{t-}\n([1, I_sql, I_conf, RTT_ewma, Loss])"),
        (79.0, "Silicon RAPL & Cloud Billing Meter\n(Package0 mJ +- U95 | vCPU+Egress)"),
    ]
    for bx, btxt in b1_cards:
        rbox(bx, 50.0, 28.0, 7.3, fc="#E5E8E8", ec="#707B7C", lw=1.3, rad=0.8)
        add_txt(bx + 14.0, 53.65, btxt, ha="center", va="center", fontsize=7.8, color="#17202A", linespacing=1.25)

    for ax_x in [33.8, 63.4, 93.0]:
        v_arrow(ax_x, 49.0, 45.8, col="#34495E")

    # =========================================================================
    # BAND 2: TRANSPORT & CONTROL FABRIC (y = 36.0 .. 45.8)
    # =========================================================================
    rbox(1.0, 36.0, 107.6, 9.8, fc="#EBF5FB", ec="#AED6F1", lw=1.1, rad=1.0)
    add_txt(9.2, 40.9, "TRANSPORT &\nCONTROL\nFABRIC", ha="center", va="center", fontsize=9.0, fontweight="bold", color="#1B2631", linespacing=1.2)

    rbox(18.2, 36.9, 90.4, 8.0, fc="#99A8B8", ec="#5D6D7E", lw=1.55, rad=0.9)
    add_txt(
        63.4, 42.5,
        "PER-CLASS FEASIBILITY GATE & TRANSPORT SELECTOR (STATIC M8 | RFC 9308 RULE | 5D LINUCB)",
        ha="center", va="center", fontsize=9.1, fontweight="bold", color="#0B131B",
    )
    add_txt(
        63.4, 39.1,
        "(Per-Query Deadlines: <=150 ms Public & <=250 ms Conf. | Prunes Non-FHE Arms on Confidential Queries)",
        ha="center", va="center", fontsize=7.9, color="#111111",
    )

    for cx in [32.4, 63.4, 94.4]:
        v_arrow(cx - 2.0, 36.9, 32.4, col="#2C3E50")
        v_arrow(cx + 2.0, 32.4, 36.9, col="#1E8449")
    add_txt(47.9, 34.65, "Protocol / Arm Dispatch", ha="center", va="center", fontsize=7.1, fontweight="bold", color="#1A5276")
    add_txt(78.9, 34.65, "Observed Latency & SLA", ha="center", va="center", fontsize=7.1, fontweight="bold", color="#196F3D")

    # =========================================================================
    # BAND 3: TRI-REGIME CRYPTO & QUANTIZATION LAYER (y = 18.2 .. 32.4)
    # =========================================================================
    rbox(1.0, 18.2, 107.6, 14.2, fc="#F2F4F4", ec="#BDC3C7", lw=1.1, rad=1.0)
    add_txt(
        9.2, 25.3,
        "TRI-REGIME\nCRYPTO &\nQUANT LAYER\n(PQC + CKKS)",
        ha="center", va="center", fontsize=8.6, fontweight="bold", color="#1B2631", linespacing=1.18,
    )

    regimes = [
        (
            18.2, "#D4E6F1", "#2874A6",
            "REGIME A: POOLED HYBRID PQC\n& RFC 9308 TCP FALLBACK",
            "(+52.15 pp Factorial SLA, p=0.0005)",
            "OpenSSL 3.5.5 X25519+ML-KEM-768\nML-DSA-65 Pool & 64Kb Bloom Guard",
        ),
        (
            49.2, "#D1F2EB", "#148F77",
            "REGIME B: SINGLE-MTU INT8\nPUBLIC HNSW VECTOR PATH",
            "(210B / 1,032B <= 1,500B MTU)",
            "Cond. Var <= dq^2/4; E[Var]=dq^2/6\n5.39x-22.76x Tighter 95% Bound",
        ),
        (
            80.2, "#FCF3CF", "#D68910",
            "REGIME C: CONSTANT-PROBE\nTRUE 1-CT RNS-CKKS PATH",
            "(N=8192, log2 Q=140 <= 218b)",
            "r=2 Padded B_max | 909.3->129.0 KiB\nDecrypt 0.59ms | B* = 423-464 Mbps",
        ),
    ]
    for rx, rfc, rec_c, rtitle, rsub, rfoot in regimes:
        rbox(rx, 19.0, 28.4, 12.6, fc=rfc, ec=rec_c, lw=1.55, rad=0.9)
        add_txt(rx + 14.2, 29.1, rtitle, ha="center", va="center", fontsize=7.9, fontweight="bold", color="#111111", linespacing=1.14)
        add_txt(rx + 14.2, 25.3, rsub, ha="center", va="center", fontsize=7.2, fontweight="bold", color="#1F2D3D")
        add_txt(rx + 14.2, 21.6, rfoot, ha="center", va="center", fontsize=6.95, color="#17202A", linespacing=1.22)

    h_bi_arrow(46.6, 49.2, 25.3, col="#2C3E50")
    h_bi_arrow(77.6, 80.2, 25.3, col="#2C3E50")

    v_arrow(40.1, 19.0, 15.6, col="#2C3E50")
    v_arrow(86.7, 19.0, 15.6, col="#148F77")

    # =========================================================================
    # BAND 4: MULTI-TIER HARDWARE & CLOUD TESTBED (y = 1.0 .. 15.6)
    # =========================================================================
    rbox(1.0, 1.0, 107.6, 14.6, fc="#EAECEE", ec="#BDC3C7", lw=1.1, rad=1.0)
    add_txt(
        9.2, 8.3,
        "MULTI-TIER\nHARDWARE &\nCLOUD TIERS\n(Edge & 3 GCP)",
        ha="center", va="center", fontsize=8.6, fontweight="bold", color="#1B2631", linespacing=1.18,
    )

    # Left Testbed Box: Edge Tier
    rbox(18.2, 1.7, 43.8, 13.2, fc="#D6DBDF", ec="#5D6D7E", lw=1.55, rad=0.9)
    add_txt(40.1, 13.3, "BARE-METAL EDGE & CALIBRATED WAN TIER", ha="center", va="center", fontsize=8.4, fontweight="bold", color="#111111")
    add_txt(40.1, 11.2, "(Kernel Pooled TCP vs. User-Space Paced UDP NACK)", ha="center", va="center", fontsize=7.3, color="#1B2631")

    edge_pills = [
        (19.4, "Core Ultra 9\n185H (x86_64)"),
        (33.6, "Intel RAPL\n(GUM +- U95)"),
        (47.8, "netns + netem\n(300/250/200M)"),
    ]
    for px, ptxt in edge_pills:
        rbox(px, 5.1, 13.0, 4.4, fc="#EAECEE", ec="#5D6D7E", lw=1.1, rad=0.6)
        add_txt(px + 6.5, 7.3, ptxt, ha="center", va="center", fontsize=6.95, fontweight="bold", color="#17202A", linespacing=1.16)
    add_txt(40.1, 3.1, "AES-NI/AVX2, liboqs, +3.8 ms Python UDP vs. C TCP Disclosed", ha="center", va="center", fontsize=6.85, color="#17202A")

    h_bi_arrow(62.0, 64.8, 8.3, col="#1A5276", lw=1.6)
    add_txt(63.4, 10.3, "WAN", ha="center", va="center", fontsize=6.7, fontweight="bold", color="#1A5276")

    # Right Testbed Box: Cloud Tier
    rbox(64.8, 1.7, 43.8, 13.2, fc="#D1F2EB", ec="#148F77", lw=1.55, rad=0.9)
    add_txt(86.7, 13.3, "3-REGION GOOGLE CLOUD & ARM64 DB TIER", ha="center", va="center", fontsize=8.4, fontweight="bold", color="#111111")
    add_txt(86.7, 11.2, "(Dual-Track WAN: Live Wireless & Calibrated Namespace)", ha="center", va="center", fontsize=7.3, color="#0E6251")

    cloud_pills = [
        (66.0, "BKK / SG / US\n(n4-standard-4)"),
        (80.2, "Altra ARM64\n(Crypto Ext.)"),
        (94.4, "PG18.6 HNSW\n& OpenFHE C++"),
    ]
    for px, ptxt in cloud_pills:
        rbox(px, 5.1, 13.0, 4.4, fc="#E8F8F5", ec="#148F77", lw=1.1, rad=0.6)
        add_txt(px + 6.5, 7.3, ptxt, ha="center", va="center", fontsize=6.95, fontweight="bold", color="#0B5345", linespacing=1.16)
    add_txt(86.7, 3.1, "9.8 / 56.1 / 225.5 ms 1-RTT, Unified vCPU + RNS Egress Cost", ha="center", va="center", fontsize=6.85, color="#0B5345")

    # =========================================================================
    # RIGHT SIDEBAR: EMPIRICAL SCOPE & VERIFIED FINDINGS (x = 110.0 .. 137.0)
    # =========================================================================
    rbox(110.0, 1.0, 27.0, 60.6, fc="#F2F4F4", ec="#7F8C8D", lw=1.55, rad=1.1)
    add_txt(
        123.5, 56.8,
        "CAPQ-PARETO\nEMPIRICAL SCOPE\n& VERIFICATION",
        ha="center", va="center", fontsize=9.1, fontweight="bold", color="#111111", linespacing=1.18,
    )

    # Inner Warm Orange Card (Signature Reference Style)
    rbox(111.3, 9.8, 24.4, 42.4, fc="#FEF5E7", ec="#E67E22", lw=2.1, rad=1.0)
    add_txt(123.5, 49.6, "VERIFIED FINDINGS", ha="center", va="center", fontsize=8.9, fontweight="bold", color="#111111")

    findings = [
        (43.2, "1. Primary Latency Drivers\n   Pooling (+52.2 pp SLA) &\n   1-MTU INT8 Dominate Gains"),
        (33.8, "2. Plain-vs-CKKS & Scaling\n   |dR_CKKS|<=0.0052; M=224:\n   0.788(5k)->0.550(100k)"),
        (24.4, "3. Leakage & IND-CPA^D\n   Pad B_max: A1=0b; A3 Zipf\n   Top-5=100% w/o ORAM"),
        (15.0, "4. RFC 9308 Fallback Rule\n   M8 Wins Normal WAN; Rule\n   Recovers 92.4% SLA on C3"),
    ]
    for gy, gtxt in findings:
        add_txt(112.2, gy, gtxt, ha="left", va="center", fontsize=7.25, color="#17202A", linespacing=1.22)

    add_txt(
        123.5, 5.3,
        "3,744 WAN + 500 CKKS Tx ->\n14 Vector Figures ->\n64-Hex SHA-256 Verified",
        ha="center", va="center", fontsize=7.8, fontweight="bold", color="#111111", linespacing=1.22,
    )

    # Programmatic Zero-Overlap BBox Verification across all text elements!
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bboxes = [t.get_window_extent(renderer=renderer).expanded(0.96, 0.92) for t in text_nodes]
    overlaps = []
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            if bboxes[i].overlaps(bboxes[j]):
                overlaps.append((text_nodes[i].get_text()[:25], text_nodes[j].get_text()[:25]))
    assert len(overlaps) == 0, f"Detected overlapping text in Figure 1: {overlaps}"

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return len(text_nodes)

n_txt = build_flagship_v6_fig1(
    fig_dir / "fig1_architecture.pdf",
    fig_dir / "fig1_architecture.png",
)
shutil.copy2(fig_dir / "fig1_architecture.png", res_dir / "CAPQ-Pareto-Fig1-Preview-V6.png")
print(f"  [FIG1 STYLE RESTORED] Verified 0 text overlaps across {n_txt} text blocks -> saved PDF & 300-DPI PNG.")

# ==============================================================================
# 2. FIX SECTION / SUBSECTION / FLOAT SPACING IN main.tex AND supplement.tex
# ==============================================================================
print("\n=== [2/4] HARMONIZING SECTION, SUBSECTION & FLOAT SPACING IN main.tex & supplement.tex ===")

main_p = ms_v6 / "main.tex"
m_txt = main_p.read_text(encoding="utf-8")

# 2A. Replace cramped 8pt float sep with balanced IEEE 14pt spacing + \raggedbottom + clean section glue
old_preamble_spacing = (
    "\\emergencystretch=2.0em\n"
    "\\setlength{\\textfloatsep}{8pt plus 2pt minus 2pt}\n"
    "\\setlength{\\dbltextfloatsep}{8pt plus 2pt minus 2pt}"
)
new_preamble_spacing = (
    "\\emergencystretch=2.0em\n"
    "\\raggedbottom\n"
    "\\setlength{\\textfloatsep}{14pt plus 2pt minus 2pt}\n"
    "\\setlength{\\dbltextfloatsep}{14pt plus 2pt minus 2pt}\n"
    "\\setlength{\\floatsep}{10pt plus 2pt minus 2pt}\n"
    "\\setlength{\\dblfloatsep}{10pt plus 2pt minus 2pt}\n"
    "\\setlength{\\abovecaptionskip}{5pt plus 1pt minus 1pt}\n"
    "\\setlength{\\belowcaptionskip}{2pt plus 1pt minus 1pt}"
)
assert old_preamble_spacing in m_txt, "Expected preamble spacing block not found in main.tex!"
m_txt = m_txt.replace(old_preamble_spacing, new_preamble_spacing, 1)

# 2B. Relocate floats that were sitting directly above \section or stacked back-to-back:
# - Move tab1_related.tex to start of Section 2 (right after \section{Related Work}\label{sec:related})
m_txt = m_txt.replace("\n\\input{tables/tab1_related.tex}\n\n\\section{System Model", "\n\n\\section{System Model", 1)
m_txt = m_txt.replace(
    "\\section{Related Work}\\label{sec:related}\n",
    "\\section{Related Work}\\label{sec:related}\n\\input{tables/tab1_related.tex}\n",
    1,
)

# - Move tab3_prop1_hnsw.tex inside Section 4 (right after \section{Public Tier...}) instead of end of Section 3
m_txt = m_txt.replace("\n\\input{tables/tab3_prop1_hnsw.tex}\n\n\\section{Public Tier", "\n\n\\section{Public Tier", 1)
m_txt = m_txt.replace(
    "\\section{Public Tier: Asymmetric INT8 Bounds and Million-Scale PG18.6 HNSW}\\label{sec:public}\n",
    "\\section{Public Tier: Asymmetric INT8 Bounds and Million-Scale PG18.6 HNSW}\\label{sec:public}\n\\input{tables/tab3_prop1_hnsw.tex}\n",
    1,
)

# - Move tab4_ckks.tex inside Section 5 (right after \section{Coordinate-Confidential CKKS Tier...})
m_txt = m_txt.replace("\n\\input{tables/tab4_ckks.tex}\n\n\\section{Coordinate-Confidential", "\n\n\\section{Coordinate-Confidential", 1)
m_txt = m_txt.replace(
    "\\section{Coordinate-Confidential CKKS Tier: True 1-CT Aggregation, Noise Decomposition, and Scaling Limits}\\label{sec:ckks}\n",
    "\\section{Coordinate-Confidential CKKS Tier: True 1-CT Aggregation, Noise Decomposition, and Scaling Limits}\\label{sec:ckks}\n\\input{tables/tab4_ckks.tex}\n",
    1,
)

# - Move tab5_bandit.tex inside Section 6 (right after \section{Online Transport Control...})
m_txt = m_txt.replace("\n\\input{tables/tab5_bandit.tex}\n\n\\section{Online Transport Control", "\n\n\\section{Online Transport Control", 1)
m_txt = m_txt.replace(
    "\\section{Online Transport Control vs.\\ RFC 9308 Fallback and Silicon Energy Metrology}\\label{sec:bandit_energy}\n",
    "\\section{Online Transport Control vs.\\ RFC 9308 Fallback and Silicon Energy Metrology}\\label{sec:bandit_energy}\n\\input{tables/tab5_bandit.tex}\n",
    1,
)

# - Separate stacked tab6_rapl.tex + tab7_factorial.tex: put tab6_rapl inside Sec 6.2 and tab7_factorial inside Sec 7.1!
m_txt = m_txt.replace(
    "\\input{tables/tab6_rapl.tex}\n\\input{tables/tab7_factorial.tex}\n\n\\subsection{Silicon RAPL Energy Metrology and Idle-Drift Uncertainty Limits}\n",
    "\\subsection{Silicon RAPL Energy Metrology and Idle-Drift Uncertainty Limits}\n\\input{tables/tab6_rapl.tex}\n",
    1,
)
m_txt = m_txt.replace(
    "\\subsection{\\texorpdfstring{$2^5$}{2\\^{}5} Factorial Monte Carlo Within-Block Randomization}\n",
    "\\subsection{\\texorpdfstring{$2^5$}{2\\^{}5} Factorial Monte Carlo Within-Block Randomization}\n\\input{tables/tab7_factorial.tex}\n",
    1,
)

# - Add uniform \smallskip\noindent before standalone bold sub-topic paragraphs so visual rhythm is crisp
bold_lead_phrases = [
    r"\textbf{Summary of Empirical Contributions:}",
    r"\textbf{Post-Quantum TLS and QUIC Measurements.}",
    r"\textbf{Encrypted Nearest-Neighbor Search.}",
    r"\textbf{QUIC Applicability and Middlebox Fallback.}",
    r"\textbf{Contextual Bandits in Non-Stationary Systems.}",
    r"\textbf{$\text{IND-CPA}^D$ Considerations for Approximate HE.}",
    r"\textbf{Empirical Leakage Audit (\texttt{A1--A3}) and Miller--Madow Entropy.}",
    r"\textbf{Analytical Downlink Bandwidth Crossover ($B^\star$).}",
]
for bp in bold_lead_phrases:
    m_txt = m_txt.replace("\n\n" + bp, "\n\n\\smallskip\\noindent " + bp)

main_p.write_text(m_txt, encoding="utf-8", newline="\n")

# 2C. Fix supplement.tex spacing: add \raggedbottom, stfloats/placeins, fix invalid table*[h!], pair S1-S6 cleanly
supp_p = ms_v6 / "supplement.tex"
s_txt = supp_p.read_text(encoding="utf-8")

s_txt = s_txt.replace(
    "\\usepackage{amsmath,amssymb,amsfonts,booktabs,graphicx,url,microtype}\n\\usepackage[hidelinks]{hyperref}\n\\Urlmuskip=0mu plus 2mu\\relax\n\\emergencystretch=2.0em",
    "\\usepackage{amsmath,amssymb,amsfonts,booktabs,graphicx,url,microtype}\n"
    "\\IfFileExists{stfloats.sty}{\\usepackage{stfloats}}{}\n"
    "\\IfFileExists{placeins.sty}{\\usepackage{placeins}}{\\newcommand{\\FloatBarrier}{\\clearpage}}\n"
    "\\usepackage[hidelinks]{hyperref}\n"
    "\\Urlmuskip=0mu plus 2mu\\relax\n"
    "\\emergencystretch=2.0em\n"
    "\\raggedbottom\n"
    "\\setlength{\\textfloatsep}{14pt plus 2pt minus 2pt}\n"
    "\\setlength{\\dbltextfloatsep}{14pt plus 2pt minus 2pt}\n"
    "\\setlength{\\floatsep}{10pt plus 2pt minus 2pt}\n"
    "\\setlength{\\dblfloatsep}{10pt plus 2pt minus 2pt}",
    1,
)

# Add brief intro sentence under Supplement Section 2 so the heading never sits孤立 above floats
s_txt = s_txt.replace(
    "\\section{Extended Operational Fault Audit, 1-CT Bandwidth Schedule, and Supplementary Figures}\n\n\\begin{table}[h!]",
    "\\section{Extended Operational Fault Audit, 1-CT Bandwidth Schedule, and Supplementary Figures}\n"
    "Tables~1--2 and Figs.~S1--S6 provide the granular link-bandwidth crossover schedule, nine-scenario operational fault-injection results, stage-level CKKS latency/wire breakdowns, socket regime distributions, OpenMP RAPL uncertainty profiles, uncapped tail-latency comparisons, cross-ISA AEAD goodput, and all ten two-way factorial interaction terms.\n\n"
    "\\begin{table}[!t]",
    1,
)

# Pair supplementary figures cleanly ([!t] and [!b]) and use \FloatBarrier before Section 3 + [!t] for SHA-256 table*
s_txt = s_txt.replace("\\begin{table*}[t]", "\\begin{table*}[!t]")
fig_blocks = list(re.finditer(r"\\begin\{figure\*\}\[t\]", s_txt))
assert len(fig_blocks) == 6, f"Expected 6 figure* blocks in supplement.tex, found {len(fig_blocks)}"
# Alternate [!t] and [!b] with page flush between pairs so every supplement page is evenly balanced!
new_s_parts = []
last_idx = 0
for idx_f, m_f in enumerate(fig_blocks):
    new_s_parts.append(s_txt[last_idx:m_f.start()])
    pos_spec = "[!t]" if (idx_f % 2 == 0) else "[!b]"
    prefix_barrier = "\\FloatBarrier\n" if (idx_f in (2, 4)) else ""
    new_s_parts.append(prefix_barrier + f"\\begin{{figure*}}{pos_spec}")
    last_idx = m_f.end()
new_s_parts.append(s_txt[last_idx:])
s_txt = "".join(new_s_parts)

s_txt = s_txt.replace(
    "\\clearpage\n\\section{Full 64-Character SHA-256 Provenance Manifest}",
    "\\FloatBarrier\n\\clearpage\n\\section{Full 64-Character SHA-256 Provenance Manifest}",
    1,
)
s_txt = s_txt.replace("\\begin{table*}[h!]", "\\begin{table*}[!b]", 1)
s_txt = s_txt.replace(
    "\\textbf{Part (2) Exact Conditional Variance and Sharp $\\Delta_q^2/4$ Bound:}",
    "\\smallskip\\noindent\\textbf{Part (2) Exact Conditional Variance and Sharp $\\Delta_q^2/4$ Bound:}",
    1,
)
supp_p.write_text(s_txt, encoding="utf-8", newline="\n")
print("  [SPACING UPDATED] Balanced 14pt float separation, \\raggedbottom, relocated section-leading tables, and fixed Supplement FloatBarriers.")

# ==============================================================================
# 3. COMPILE ALL 4 PDFs IN WSL2 & RUN AUTOMATED BBOX HEADING SPACING AUDITOR
# ==============================================================================
print("\n=== [3/4] COMPILING ALL 4 PDFs & AUDITING ACTUAL RENDERED SECTION SPACING (pt) ===")
wsl_ms = "/mnt/d/Research/CAPQ-Pareto-V6/08_manuscript_ieee_tcc_v6"

def compile_and_audit_pdf(stem: str):
    bib_cmd = f"bibtex {stem} >> build.log 2>&1 || (cat {stem}.blg && exit 1)" if stem == "main" else ""
    sh_cmd = f"""
set -e
rm -rf /tmp/capq_v6_sp_{stem} && mkdir -p /tmp/capq_v6_sp_{stem}
cp -a "{wsl_ms}/." /tmp/capq_v6_sp_{stem}/
cd /tmp/capq_v6_sp_{stem}
pdflatex -interaction=nonstopmode -halt-on-error {stem}.tex > build.log 2>&1 || (tail -n 50 build.log && exit 1)
{bib_cmd}
pdflatex -interaction=nonstopmode -halt-on-error {stem}.tex >> build.log 2>&1 || (tail -n 50 build.log && exit 1)
pdflatex -interaction=nonstopmode -halt-on-error {stem}.tex >> build.log 2>&1 || (tail -n 50 build.log && exit 1)
pdftotext -bbox-layout {stem}.pdf {stem}_bbox.xhtml
cp -f {stem}.pdf {stem}.log {stem}_bbox.xhtml "{wsl_ms}/"
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

    # Parse XHTML bounding boxes to audit heading vertical gaps in points!
    bbox_p = ms_v6 / f"{stem}_bbox.xhtml"
    heading_Audit = []
    if bbox_p.exists() and stem in ("main", "supplement"):
        raw_xml = bbox_p.read_text(encoding="utf-8", errors="ignore")
        raw_xml = re.sub(r'\sxmlns="[^"]+"', "", raw_xml, count=1)
        root = ET.fromstring(raw_xml)
        for p_idx, page in enumerate(root.findall(".//page"), 1):
            p_h = float(page.attrib.get("height", "792"))
            lines_info = []
            for ln in page.findall(".//line"):
                words = [w.text or "" for w in ln.findall("word")]
                txt_ln = " ".join(words).strip()
                if not txt_ln:
                    continue
                ymin = float(ln.attrib["ymin"])
                ymax = float(ln.attrib["ymax"])
                xmin = float(ln.attrib["xmin"])
                lines_info.append((xmin, ymin, ymax, txt_ln))
            for idx_l, (xmin, ymin, ymax, txt_ln) in enumerate(lines_info):
                # Detect IEEEtran numbered sections & subsections (e.g., "1 Introduction", "3.1 Workload Model...")
                if re.match(r"^(?:[1-8](?:\.[1-4])?\s+[A-Z]|Acknowledgment,\s+Data)", txt_ln):
                    # Find nearest line strictly above in the same column (|xmin_other - xmin| < 120 pt)
                    same_col_above = [l for l in lines_info[:idx_l] if abs(l[0] - xmin) < 120 and l[2] <= ymin + 0.5]
                    same_col_below = [l for l in lines_info[idx_l + 1:] if abs(l[0] - xmin) < 120 and l[1] >= ymax - 0.5]
                    gap_above = round(ymin - max(l[2] for l in same_col_above), 2) if same_col_above else -1.0
                    gap_below = round(min(l[1] for l in same_col_below) - ymax, 2) if same_col_below else -1.0
                    dist_bottom = round(p_h - ymax, 1)
                    assert dist_bottom > 42.0, f"Orphan heading near bottom of {stem}.pdf p.{p_idx}: {txt_ln} (dist={dist_bottom}pt)"
                    if gap_above >= 0.0:
                        assert gap_above >= 4.0, f"Heading crowded from above in {stem}.pdf p.{p_idx}: {txt_ln} ({gap_above}pt)"
                    heading_Audit.append((p_idx, txt_ln[:44], gap_above, gap_below))
        bbox_p.unlink()
    elif bbox_p.exists():
        bbox_p.unlink()

    return pgs, heading_Audit

for st in ("main", "supplement", "response_to_reviewers", "cover_letter"):
    pg, h_audit = compile_and_audit_pdf(st)
    print(f"  [VERIFIED {st+'.pdf':<26}] Pages={pg:2d} | OverfullHbox=0.00pt | OverfullVbox=0.00pt | Undefined=0")
    for p_no, h_title, g_up, g_dn in h_audit:
        up_s = f"{g_up:5.1f}pt" if g_up >= 0 else "col-top"
        dn_s = f"{g_dn:5.1f}pt" if g_dn >= 0 else "col-end"
        print(f"    p.{p_no:02d} | {h_title:<44} | gap_above={up_s} | gap_below={dn_s}")

# ==============================================================================
# 4. EXPORT REFRESHED V6 DELIVERABLES TO D:\Research\ & VERIFY ZERO-TOUCH V5
# ==============================================================================
print("\n=== [4/4] EXPORTING UPDATED V6 PACKAGE & VERIFYING ZERO-TOUCH V5 ===")
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
print("\n[ZERO-TOUCH V5 VERIFIED 100%] All V5 files remain completely untouched.")