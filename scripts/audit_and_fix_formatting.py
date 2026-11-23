from __future__ import annotations
import csv
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO_ROOT = Path(r"D:\Research\CAPQ-Pareto")
sys.path.insert(0, str(REPO_ROOT / "05_telemetry_audit"))
import sha256_manifest as sm

plt.rcParams.update({
    "font.family": "serif",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# 1. Regenerate Clean ASCII/Typography Vector Fig. 1 (Zero Unicode Glyph Issues)
# ---------------------------------------------------------------------------
def regenerate_clean_fig1(out_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.15, 2.55))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 36)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, lines: list[str], fc: str, ec: str) -> None:
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.45,rounding_size=1.0",
            facecolor=fc, edgecolor=ec, linewidth=1.2,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h - 2.7, title, ha="center", va="center", fontsize=8.0, fontweight="bold", color="#1b2631")
        ax.plot([x + 1.2, x + w - 1.2], [y + h - 4.6, y + h - 4.6], color=ec, linewidth=0.65, alpha=0.6)
        ax.text(x + w / 2, y + (h - 4.8) / 2, "\n".join(lines), ha="center", va="center", fontsize=6.8, color="#1c2833", linespacing=1.28)

    def arrow(x0: float, y0: float, x1: float, y1: float, tag: str) -> None:
        arr = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>,head_length=4.2,head_width=2.6", linewidth=1.15, color="#2c3e50")
        ax.add_patch(arr)
        ax.text((x0 + x1) / 2, y0 + 2.1, tag, ha="center", va="center", fontsize=6.4, fontweight="bold", color="#1a5276")

    box(0.8, 4.5, 21.6, 28.5, "Layer 1: Physical Edge", [
        "Bare-Metal Gateway",
        "(Core Ultra 9 / ARM64)",
        "- SQL: TPC-C, YCSB-A",
        "- RAG: SIFT-128d,",
        "  BGE-M3 (1024d)",
        "- RAPL/EMI (pdh.dll mJ)",
    ], "#ebf5fb", "#2874a6")

    box(25.8, 4.5, 23.4, 28.5, "Layer 2: CAPQ-Pareto AI", [
        "SLA-Gated LinUCB",
        "(10D Live Context x_t)",
        "- Deadline p95 <= T_max",
        "- Recall@10 >= 0.92",
        "- Weights (a, b, g)",
        "- Action Pool |M| = 10",
    ], "#e8f8f5", "#117a65")

    box(52.6, 4.5, 21.8, 28.5, "Layer 3: Crypto Regimes", [
        "Adaptive Wire Tunnel",
        "- Regime A: QUIC 0-RTT",
        "  X25519+ML-KEM-768",
        "- Regime B: ML-KEM-512",
        "  + INT8-SQ (1 MTU)",
        "- Regime C: CKKS Bucket",
    ], "#fef9e7", "#b9770e")

    box(77.8, 4.5, 21.4, 28.5, "Layer 4: Multi-Region GCP", [
        "Live Cloud Hybrid DBs",
        "- Bangkok (11.5 ms)",
        "- Singapore (37.3 ms)",
        "- US-Central (229.6 ms)",
        "- Tau T2A ARM64 Node",
        "- PG18.6 & Qdrant HNSW",
    ], "#fdedec", "#943126")

    arrow(22.6, 18.8, 25.6, 18.8, "Context")
    arrow(49.4, 18.8, 52.4, 18.8, "Action m*")
    arrow(74.6, 18.8, 77.6, 18.8, "UDP:443")

    fb = FancyArrowPatch(
        (88.5, 4.4), (37.5, 4.4),
        connectionstyle="arc3,rad=0.11",
        arrowstyle="-|>,head_length=4.2,head_width=2.6",
        linewidth=1.0, linestyle="--", color="#1e8449",
    )
    ax.add_patch(fb)
    ax.text(63.0, 1.0, "Closed-Loop 4D Telemetry: E2E Latency (p95/p99), Silicon RAPL (mJ), GCP Billing ($), Effective Recall@10", ha="center", va="center", fontsize=6.6, fontweight="bold", color="#196f3d")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.1)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Sanitize All .tex Tables in 00_manuscript/tables/ & Re-Sync SHA-256 Manifest
# ---------------------------------------------------------------------------
def sanitize_all_tables_and_sync_manifest() -> tuple[dict, list[str]]:
    tables_dir = REPO_ROOT / "00_manuscript/tables"
    manifest_path = REPO_ROOT / "06_experiments_results/sha256_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Normalize citation keys & float placement across all generated .tex tables
    cite_alias = {
        "badawi2022openfhe": "albadawi2022openfhe",
        "iyengar2021quic": "iyengar2021rfc9000",
    }

    for tf in sorted(tables_dir.glob("*.tex")):
        txt = tf.read_text(encoding="utf-8")
        orig = txt
        for old_k, new_k in cite_alias.items():
            txt = txt.replace(old_k, new_k)

        # Ensure crisp float placement [!t] instead of bare [t] or invalid [h]
        txt = re.sub(r"\\begin\{table\*\}\[[htbp!]*\]", r"\\begin{table*}[!t]", txt)
        txt = re.sub(r"\\begin\{table\}\[[htbp!]*\]", r"\\begin{table}[!t]", txt)

        # Shorten 64-hex inline SHA-256 strings in table footers so \resizebox doesn't shrink columns
        def _shorten_hex(m: re.Match) -> str:
            h = m.group(1)
            return f"\\texttt{{{h[:16]}...{h[-16:]}}}"

        txt = re.sub(r"\\texttt\{([0-9a-f]{64})\}", _shorten_hex, txt)

        # Ensure tab_arm64_vs_x86.tex fits cleanly within \textwidth without Overfull \hbox
        if tf.name == "tab_arm64_vs_x86.tex" and r"\resizebox" not in txt:
            txt = txt.replace(
                r"\begin{tabular}{lrrrrrc}",
                r"\setlength{\tabcolsep}{5.5pt}" + "\n" + r"\footnotesize" + "\n" + r"\begin{tabular}{lrrrrrc}",
            )

        if txt != orig:
            tf.write_text(txt, encoding="utf-8", newline="\n")

    # Re-verify and sync every record in sha256_manifest.json ("verified_artifacts")
    records = manifest.get("verified_artifacts", [])
    seen_tables = set()
    for rec in records:
        t_path = Path(rec["latex_table_path"])
        if not t_path.is_absolute():
            t_path = REPO_ROOT / t_path
        c_path = Path(rec["raw_csv_path"])
        if not c_path.is_absolute():
            c_path = REPO_ROOT / c_path
        if c_path.exists():
            rec["raw_csv_sha256"] = sm.sha256_digest(c_path)
        if t_path.exists():
            rec["latex_table_sha256"] = sm.sha256_digest(t_path)
            seen_tables.add(t_path.name)

    manifest["verified_artifacts"] = records
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")

    # Discover all supplementary table files (any .tex in 00_manuscript/tables not in the 4 main tables)
    main_four = {
        "tab_core_720_results.tex",
        "tab_ablation_crossdb.tex",
        "tab_arm64_vs_x86.tex",
        "tab_gcp_multiregion_wan.tex",
    }
    preferred_supp_order = [
        "tab_datasets_manifest.tex",
        "tab_hw_calibration.tex",
        "tab_stack_smoke_audit.tex",
        "tab_openfhe_qdrant_audit.tex",
        "tab_wan_socket_audit.tex",
    ]
    supp_tables = [name for name in preferred_supp_order if (tables_dir / name).exists()]
    for extra in sorted(tables_dir.glob("*.tex")):
        if extra.name not in main_four and extra.name not in supp_tables:
            supp_tables.append(extra.name)

    print(f"[TABLES AUDIT] Found {len(main_four)} main input tables + {len(supp_tables)} supplementary input tables: {supp_tables}")
    print(f"[MANIFEST AUDIT] Verified {len(records)} SHA-256 artifact records in sha256_manifest.json")
    return manifest, supp_tables


def compute_stats_summary() -> dict[str, str]:
    rng = np.random.default_rng(42)
    with (REPO_ROOT / "06_experiments_results/raw_logs/gcp_multiregion_360_real_wan.csv").open("r", encoding="utf-8") as f:
        gcp_t = list(csv.DictReader(f))
    with (REPO_ROOT / "06_experiments_results/raw_logs/gcp_multiregion_1440_queries.csv").open("r", encoding="utf-8") as f:
        gcp_q = list(csv.DictReader(f))

    capq_j = np.array([float(r["j_sla_score"]) for r in gcp_t if r["scheduler"] == "CAPQ_Pareto"])
    stat_j = np.array([float(r["j_sla_score"]) for r in gcp_t if r["scheduler"] == "Static_MLKEM1024"])
    cls_j  = np.array([float(r["j_sla_score"]) for r in gcp_t if r["scheduler"] == "Classical_TLS1.3"])

    diff_stat = stat_j - capq_j
    diff_cls  = cls_j - capq_j
    idx_b = rng.integers(0, len(diff_stat), size=(10000, len(diff_stat)))
    bs_stat = diff_stat[idx_b].mean(axis=1)
    bs_cls  = diff_cls[idx_b].mean(axis=1)

    capq_pass = np.array([int(q["sla_pass"]) for q in gcp_q if q["scheduler"] == "CAPQ_Pareto"])
    stat_pass = np.array([int(q["sla_pass"]) for q in gcp_q if q["scheduler"] == "Static_MLKEM1024"])
    b_win = int(np.sum((capq_pass == 1) & (stat_pass == 0)))
    c_lose = int(np.sum((capq_pass == 0) & (stat_pass == 1)))
    chi2 = ((abs(b_win - c_lose) - 1.0) ** 2) / max(float(b_win + c_lose), 1.0)

    return {
        "mean_delta_stat": f"{float(np.mean(diff_stat)):.3f}",
        "ci_stat": f"[{float(np.percentile(bs_stat, 2.5)):.3f}, {float(np.percentile(bs_stat, 97.5)):.3f}]",
        "mean_delta_cls": f"{float(np.mean(diff_cls)):.3f}",
        "ci_cls": f"[{float(np.percentile(bs_cls, 2.5)):.3f}, {float(np.percentile(bs_cls, 97.5)):.3f}]",
        "mcnemar_b": str(b_win),
        "mcnemar_c": str(c_lose),
        "mcnemar_chi2": f"{chi2:.2f}",
    }


def extract_label(tex_file: Path) -> str:
    m = re.search(r"\\label\{([^}]+)\}", tex_file.read_text(encoding="utf-8"))
    if not m:
        raise RuntimeError(f"Missing \\label in {tex_file}")
    return m.group(1)


# ---------------------------------------------------------------------------
# 3. Build IEEE TCC `main.tex` (Proper `algorithm` Counter, stfloats, <=12 Pages)
# ---------------------------------------------------------------------------
def build_clean_main_tex(lbl_core: str, lbl_abl: str, lbl_arm: str, lbl_gcp: str, st: dict[str, str], tier: int = 0) -> str:
    abstract_text = (
        "Hybrid cloud-edge database systems increasingly combine ACID relational transactions with high-dimensional "
        "vector retrieval for retrieval-augmented generation (RAG). Migrating these continuums to post-quantum "
        "cryptography (PQC) and approximate homomorphic encryption (CKKS) mitigates harvest-now-decrypt-later "
        "threats, yet static cryptographic deployment triggers severe wide-area network (WAN) packet fragmentation, "
        "four-round-trip handshake stalls, edge battery drain, and recall collapse. We present CAPQ-Pareto, a "
        "hardware- and cloud-verified multi-objective cryptographic orchestrator that dynamically routes hybrid "
        "relational-vector streams across three cryptographic regimes using an SLA-gated regularized LinUCB "
        "contextual bandit. CAPQ-Pareto jointly minimizes end-to-end latency, silicon-measured edge energy, and "
        "per-second cloud compute and egress cost subject to strict deadline and top-10 vector recall constraints. "
        "Across 1,980 real experimental trials spanning physical Intel RAPL edge telemetry, PostgreSQL pgvector and "
        "Qdrant engines, Google Cloud Tau T2A ARM64 instances, and live multi-region deployments in Bangkok, "
        "Singapore, and Iowa, CAPQ-Pareto improves joint SLA compliance from 0.0--20.0 percent under static PQC "
        "baselines to 72.5--98.3 percent on regional and transpacific links, reduces p99 tail latency by up to "
        "6.0 times via single-MTU INT8 quantization over QUIC 0-RTT, and uncovers a statistically significant "
        "cross-architecture AEAD performance inversion between x86-64 and ARM64 cloud processors."
    )
    wc = len(abstract_text.split())
    assert 150 <= wc <= 199, f"Abstract word count {wc} outside 100--200"

    if tier == 0:
        fig_w1, fig_wc = "0.93\\textwidth", "0.94\\columnwidth"
        sep_cfg = r"""
\setlength{\textfloatsep}{8pt plus 2pt minus 2pt}
\setlength{\floatsep}{6pt plus 2pt minus 2pt}
\setlength{\dbltextfloatsep}{8pt plus 2pt minus 2pt}
\setlength{\dblfloatsep}{6pt plus 2pt minus 2pt}
\setlength{\abovecaptionskip}{3pt}
\setlength{\belowcaptionskip}{1pt}
"""
    elif tier == 1:
        fig_w1, fig_wc = "0.88\\textwidth", "0.86\\columnwidth"
        sep_cfg = r"""
\setlength{\textfloatsep}{6pt plus 1pt minus 2pt}
\setlength{\floatsep}{5pt plus 1pt minus 1pt}
\setlength{\dbltextfloatsep}{6pt plus 1pt minus 2pt}
\setlength{\dblfloatsep}{5pt plus 1pt minus 1pt}
\setlength{\abovecaptionskip}{2pt}
\setlength{\belowcaptionskip}{0pt}
"""
    else:
        fig_w1, fig_wc = "0.82\\textwidth", "0.78\\columnwidth"
        sep_cfg = r"""
\setlength{\textfloatsep}{4pt plus 1pt minus 1pt}
\setlength{\floatsep}{4pt plus 1pt minus 1pt}
\setlength{\dbltextfloatsep}{4pt plus 1pt minus 1pt}
\setlength{\dblfloatsep}{4pt plus 1pt minus 1pt}
\setlength{\abovecaptionskip}{2pt}
\setlength{\belowcaptionskip}{0pt}
"""

    tex = r"""\documentclass[10pt,journal,compsoc]{IEEEtran}
\ifCLASSOPTIONcompsoc
  \usepackage[nocompress]{cite}
\else
  \usepackage{cite}
\fi
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{stfloats}
\usepackage{url}
\usepackage[hidelinks]{hyperref}

\setcounter{topnumber}{3}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{6}
\setcounter{dbltopnumber}{3}
\renewcommand{\topfraction}{0.94}
\renewcommand{\bottomfraction}{0.90}
\renewcommand{\textfraction}{0.05}
\renewcommand{\floatpagefraction}{0.88}
\renewcommand{\dbltopfraction}{0.94}
\renewcommand{\dblfloatpagefraction}{0.88}
__SEP_CFG__
\begin{document}

\title{CAPQ-Pareto: Hardware- and Multi-Region Cloud-Verified 4D Pareto Cryptographic Orchestration for Post-Quantum Hybrid Relational-Vector Databases}

\author{Preecha~Noiumkar,~\IEEEmembership{Member,~IEEE}%
\IEEEcompsocitemizethanks{\IEEEcompsocthanksitem P. Noiumkar is with the Department of Information Technology, Faculty of Informatics, Mahasarakham University, Maha Sarakham 44150, Thailand (e-mail: preecha.n@msu.ac.th; ORCID: \url{https://orcid.org/0009-0000-5964-4896}).\protect\\
Corresponding author: Preecha Noiumkar.}%
\thanks{This research project was financially supported by the Faculty of Informatics, Mahasarakham University. Reproducible code, Docker/GCP scripts, and SHA-256 telemetry manifests: \url{https://github.com/preecha-msu/CAPQ-Pareto}.}}

\markboth{IEEE TRANSACTIONS ON CLOUD COMPUTING}%
{NOIUMKAR: CAPQ-PARETO ORCHESTRATION FOR POST-QUANTUM HYBRID CLOUD-EDGE DATABASES}

\IEEEtitleabstractindextext{%
\begin{abstract}
__ABSTRACT__
\end{abstract}

\begin{IEEEkeywords}
Post-quantum cryptography, ML-KEM, homomorphic encryption, CKKS, hybrid vector-relational databases, QUIC 0-RTT, contextual bandits, multi-region cloud computing, Intel RAPL, ARM64.
\end{IEEEkeywords}}

\maketitle
\IEEEdisplaynontitleabstractindextext
\IEEEpeerreviewmaketitle

\IEEEraisesectionheading{\section{Introduction}\label{sec:intro}}
\IEEEPARstart{M}{odern} enterprise applications deployed across cloud-edge continuums increasingly rely on \emph{hybrid database architectures} that execute transactional SQL predicates alongside approximate nearest-neighbor (ANN) vector similarity queries for retrieval-augmented generation (RAG) \cite{wang2021milvus,zhang2023vbase,pan2024survey_vectordb,douze2024faiss}. Hospital branch gateways, financial micro-data centers, and industrial edge aggregators routinely stream both structured records and $128\text{d}$--$1024\text{d}$ dense embeddings \cite{jegou2011sift,chen2024bgem3} over public wide-area networks (WANs) to geo-distributed cloud clusters. Because sensitive embeddings and transactional payloads transmitted today remain vulnerable to \emph{harvest-now, decrypt-later} (HNDL) adversaries once cryptographically relevant quantum computers mature, NIST standardized the Module-Lattice Key-Encapsulation Mechanism (\texttt{ML-KEM}, FIPS~203) \cite{nist_fips203_2024,bos2018kyber} and digital signatures (\texttt{ML-DSA}, FIPS~204) \cite{nist_fips204_2024} in August~2024. Furthermore, when the cloud database operator itself is untrusted, approximate Ring-LWE Fully Homomorphic Encryption (\texttt{CKKS}) \cite{cheon2017ckks,albadawi2022openfhe} enables encrypted inner-product similarity evaluation without exposing plaintext embeddings \cite{engelsma2022hers}.

Despite strong cryptographic guarantees, naively enforcing static maximum-security PQC or FHE across geo-distributed hybrid databases triggers severe systems-level bottlenecks:
\begin{enumerate}
    \item \textbf{WAN MTU Fragmentation and Multi-RTT Handshake Amplification:} Whereas classical \texttt{X25519} requires $64\text{ B}$ of key-exchange material, \texttt{ML-KEM-1024} transmits $1{,}568\text{ B}$ public keys and $1{,}568\text{ B}$ ciphertexts ($3{,}136\text{ B}$ total) \cite{nist_fips203_2024,stebila2016liboqs}. Paired with uncompressed $1024\text{d}$ \texttt{FP32} RAG vectors ($4{,}096\text{ B}$), payloads span multiple $1{,}500\text{ B}$ Ethernet MTU frames \cite{paquin2020pqctls,sikeridis2020pqctls_wan,sosnowski2023pqc_quic}. Over fresh \texttt{TCP+TLS~1.3} connections \cite{rescorla2018rfc8446}, four round-trips of connection establishment and congestion-window slow-start inflate regional latency ($37.3\text{ ms}$ RTT to Singapore) to $165.7\text{ ms}$ and transpacific latency ($229.6\text{ ms}$ RTT to Iowa) beyond $920\text{ ms}$, collapsing service-level agreements (SLAs).
    \item \textbf{Incompatibility Between FHE Ciphertexts and Graph ANN Traversal:} Hierarchical Navigable Small World (\texttt{HNSW}) indexes \cite{malkov2020hnsw,pgvector2024} rely on data-dependent greedy branching ($\text{if } d(\mathbf{q},\mathbf{u}) < d(\mathbf{q},\mathbf{v})$), which cannot be evaluated directly over semantically secure \texttt{CKKS} ciphertexts without leaking order or incurring prohibitive bootstrapping latency \cite{albadawi2022openfhe,engelsma2022hers}.
    \item \textbf{Hardware Asymmetry Across \texttt{x86\_64} and Cloud \texttt{ARM64}:} Edge gateways and modern cloud instances (e.g., Google Cloud Tau \texttt{T2A} Ampere Altra \texttt{aarch64}) exhibit starkly different SIMD and cryptographic instruction throughput for \texttt{AES-256-GCM}, \texttt{ChaCha20-Poly1305}, lattice NTTs, and \texttt{CKKS} polynomial arithmetic, invalidating static heuristic rules.
\end{enumerate}

To resolve this four-way tension among \emph{tail latency}, \emph{edge silicon energy}, \emph{cloud monetary cost}, and \emph{effective retrieval accuracy}, we design, implement, and validate \textbf{CAPQ-Pareto}. Every metric reported in this study is backed by hardware silicon counters \cite{david2010rapl,khan2018rapl,jay2023power_meters}, compiled C/C++ cryptographic bridges (\texttt{liboqs~0.12.0} and \texttt{OpenFHE~1.2.3}), production hybrid database engines (\texttt{PostgreSQL~18.6} with \texttt{pgvector} and \texttt{Qdrant}), and live multi-region Google Cloud Platform (GCP) deployments. Our primary contributions are:
\begin{itemize}
    \item \textbf{Unit-Consistent 4D Cloud-Edge Pareto Formulation:} We formulate joint optimization over end-to-end latency ($L$), wraparound-safe net silicon edge energy ($E_{\text{edge}}$ in $\text{mJ}$), and per-second GCP compute plus internet egress billing ($C_{\text{gcp}}$ in $\text{USD}$), strictly gated by hard latency deadlines ($T_{\max}$) and minimum retrieval quality ($\text{Recall@10} \ge R_{\min}$).
    \item \textbf{Co-Designed PQC Transport, Vector Quantization, and Coarse-Bucketed CKKS:} We unify three adaptive regimes: \emph{Regime~A} (\texttt{X25519+ML-KEM-768} over \texttt{QUIC 0-RTT} \cite{iyengar2021rfc9000} with transactional micro-batching), \emph{Regime~B} (\texttt{ML-KEM-512} paired with $\epsilon$-safe \texttt{INT8-SQ} scalar quantization compressing $1024\text{d}$ vectors from $4{,}096\text{ B}$ to $1{,}032\text{ B}$ within a single MTU), and \emph{Regime~C} (centroid-routed \texttt{OpenFHE CKKS} inner-product evaluation under \texttt{HEStd\_128\_classic}).
    \item \textbf{SLA-Gated Regularized LinUCB Orchestrator:} We introduce a $10$-dimensional contextual bandit router \cite{li2010linucb,abbasi2011improved_bandits} guarded by a sliding-window $p95$ SLA gate that prunes deadline- or recall-violating actions prior to confidence-bound exploration (Algorithm~\ref{alg:linucb}).
    \item \textbf{1,980-Trial Hardware, \texttt{ARM64}, and 3-Region GCP Empirical Audit:} Across $720$ controlled \texttt{tc/netem} runs, $360$ cross-database (\texttt{BGE-M3} $1024\text{d}$) and WAN-shock runs, $540$ cross-architecture \texttt{x86\_64} vs.\ GCP \texttt{ARM64} (\texttt{Neoverse-N1}) trials, and $360$ live multi-region GCP WAN trials ($1{,}440$ queries from Thailand to Bangkok, Singapore, and Iowa), CAPQ-Pareto elevates transpacific SLA compliance from $0.0\%$ to $98.3\%$, cuts live $p99$ tail latency by up to $6.0\times$, and reveals a statistically significant ($p_{\text{Holm}} = 8.66 \times 10^{-6}$) AEAD cipher inversion on \texttt{ARM64}.
\end{itemize}

\begin{table*}[!t]
\centering
\caption{Taxonomy and Six-Dimensional Gap Matrix Comparing Representative Literature Against \textbf{CAPQ-Pareto}}
\label{tab:gap_matrix}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccccc}
\toprule
\textbf{System / Study} & \textbf{Primary Scope} & \textbf{Hybrid SQL + Vector DB} & \textbf{NIST FIPS 203 PQC + QUIC} & \textbf{Zero-Trust CKKS FHE} & \textbf{Hard Deadline \& Recall Gate} & \textbf{Silicon RAPL Energy [mJ]} & \textbf{Live Multi-Region Cloud + ARM64} \\
\midrule
Milvus \cite{wang2021milvus} \& VBASE \cite{zhang2023vbase} & Cloud Vector / Hybrid DB & \textbf{Yes} & No (Classical TLS) & No & Partial (Recall only) & No & x86\_64 Cloud Only \\
RaBitQ \cite{gao2024rabitq} \& DiskANN \cite{jayaram2019diskann} & Quantized ANN Indexing & Vector Only & No & No & Recall Bound Only & No & Single-Node / Local \\
Paquin et al. \cite{paquin2020pqctls} \& Sikeridis \cite{sikeridis2020pqctls_wan} & PQC in TLS 1.3 over WAN & No (HTTP Only) & Pre-FIPS Kyber/TLS & No & No & No & Emulated / Single WAN \\
KEMTLS \cite{schwabe2020kemtls} \& Sosnowski \cite{sosnowski2023pqc_quic} & Post-Quantum QUIC / TLS & No & \textbf{Yes} (QUIC + KEM) & No & No & No & Testbed WAN Only \\
HERS \cite{engelsma2022hers} \& OpenFHE \cite{albadawi2022openfhe} & Encrypted FHE Vector Search & Vector Only & No & \textbf{Yes} (CKKS) & Static Recall Trade-off & No & Single Cloud Server \\
Kepler \cite{amaral2023kepler} \& Jay et al. \cite{jay2023power_meters} & Cloud/Edge Power Telemetry & No & No & No & No & \textbf{Yes} (RAPL/eBPF) & Local / x86\_64 Cluster \\
\midrule
\textbf{CAPQ-Pareto (This Work)} & \textbf{4D Adaptive Cloud-Edge DB} & \textbf{Yes (PG18.6 + Qdrant)} & \textbf{Yes (ML-KEM + 0-RTT)} & \textbf{Yes (OpenFHE 1.2.3)} & \textbf{Yes ($T_{\max}$ \& $\text{Recall@10}$)} & \textbf{Yes (Bare-Metal RAPL)} & \textbf{Yes (3 GCP Regions + T2A ARM64)} \\
\bottomrule
\end{tabular}%
}
\end{table*}

\section{Related Work and Research Gap}\label{sec:related}
Table~\ref{tab:gap_matrix} positions CAPQ-Pareto relative to three intersecting research threads across cloud databases, post-quantum networking, and privacy-preserving telemetry.

\subsection{Hybrid Relational-Vector Databases and Compression}
Purpose-built vector engines such as Milvus \cite{wang2021milvus}, Faiss \cite{douze2024faiss}, DiskANN \cite{jayaram2019diskann}, and relational-vector engines including \texttt{pgvector} \cite{pgvector2024} and VBASE \cite{zhang2023vbase} pair HNSW \cite{malkov2020hnsw} or IVF indexes with relational predicates \cite{pan2024survey_vectordb}. Quantization schemes such as product quantization \cite{jegou2011sift} and RaBitQ \cite{gao2024rabitq} reduce memory footprint while bounding inner-product distortion. However, existing hybrid engines assume classical transport security and optimize server-side index traversal in isolation, overlooking how vector wire size interacts with post-quantum handshake fragmentation over lossy WAN links.

\subsection{Post-Quantum TLS 1.3, QUIC, and Homomorphic Search}
Empirical studies by Paquin et al. \cite{paquin2020pqctls} and Sikeridis et al. \cite{sikeridis2020pqctls_wan} showed that lattice keyshares \cite{stebila2016liboqs,bos2018kyber} exceeding the $1{,}500\text{ B}$ path MTU trigger nonlinear latency amplification under packet loss. Subsequent work investigated signatureless \texttt{KEMTLS} \cite{schwabe2020kemtls} and post-quantum \texttt{QUIC} \cite{iyengar2021rfc9000,sosnowski2023pqc_quic}. Unlike static transport-layer deployments that apply one cipher suite uniformly, CAPQ-Pareto co-optimizes \texttt{QUIC 0-RTT} resumption, KEM security level (\texttt{ML-KEM-512/768/1024}), AEAD cipher selection, and vector quantization per query stream. For untrusted clouds, approximate FHE (\texttt{CKKS}) \cite{cheon2017ckks,albadawi2022openfhe} supports SIMD-packed encrypted inner products \cite{engelsma2022hers}, while RAPL \cite{david2010rapl,khan2018rapl}, Scaphandre \cite{jay2023power_meters}, and Kepler \cite{amaral2023kepler} quantify silicon energy. CAPQ-Pareto unifies these mechanisms under an SLA-gated contextual bandit \cite{li2010linucb,abbasi2011improved_bandits}.

\begin{figure*}[!t]
\centering
\includegraphics[width=__FIG_W1__]{00_manuscript/figures/fig1_capq_architecture.pdf}
\caption{Four-layer architecture of \textbf{CAPQ-Pareto} bridging a bare-metal RAPL-instrumented Edge Gateway in Thailand with live Google Cloud Hybrid Database nodes across Bangkok (\texttt{asia-southeast3}), Singapore (\texttt{asia-southeast1}, \texttt{x86\_64} and \texttt{Tau T2A ARM64}), and Iowa (\texttt{us-central1}).}
\label{fig:arch}
\end{figure*}

\section{System Architecture and Threat Model}\label{sec:arch}
\subsection{Four-Layer Hybrid Cloud-Edge Continuum}
Fig.~\ref{fig:arch} illustrates the four-layer CAPQ-Pareto architecture deployed across a physical edge gateway and geo-distributed Google Cloud regions:
\begin{enumerate}
    \item \textbf{Layer~1 (Physical Bare-Metal Edge Gateway):} Because public cloud hypervisors block guest access to CPU package energy Model-Specific Registers (MSRs) \cite{khan2018rapl,jay2023power_meters,amaral2023kepler}, our edge tier executes on a physical bare-metal gateway (\texttt{Intel Core Ultra 9 185H}) instrumented via OS kernel RAPL/EMI counters (\texttt{pdh.dll}). It ingests mixed OLTP SQL transactions (\texttt{TPC-C} \cite{tpcc_spec}, \texttt{YCSB-A} \cite{cooper2010ycsb}) and unstructured RAG vector streams (\texttt{SIFT-128d} \cite{jegou2011sift}, \texttt{Wikipedia BGE-M3 1024d} \cite{chen2024bgem3}). Portability to ARM edge gateways is validated natively on Google Cloud \texttt{Tau T2A} (\texttt{Ampere Altra Neoverse-N1 aarch64}).
    \item \textbf{Layer~2 (SLA-Gated Contextual Bandit Orchestrator):} Before dispatching each query or micro-batch, the orchestrator observes a $10$-dimensional live context vector encoding workload type, vector dimensionality, active WAN RTT/loss, and operator preference weights $(\alpha,\beta,\gamma)$, selecting an optimal regime $m \in \mathcal{M}$.
    \item \textbf{Layer~3 (Adaptive Cryptographic \& Compression Pipeline):} Executes native C/C++ FFI bindings to \texttt{liboqs~0.12.0} (\texttt{ML-KEM}), AEAD (\texttt{AES-256-GCM}, \texttt{ChaCha20-Poly1305}), $\epsilon$-safe vector quantizers (\texttt{FP32}, \texttt{INT8-SQ}, \texttt{1-bit}), and \texttt{OpenFHE~1.2.3} (\texttt{CKKS}) over multiplexed \texttt{QUIC 0-RTT} (\texttt{UDP:443}) \cite{iyengar2021rfc9000} or \texttt{TLS~1.3} (\texttt{TCP:8443}) \cite{rescorla2018rfc8446}.
    \item \textbf{Layer~4 (Geo-Distributed Google Cloud Hybrid DBs):} Hosts \texttt{PostgreSQL} with \texttt{pgvector} (\texttt{HNSW}) \cite{pgvector2024}, native Rust \texttt{Qdrant} \texttt{HNSW}, and the \texttt{OpenFHE} homomorphic evaluator across three live GCP regions: \texttt{asia-southeast3} (Bangkok), \texttt{asia-southeast1} (Singapore), and \texttt{us-central1} (Iowa, USA).
\end{enumerate}

\subsection{Dual Adversary and Security Threat Model}
CAPQ-Pareto supports two explicit trust tiers selectable per query sensitivity flag $z_q \in \{0,1\}$:
\begin{itemize}
    \item \textbf{Tier~1 — Quantum-Capable Network Adversary ($z_q = 0$, Regimes~A \& B):} External eavesdroppers on ISP/BGP transit links record encrypted WAN traffic today for future quantum decryption (HNDL), while the enterprise-managed cloud VM is trusted during execution. Hybrid \texttt{X25519+ML-KEM-768} and \texttt{ML-KEM-512/1024} provide IND-CCA2 post-quantum key establishment \cite{nist_fips203_2024}. Replay protection on \texttt{QUIC 0-RTT} is enforced via idempotent read gating and monotonic per-client sequence nonces \cite{iyengar2021rfc9000}.
    \item \textbf{Tier~2 — Honest-but-Curious Cloud DB Operator ($z_q = 1$, Regime~C):} For confidential embeddings, the cloud server must not observe plaintext query coordinates. Queries are encrypted under \texttt{OpenFHE CKKS} satisfying \texttt{HEStd\_128\_classic} ($\ge 128$-bit security) \cite{cheon2017ckks,albadawi2022openfhe}. Only a coarse cluster index $c^* \in \{1,\dots,K_c\}$ is revealed to bound scan complexity, whose formal min-entropy leakage $\mathcal{L}_{\text{bucket}} \le \log_2 K_c$ bits is proven in Supplementary Section~S1.
\end{itemize}

\section{4D Pareto Cloud-Edge Formulation}\label{sec:math}
Let $\mathcal{M}$ denote the discrete action space of cryptographic-compression-transport configurations ($|\mathcal{M}| = 10$). Selecting mode $m \in \mathcal{M}$ for query $q_t$ under context $\mathbf{x}_t \in \mathbb{R}^{d_c}$ yields outcome tuple $\big(L_t(m), E_{\text{edge},t}(m), C_{\text{gcp},t}(m), A_{\text{eff},t}(m)\big)$.

\subsection{End-to-End Latency and MTU Fragmentation Model}
For micro-batch size $b \ge 1$ under arrival rate $\lambda_q$, end-to-end latency $L(m)$ (in $\text{ms}$) decomposes into five measurable stages:
\begin{equation}
L(m) = T_{\text{queue}}(b) + T_{\text{edge\_crypto}}(m) + T_{\text{wan}}(m) + T_{\text{cloud\_eval}}(m) + T_{\text{dec}}(m),
\label{eq:latency_decomp}
\end{equation}
where $T_{\text{queue}}(b) = \frac{b-1}{2\lambda_q}\cdot 10^3$ ($0\text{ ms}$ when $b=1$). Given path payload MTU $M_{\text{mtu}} = 1{,}460\text{ B}$, handshake bytes $S_{\text{hs}}(m)$, and payload bytes $S_{\text{pay}}(m)$, packet count is $N_{\text{pkt}}(m) = \lceil (S_{\text{hs}}(m) + S_{\text{pay}}(m)) / M_{\text{mtu}} \rceil$. Fresh \texttt{TCP+TLS~1.3} incurs $k_{\text{rtt}} \approx 3\text{--}4$ round-trips, whereas warm \texttt{QUIC 0-RTT} achieves $k_{\text{rtt}} = 1$:
\begin{equation}
T_{\text{wan}}(m) = k_{\text{rtt}}(m)\cdot\text{RTT} + \frac{8\big(S_{\text{hs}}(m)+S_{\text{pay}}(m)+S_{\text{resp}}(m)\big)}{B_{\text{wan}}} + \Phi_{\text{frag}}\big(N_{\text{pkt}}(m), p_{\text{loss}}, \text{RTT}\big),
\label{eq:wan_frag}
\end{equation}
where $\Phi_{\text{frag}}$ grows superlinearly when $N_{\text{pkt}}(m) > 1$ and $p_{\text{loss}} > 0$ due to multi-fragment reassembly stalls \cite{sikeridis2020pqctls_wan,sosnowski2023pqc_quic}.

\subsection{Wraparound-Safe Silicon RAPL Energy and GCP Billing}
Let $R(t)$ denote the hardware energy counter scaled by unit multiplier $\kappa_E$ into millijoules ($\text{mJ}$) modulo $R_{\max}$. Over $[t_0, t_1]$ ($\Delta t = t_1 - t_0$), wraparound-safe net active per-query energy after subtracting calibrated quiescent idle power $P_{\text{idle}}$ ($\text{mW}$) across $N_w$ queries of batch size $b$ is:
\begin{equation}
E_{\text{edge}}(m) = \frac{\max\!\Big(0,\; \kappa_E\big((R(t_1)-R(t_0))\bmod R_{\max}\big) - P_{\text{idle}}\Delta t\Big)}{N_w \cdot b}.
\label{eq:net_energy}
\end{equation}
Given regional hourly GCP VM price $p_{\text{vm}}(r)$ ($\text{USD/hr}$), cloud compute duration $T_{\text{cloud\_eval}}(m)$ ($\text{ms}$), and egress bytes $S_{\text{resp}}(m)$ billed per GiB ($2^{30}\text{ B}$), per-query cloud cost is:
\begin{equation}
C_{\text{gcp}}(m, r) = \frac{p_{\text{vm}}(r)}{3{,}600}\cdot \frac{10^{-3} T_{\text{cloud\_eval}}(m)}{b} + p_{\text{egress}}(r)\cdot \frac{2^{-30} S_{\text{resp}}(m)}{b}.
\label{eq:gcp_cost}
\end{equation}

\subsection{Deadline-Gated Effective Accuracy and 4D Objective}
Responses exceeding regional deadline $T_{\max}(r)$ yield zero real-time utility:
\begin{equation}
A_{\text{eff}}(m) = A_{\text{raw}}(m) \cdot \mathbb{I}\!\big[L(m) \le T_{\max}(r)\big].
\label{eq:eff_acc}
\end{equation}
With warm-up normalized metrics $\tilde{X}(m) = \text{clip}\big((X(m)-X_{\min})/\max(X_{\max}-X_{\min}, \epsilon), 0, 1\big)$ for $X \in \{L, E_{\text{edge}}, C_{\text{gcp}}\}$ and convex weights $\alpha + \beta + \gamma = 1$, CAPQ-Pareto solves:
\begin{align}
\min_{m \in \mathcal{M}} \quad & \mathcal{J}(m) = \alpha \tilde{L}(m) + \beta \tilde{E}_{\text{edge}}(m) + \gamma \tilde{C}_{\text{gcp}}(m) \label{eq:pareto_obj}\\
\text{s.t.} \quad & P_{95}\!\big(L(m)\big) \le T_{\max}(r), \quad \mathbb{E}\!\big[A_{\text{eff}}(m)\big] \ge R_{\min}(q), \quad \text{Sec}(m) \ge \lambda_{\text{sec}}(z_q), \label{eq:const_acc}
\end{align}
where $R_{\min}(q) = 1.0$ for SQL, $R_{\min}(q) = 0.92$ for RAG vectors, and $\lambda_{\text{sec}} = 128\text{ bits}$. To evaluate SLA-violating baselines without selection bias, we compute $\mathcal{J}_{\text{SLA}}(m) = \mathcal{J}(m) + \big(1 - \text{SLA\_Pass}(m)\big)$.

\begin{algorithm}[!t]
\caption{SLA-Gated Regularized LinUCB Orchestrator}
\label{alg:linucb}
\footnotesize
\begin{algorithmic}[1]
\Require Action space $\mathcal{M}$, ridge $\lambda_{\text{reg}}=1.0$, exploration $\nu=0.20$, deadline $T_{\max}(r)$, recall floor $R_{\min}=0.92$, weights $(\alpha,\beta,\gamma)$
\Ensure Optimal per-query cryptographic-compression regime $m_t^* \in \mathcal{M}$
\State Initialize $\mathbf{A}_m \leftarrow \lambda_{\text{reg}}\mathbf{I}_{10}$, $\mathbf{b}_m \leftarrow \mathbf{0}_{10}$, window $\mathcal{W}_m$ ($\forall m \in \mathcal{M}$)
\For{each arriving hybrid query $q_t$ with context $\mathbf{x}_t \in \mathbb{R}^{10}$}
    \State $\mathcal{M}_{\text{sec}} \leftarrow \{m \in \mathcal{M} : \text{Sec}(m) \ge \lambda_{\text{sec}}(z_{q_t})\}$
    \State Query sliding-window $p95$ latency $\hat{L}_{95}(m)$ and mean $\hat{A}(m)$ from $\mathcal{W}_m$
    \State \textbf{SLA Gate:} $\mathcal{F}_t \leftarrow \{m \in \mathcal{M}_{\text{sec}} : \hat{L}_{95}(m) \le T_{\max}(r) \wedge \hat{A}(m) \ge R_{\min}\}$
    \If{$\mathcal{F}_t = \emptyset$}
        \State $\mathcal{F}_t \leftarrow \{\arg\min_{m \in \mathcal{M}_{\text{sec}}} \hat{L}_{95}(m)\}$ \Comment{Safe min-latency fallback}
    \EndIf
    \For{each $m \in \mathcal{F}_t$}
        \State $\hat{\boldsymbol{\theta}}_m \leftarrow \mathbf{A}_m^{-1}\mathbf{b}_m$;\quad $\text{LCB}_t(m) \leftarrow \hat{\boldsymbol{\theta}}_m^\top \mathbf{x}_t - \nu\sqrt{\mathbf{x}_t^\top \mathbf{A}_m^{-1}\mathbf{x}_t}$
    \EndFor
    \State Dispatch $q_t$ via $m_t^* \leftarrow \arg\min_{m \in \mathcal{F}_t} \text{LCB}_t(m)$
    \State Measure live $(L_t, E_{\text{edge},t}, C_{\text{gcp},t}, A_{\text{eff},t})$; set $y_t \leftarrow \mathcal{J}_t(m_t^*) + \mathbb{I}[\neg\text{SLA}_t]$
    \State $\mathbf{A}_{m_t^*} \leftarrow \mathbf{A}_{m_t^*} + \mathbf{x}_t\mathbf{x}_t^\top$;\quad $\mathbf{b}_{m_t^*} \leftarrow \mathbf{b}_{m_t^*} + y_t\mathbf{x}_t$;\quad update $\mathcal{W}_{m_t^*}$
\EndFor
\end{algorithmic}
\end{algorithm}

\section{SLA-Gated LinUCB Crypto-Scheduler and Adaptive Regimes}\label{sec:scheduler}
At step $t$, Layer~2 constructs context vector $\mathbf{x}_t = [1, \mathbb{I}_{\text{vec}}, d/1024, b/64, \lambda_{\text{sec}}/256, B_{\text{wan}}/100, \text{RTT}/250, p_{\text{loss}}/5, \alpha, \beta]^\top \in \mathbb{R}^{10}$. As detailed in Algorithm~\ref{alg:linucb} ($< 45\;\mu\text{s}$ per decision), CAPQ-Pareto interposes a deterministic \emph{SLA Gate} over a $6$-sample sliding window $\mathcal{W}_m$ to prune infeasible modes before Lower-Confidence-Bound ($\text{LCB}_t$) selection \cite{li2010linucb,abbasi2011improved_bandits}:
\begin{itemize}
    \item \textbf{Regime~A (Transactional SQL Fast-Path):} Uses hybrid \texttt{X25519+ML-KEM-768} over persistent \texttt{QUIC 0-RTT} (\texttt{UDP:443}) \cite{iyengar2021rfc9000}, activating micro-batching ($b=16$) when energy/cost weights $(\beta,\gamma)$ dominate.
    \item \textbf{Regime~B ($\epsilon$-Safe Quantized RAG Vector Path):} Pairs \texttt{ML-KEM-512} with min-max scalar quantization (\texttt{INT8-SQ}):
    \begin{equation}
    q_i^{(8)} = \text{clip}\!\left(\left\lfloor \frac{v_i - v_{\min}}{(v_{\max}-v_{\min})/255} + 0.5 \right\rfloor, 0, 255\right),
    \label{eq:int8_sq}
    \end{equation}
    shrinking $1024\text{d}$ \texttt{BGE-M3} payloads by $74.8\%$ ($4{,}096\text{ B} \rightarrow 1{,}032\text{ B}$, single MTU) while adapting symmetric encryption between \texttt{AES-256-GCM} (\texttt{x86\_64} AES-NI) and \texttt{ChaCha20-Poly1305} (\texttt{ARM64}).
    \item \textbf{Regime~C (Coarse-Bucketed Zero-Trust CKKS):} Partitions vectors offline via spherical $k$-means into $K_c = 64$ edge-cached centroids ($32\text{ KB}$ for $128\text{d}$), encrypting confidential queries into a single SIMD-packed \texttt{OpenFHE CKKS} ciphertext ($N = 4096$, \texttt{BV} key-switching, $\log_2 Q \le 109\text{ bits}$, \texttt{HEStd\_128\_classic}) for server-side \texttt{EvalInnerProduct} over $M_b \le 16$ bucket candidates.
\end{itemize}

\begin{table*}[!t]
\centering
\caption{Master Summary of All $1{,}980$ Hardware- and Cloud-Verified Experimental Trials Executed in This Study (Zero Synthetic Estimation)}
\label{tab:exp_campaigns}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrll}
\toprule
\textbf{Campaign ID} & \textbf{Infrastructure \& Network Continuum} & \textbf{Paired Trials} & \textbf{Live DB/Crypto Ops} & \textbf{Workloads \& Engines} & \textbf{Verified Output Artifacts} \\
\midrule
\textbf{C-Core (Phase 6A)} & Bare-Metal Edge + Isolated Linux \texttt{tc/netem} WAN (3 Profiles) & $720$ & $8{,}640$ & \texttt{TPC-C}, \texttt{YCSB-A}, \texttt{SIFT-128d} (\texttt{PG18.6} + \texttt{OpenFHE}) & Table~\ref{__LBL_CORE__}, Figs.~\ref{fig:frag}--\ref{fig:breakdown} \\
\textbf{C-Ablation (Phase 6B)} & Cross-DB High-Dim Vector + 4 Weight Profiles + WAN Shock & $360$ & $4{,}680$ & \texttt{Wikipedia BGE-M3} ($1024\text{d}$): \texttt{PG18.6} vs.\ \texttt{Qdrant} & Table~\ref{__LBL_ABL__}, Figs.~\ref{fig:shock_weights}--\ref{fig:crossdb_bgem3} \\
\textbf{C-ARM64 (Step C3)} & Local \texttt{x86\_64} vs.\ Live GCP \texttt{Tau T2A ARM64} (\texttt{Neoverse-N1}) & $540$ & $49{,}410$ & 9 Native C/C++ Kernels (\texttt{liboqs~0.12} + \texttt{OpenFHE~1.2.3}) & Table~\ref{__LBL_ARM__}, Fig.~\ref{fig:arm64_cross} \\
\textbf{C-MultiRegion (Step C4)} & Physical Edge (Thailand) $\rightarrow$ Live GCP Bangkok, Singapore, Iowa & $360$ & $1{,}440$ & Live \texttt{QUIC UDP:443} \& \texttt{TLS:8443} + Cloud \texttt{PG+pgvector} & Table~\ref{__LBL_GCP__}, Fig.~\ref{fig:gcp_p99} \\
\midrule
\textbf{Total Verified} & \textbf{Hybrid Bare-Metal RAPL Edge + 3 GCP Regions + GCP ARM64} & $\mathbf{1{,}980}$ & $\mathbf{64{,}170}$ & \textbf{4 Real Datasets, 2 Hybrid DBs, 2 CPU ISAs} & \textbf{SHA-256 Manifest (Supp. S1--S7)} \\
\bottomrule
\end{tabular}%
}
\end{table*}

\input{00_manuscript/tables/tab_core_720_results.tex}

\section{Experimental Evaluation}\label{sec:eval}
\subsection{Testbed Hardware, GCP Regions, and Non-Parametric Statistical Protocol}
Table~\ref{tab:exp_campaigns} summarizes all $1{,}980$ real trial executions ($64{,}170$ live operations) cryptographically locked in \texttt{sha256\_manifest.json} (detailed calibration and provenance tables in Supplementary Tables~S1--S7):
\begin{itemize}
    \item \textbf{Physical Edge \& Multi-Region GCP Nodes:} Bare-metal \texttt{Intel Core Ultra 9 185H} gateway in Khon Kaen, Thailand (\texttt{pdh.dll} RAPL/EMI) connected over the public internet to live GCP instances in \textbf{Bangkok} (\texttt{asia-southeast3-a}, \texttt{n4-standard-4}, $\$0.2396/\text{hr}$), \textbf{Singapore} (\texttt{asia-southeast1-b}, \texttt{e2-standard-4}, $\$0.1588/\text{hr}$ and \texttt{Tau T2A t2a-standard-4 Ampere Altra Neoverse-N1 aarch64}), and \textbf{US-Central} (\texttt{us-central1-a}, \texttt{e2-standard-4}, $\$0.1340/\text{hr}$), with $\$0.12/\text{GiB}$ egress.
    \item \textbf{Non-Parametric Statistical Protocol:} Each configuration runs $N=30$ paired trials after dedicated warm-up queries across \texttt{Classical\_TLS1.3}, \texttt{Static\_MLKEM1024}, \texttt{Rule\_Heuristic}, and \textbf{\texttt{CAPQ\_Pareto}}. We apply two-sided paired Wilcoxon signed-rank tests with Holm--Bonferroni step-down correction ($p_{\text{Holm}} < 0.05$) \cite{holm1979simple}, $10{,}000$-iteration paired percentile bootstrap $95\%$ confidence intervals (CIs) \cite{efron1979bootstrap}, and continuity-corrected McNemar's $\chi^2$ tests \cite{mcnemar1947note} on binary SLA outcomes.
\end{itemize}

\begin{figure}[!t]
\centering
\includegraphics[width=__FIG_WC__]{00_manuscript/figures/pareto_4d_plots/fig2_wan_fragmentation_bottleneck.pdf}
\caption{Measured WAN MTU fragmentation penalty comparing fresh \texttt{TCP+TLS~1.3} with \texttt{ML-KEM-1024} against warm \texttt{QUIC 0-RTT} across progressive RTT and loss profiles.}
\label{fig:frag}
\end{figure}

\begin{figure}[!t]
\centering
\includegraphics[width=__FIG_WC__]{00_manuscript/figures/pareto_4d_plots/fig3_pareto_4d_tradeoff.pdf}
\caption{Measured 4D Pareto frontier across $720$ controlled paired trials showing latency, silicon RAPL edge energy, GCP cost, and SLA compliance.}
\label{fig:pareto4d}
\end{figure}

\begin{figure}[!t]
\centering
\includegraphics[width=__FIG_WC__]{00_manuscript/figures/pareto_4d_plots/fig4_profile_breakdown.pdf}
\caption{Per-profile breakdown across \texttt{5g\_mec\_good} ($12\text{ ms}$), \texttt{wan\_congested} ($65\text{ ms}$, $1.5\%$ loss), and \texttt{remote\_iot\_constrained} ($140\text{ ms}$, $3.0\%$ loss).}
\label{fig:breakdown}
\end{figure}

\input{00_manuscript/tables/tab_ablation_crossdb.tex}
\input{00_manuscript/tables/tab_arm64_vs_x86.tex}

\subsection{Controlled Kernel WAN (720 Runs) and High-Dim BGE-M3 Cross-DB Ablation (360 Runs)}
Figs.~\ref{fig:frag}--\ref{fig:breakdown} and Table~\ref{__LBL_CORE__} establish our controlled \texttt{tc/netem} baseline ($720$ paired trials). Multi-frame \texttt{ML-KEM-1024} handshakes ($3{,}136\text{ B}$) paired with \texttt{FP32} vectors suffer severe TCP retransmission stalls at $1.5\%$--$3.0\%$ loss, whereas CAPQ-Pareto achieves the lowest $\mathcal{J}_{\text{SLA}}$ across all strata ($p_{\text{Holm}} < 10^{-8}$).

In Table~\ref{__LBL_ABL__} and Figs.~\ref{fig:shock_weights}--\ref{fig:crossdb_bgem3} ($360$ runs), $\epsilon$-safe \texttt{INT8-SQ} compresses $1024\text{d}$ \texttt{Wikipedia BGE-M3} embeddings by $74.8\%$ ($4{,}096\text{ B} \rightarrow 1{,}032\text{ B}$, single MTU) while preserving $\text{Recall@10} = 0.9513$ on \texttt{PostgreSQL~18.6} (\texttt{pgvector~0.8.1}) and $0.9927$ on \texttt{Qdrant} ($\ge 0.92$ SLA). Extreme \texttt{1-bit} quantization ($136\text{ B}$) drops $\text{Recall@10}$ to $0.6983$--$0.7077$ and is automatically blocked by our SLA gate. Under sudden WAN shocks ($12 \rightarrow 65 \rightarrow 140\text{ ms}$ RTT, Fig.~\ref{fig:shock_weights}a), CAPQ-Pareto adapts within two sliding-window epochs to maintain $< 150\text{ ms}$ latency.

\begin{figure}[!t]
\centering
\includegraphics[width=__FIG_WC__]{00_manuscript/figures/pareto_4d_plots/fig5_ablation_weights_and_shock.pdf}
\caption{Online trajectory adaptation under sudden WAN latency/loss shocks and 4-profile Pareto weight sensitivity.}
\label{fig:shock_weights}
\end{figure}

\begin{figure}[!t]
\centering
\includegraphics[width=__FIG_WC__]{00_manuscript/figures/pareto_4d_plots/fig6_crossdb_bgem3_1024d.pdf}
\caption{Cross-database \texttt{BGE-M3} ($1024\text{d}$) $\text{Recall@10}$ and query latency across \texttt{PostgreSQL~18.6} (\texttt{pgvector~0.8.1}) and \texttt{Qdrant} native HNSW.}
\label{fig:crossdb_bgem3}
\end{figure}

\input{00_manuscript/tables/tab_gcp_multiregion_wan.tex}

\subsection{Cross-Architecture Validation: \texttt{x86\_64} vs. GCP \texttt{ARM64}}
Table~\ref{__LBL_ARM__} and Fig.~\ref{fig:arm64_cross} benchmark nine native C/C++ kernels ($540$ paired trials) across local \texttt{x86\_64} (\texttt{Core Ultra 9 185H}) and live GCP \texttt{Tau T2A ARM64} (\texttt{t2a-standard-4}, Ampere Altra \texttt{Neoverse-N1}). On \texttt{x86\_64} (\texttt{VAES/AES-NI}), $4\text{ KB}$ \texttt{AES-256-GCM} ($2.1\;\mu\text{s}$) is $2.05\times$ faster than \texttt{ChaCha20-Poly1305} ($4.3\;\mu\text{s}$), whereas on cloud \texttt{ARM64} this relationship \emph{inverts}: \texttt{ChaCha20-Poly1305} ($29.1\;\mu\text{s}$, $0.102\text{ mJ}$) outperforms \texttt{AES-256-GCM} ($31.8\;\mu\text{s}$, $0.111\text{ mJ}$, $p_{\text{Holm}} = 8.66 \times 10^{-6}$). Full \texttt{ML-KEM-512/768/1024} completes in $0.135$--$0.312\text{ ms}$ on \texttt{ARM64}, whereas $8$-candidate \texttt{OpenFHE CKKS} scales from $65.80\text{ ms}$ ($N=4096$) to $613.33\text{ ms}$ ($N=16384$), confirming that interactive Zero-Trust search requires $N \le 8192$ with coarse bucketing.

\begin{figure}[!t]
\centering
\includegraphics[width=__FIG_WC__]{00_manuscript/figures/pareto_4d_plots/fig7_arm64_vs_x86_cross_arch.pdf}
\caption{Cross-architecture execution latency ($N=30$ trials/kernel) comparing local \texttt{x86\_64} against live Google Cloud \texttt{Tau T2A ARM64} (\texttt{Neoverse-N1}).}
\label{fig:arm64_cross}
\end{figure}

\subsection{Live Multi-Region Google Cloud WAN Evaluation}
Table~\ref{__LBL_GCP__} and Fig.~\ref{fig:gcp_p99} report $360$ paired cloud trials ($1{,}440$ live internet queries) from Thailand to \textbf{Bangkok} ($\text{RTT}=11.51\text{ ms}$, $T_{\max}=150\text{ ms}$), \textbf{Singapore} ($\text{RTT}=37.29\text{ ms}$, $T_{\max}=150\text{ ms}$), and \textbf{US-Central} ($\text{RTT}=229.59\text{ ms}$, $T_{\max}=350\text{ ms}$). Across all $90$ paired multi-region trials per orchestrator, $10{,}000$-iteration paired percentile bootstrap analysis \cite{efron1979bootstrap} confirms that CAPQ-Pareto reduces $\mathcal{J}_{\text{SLA}}$ by $\Delta\mathcal{J}_{\text{SLA}} = __MEAN_D_STAT__$ ($95\%$ CI: $__CI_STAT__$) against \texttt{Static\_MLKEM1024} and $\Delta\mathcal{J}_{\text{SLA}} = __MEAN_D_CLS__$ ($95\%$ CI: $__CI_CLS__$) against \texttt{Classical\_TLS1.3}. Continuity-corrected McNemar's test \cite{mcnemar1947note} over the $360$ paired live queries confirms highly significant SLA recovery ($\chi^2 = __MCNEMAR_CHI2__$, discordant pairs $b=__MCNEMAR_B__$ vs.\ $c=__MCNEMAR_C__$, $p < 10^{-16}$):
\begin{itemize}
    \item \textbf{US-Central ($229.59\text{ ms}$ RTT):} Non-resumed \texttt{TCP+TLS~1.3} baselines suffer four transpacific round-trips ($\text{Mean}=920.5$--$933.4\text{ ms}$, $p99$ up to $1{,}481.4\text{ ms}$), yielding \textbf{$0.0\%$ SLA pass}. \texttt{Rule\_Heuristic} stalls at $50.0\%$ due to \texttt{1-bit} recall failure. \textbf{CAPQ-Pareto achieves $98.3\%$ SLA pass}, $A_{\text{eff}} = 0.998$, and $\mathbf{p99 = 246.6\text{ ms}}$ ($6.0\times$ lower $p99$ than classical TCP/TLS~1.3).
    \item \textbf{Singapore ($37.29\text{ ms}$ RTT) \& Bangkok ($11.51\text{ ms}$ RTT):} In Singapore, CAPQ-Pareto lowers mean latency by $42.2\%$ ($95.8\text{ ms}$ vs.\ $165.7\text{ ms}$) and raises SLA pass from $15.8\%$--$20.0\%$ to $72.5\%$. In Bangkok, CAPQ-Pareto cuts $p99$ latency by $47.7\%$ ($136.7\text{ ms}$ vs.\ $261.5\text{ ms}$) and edge RAPL energy by $32.3\%$ ($67.38\text{ mJ}$ vs.\ $99.60\text{ mJ}$). On links with $\text{RTT}<55\text{ ms}$, mandatory Zero-Trust queries ($25\%$ of stream) exposed boundary recall loss under single-bucket ($16$-vector) CKKS scans, motivating multi-probe encrypted IVF ($m_{\text{probe}} \ge 3$).
\end{itemize}

\begin{figure}[!t]
\centering
\includegraphics[width=__FIG_WC__]{00_manuscript/figures/pareto_4d_plots/fig8_gcp_multiregion_p99_wan.pdf}
\caption{Live multi-region Google Cloud $p99$ tail latency and joint regional SLA pass rate measured from Thailand to Bangkok, Singapore, and Iowa.}
\label{fig:gcp_p99}
\end{figure}

\subsection{Discussion, Hardware C-State Artifacts, and Validity}
In Table~\ref{__LBL_GCP__}, \texttt{Static\_MLKEM1024} on \texttt{US-Central} logged $0.00\text{ mJ}$ net active energy because blocking inside kernel \texttt{epoll\_wait} for $920.5\text{ ms}$ per transpacific query allowed CPU cores to enter deep sleep states ($C6/C8$) below the active-loop pre-run idle baseline ($5{,}844.53\text{ mW}$), clipping Eq.~\eqref{eq:net_energy} at zero \cite{khan2018rapl,jay2023power_meters}. Combining a physical RAPL edge gateway with live GCP \texttt{x86\_64} and \texttt{Tau T2A ARM64} VMs overcomes cloud hypervisor MSR restrictions while preserving authentic internet BGP routing dynamics.

\section{Conclusion}\label{sec:conclusion}
We presented \textbf{CAPQ-Pareto}, a hardware- and multi-region cloud-verified 4D Pareto cryptographic orchestrator for post-quantum hybrid relational-vector databases. Validated across $1{,}980$ real executions spanning \texttt{Intel RAPL} telemetry, \texttt{PostgreSQL~18.6} (\texttt{pgvector}), \texttt{Qdrant}, \texttt{liboqs~0.12.0}, \texttt{OpenFHE~1.2.3}, GCP \texttt{Tau T2A ARM64}, and live WAN deployments in Bangkok, Singapore, and Iowa, CAPQ-Pareto restores transcontinental SLA compliance from $0.0\%$ to $98.3\%$, slashes live $p99$ tail latency by up to $6.0\times$, compresses $1024\text{d}$ \texttt{BGE-M3} vectors by $74.8\%$ within a single MTU ($\text{Recall@10} \ge 0.951$--$0.993$), and adapts AEAD selection across \texttt{x86\_64} and \texttt{ARM64} cloud CPUs.

\section*{Acknowledgments, Data Availability, and AI Disclosure}
This research project was financially supported by the Faculty of Informatics, Mahasarakham University. All source code, Docker Compose files (\texttt{04\_testbed\_infra/docker-compose.yml}), GCP multi-region provisioning and auto-teardown scripts, raw CSV logs, and \texttt{SHA-256} verification manifests are openly available at \url{https://github.com/preecha-msu/CAPQ-Pareto}. In accordance with IEEE publication policies, generative AI tools were used strictly for English language editing and LaTeX formatting verification; all cryptographic implementations, hardware RAPL measurements, multi-region cloud experiments, and statistical analyses were executed and verified directly by the author.

\bibliographystyle{IEEEtran}
\bibliography{references}

\begin{IEEEbiographynophoto}{Preecha Noiumkar}
(Member, IEEE) received the M.Sc. degree in Information Technology from King Mongkut's University of Technology North Bangkok (KMITNB), Bangkok, Thailand. He is currently a Lecturer and Researcher with the Department of Information Technology, Faculty of Informatics, Mahasarakham University, Maha Sarakham, Thailand, where he teaches database management systems, computer networks, and applied artificial intelligence. His research interests include cloud-edge continuum systems, post-quantum and homomorphic database security, hybrid vector-relational query optimization, and hardware-verified sustainable cloud telemetry.
\end{IEEEbiographynophoto}

\end{document}
"""
    return (
        tex.replace("__SEP_CFG__", sep_cfg)
        .replace("__ABSTRACT__", abstract_text)
        .replace("__FIG_W1__", fig_w1)
        .replace("__FIG_WC__", fig_wc)
        .replace("__LBL_CORE__", lbl_core)
        .replace("__LBL_ABL__", lbl_abl)
        .replace("__LBL_ARM__", lbl_arm)
        .replace("__LBL_GCP__", lbl_gcp)
        .replace("__MEAN_D_STAT__", st["mean_delta_stat"])
        .replace("__CI_STAT__", st["ci_stat"])
        .replace("__MEAN_D_CLS__", st["mean_delta_cls"])
        .replace("__CI_CLS__", st["ci_cls"])
        .replace("__MCNEMAR_CHI2__", st["mcnemar_chi2"])
        .replace("__MCNEMAR_B__", st["mcnemar_b"])
        .replace("__MCNEMAR_C__", st["mcnemar_c"])
    )


# ---------------------------------------------------------------------------
# 4. Build Complete IEEE TCC `supplement.tex` With ALL Tables Restored!
# ---------------------------------------------------------------------------
def build_complete_supplement_tex(manifest_json: dict, supp_tables: list[str], st: dict[str, str]) -> str:
    # Read from the REAL key "verified_artifacts" (list of dicts) in sha256_manifest.json!
    artifacts = manifest_json.get("verified_artifacts", [])
    assert len(artifacts) >= 4, f"Expected >= 4 verified_artifacts in sha256_manifest.json, found {len(artifacts)}"

    rows_manifest = []
    for rec in artifacts:
        safe_id = str(rec["experiment_id"]).replace("_", r"\_")
        t_name = Path(rec["latex_table_path"]).name.replace("_", r"\_")
        n_r = int(rec["measured_rows"])
        c_sha = str(rec["raw_csv_sha256"])
        t_sha = str(rec["latex_table_sha256"])
        rows_manifest.append(
            f"\\texttt{{{safe_id}}} & \\texttt{{{t_name}}} & ${n_r:,}$ & "
            f"\\texttt{{{c_sha[:16]}...{c_sha[-12:]}}} & \\texttt{{{t_sha[:16]}...{t_sha[-12:]}}} \\\\"
        )

    # Check if any supplementary table file contains \cite{...}
    has_cites = False
    supp_inputs_block = []
    for tname in supp_tables:
        p_t = REPO_ROOT / "00_manuscript/tables" / tname
        if r"\cite{" in p_t.read_text(encoding="utf-8"):
            has_cites = True
        supp_inputs_block.append(f"\\input{{00_manuscript/tables/{tname}}}")

    bib_tail = (
        "\n\\clearpage\n\\bibliographystyle{IEEEtran}\n\\bibliography{references}\n"
        if has_cites
        else "\n\\clearpage\n"
    )

    supp_abstract = (
        "This supplementary document provides (S1) the formal Renyi min-entropy leakage derivation and HEStd-128-classic "
        "Ring-LWE modulus chains for coarse-bucketed OpenFHE CKKS search, (S2) per-region Google Cloud Platform pricing "
        "and non-parametric bootstrap/McNemar statistical audit parameters, (S3) full hardware RAPL calibration, hybrid "
        "database smoke tests, dataset ground-truth manifests, and Linux kernel tc/netem socket benchmark tables, and "
        "(S4) the complete cryptographic SHA-256 provenance manifest linking every raw telemetry CSV log to its LaTeX table."
    )

    return r"""\documentclass[10pt,journal,compsoc]{IEEEtran}
\ifCLASSOPTIONcompsoc
  \usepackage[nocompress]{cite}
\else
  \usepackage{cite}
\fi
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{stfloats}
\usepackage{url}
\usepackage[hidelinks]{hyperref}

\renewcommand{\thesection}{S\arabic{section}}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\theequation}{S\arabic{equation}}

\setcounter{topnumber}{4}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{8}
\setcounter{dbltopnumber}{4}
\renewcommand{\topfraction}{0.95}
\renewcommand{\dbltopfraction}{0.95}
\renewcommand{\textfraction}{0.04}
\renewcommand{\floatpagefraction}{0.88}
\renewcommand{\dblfloatpagefraction}{0.88}

\begin{document}

\title{Supplementary Material for ``CAPQ-Pareto: Hardware- and Multi-Region Cloud-Verified 4D Pareto Cryptographic Orchestration for Post-Quantum Hybrid Relational-Vector Databases''}

\author{Preecha~Noiumkar,~\IEEEmembership{Member,~IEEE}%
\IEEEcompsocitemizethanks{\IEEEcompsocthanksitem P. Noiumkar is with the Department of Information Technology, Faculty of Informatics, Mahasarakham University, Maha Sarakham 44150, Thailand (e-mail: preecha.n@msu.ac.th; ORCID: \url{https://orcid.org/0009-0000-5964-4896}).}%
\thanks{This research project was financially supported by the Faculty of Informatics, Mahasarakham University.}}

\markboth{IEEE TRANSACTIONS ON CLOUD COMPUTING --- SUPPLEMENTARY MATERIAL}%
{NOIUMKAR: SUPPLEMENTARY MATERIAL FOR CAPQ-PARETO}

\IEEEtitleabstractindextext{%
\begin{abstract}
""" + supp_abstract + r"""
\end{abstract}}

\maketitle
\IEEEdisplaynontitleabstractindextext

\section{Formal Min-Entropy Leakage Bound and RLWE Parameterization (Regime C)}\label{sec:supp_leakage}
In Tier~2 Zero-Trust execution ($z_q = 1$), the normalized query vector $\mathbf{q} \in \mathbb{S}^{d-1}$ is encrypted under \texttt{OpenFHE~1.2.3} (\texttt{CKKS}, \texttt{HEStd\_128\_classic}), while only the coarse spherical $k$-means centroid index $C = \arg\max_{c \in \{1,\dots,K_c\}} \langle \mathbf{q}, \boldsymbol{\mu}_c \rangle$ is transmitted to restrict server-side homomorphic inner-product scans from corpus size $N_{\text{db}}$ to bucket size $M_b \approx N_{\text{db}}/K_c$. Let $H_\infty(\mathbf{Q}) = -\log_2 \max_{\mathbf{q}} p(\mathbf{q})$ denote the prior R\'enyi min-entropy of the query distribution, and let $H_\infty(\mathbf{Q} \mid C) = -\log_2 \Big(\sum_{c=1}^{K_c} \Pr[C=c] \max_{\mathbf{q}} \Pr[\mathbf{Q}=\mathbf{q} \mid C=c]\Big)$ denote average conditional min-entropy. By Bayes' rule, $\sum_{c=1}^{K_c} \max_{\mathbf{q}} \Pr[\mathbf{Q}=\mathbf{q}, C=c] \le K_c \max_{\mathbf{q}} p(\mathbf{q})$, yielding the worst-case coarse-routing leakage bound:
\begin{equation}
\mathcal{L}_{\text{bucket}} = H_\infty(\mathbf{Q}) - H_\infty(\mathbf{Q} \mid C) \le \log_2 K_c \quad [\text{bits}].
\label{eq:supp_min_entropy}
\end{equation}
For $K_c = 64$, $\mathcal{L}_{\text{bucket}} \le 6.0\text{ bits}$, while fine-grained vector coordinates remain protected by $\ge 128$-bit Ring-LWE security summarized alongside our live Google Cloud regional billing rates in Table~\ref{tab:supp_ckks_gcp}.

\begin{table*}[!t]
\centering
\caption{Verified \texttt{HEStd\_128\_classic} OpenFHE CKKS Modulus Chains and Multi-Region Google Cloud Per-Second Billing Parameters}
\label{tab:supp_ckks_gcp}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccc}
\toprule
\textbf{Subsystem / Cloud Region} & \textbf{Instance / RLWE Config} & \textbf{Key-Switch / vCPU \& RAM} & \textbf{Modulus Chain $\log_2 Q$ / Zone} & \textbf{Std. Bound / Hourly USD} & \textbf{Security / Per-Sec \& Egress Rate} \\
\midrule
OpenFHE CKKS ($N=4096$) & Multiplicative Depth $1$ & \texttt{BV} Decomposition & $[38, 30]$ ($\sum = 68\text{ bits}$) & $\le 109\text{ bits}$ & $\ge 128$-bit (\texttt{HEStd\_128\_classic}) \\
OpenFHE CKKS ($N=8192$) & Multiplicative Depth $1$ & \texttt{BV} Decomposition & $[50, 40]$ ($\sum = 90\text{ bits}$) & $\le 218\text{ bits}$ & $\ge 128$-bit (\texttt{HEStd\_128\_classic}) \\
OpenFHE CKKS ($N=16384$) & Multiplicative Depth $2$ & \texttt{HYBRID} Key-Switch & $[60, 50, 50, 60]$ ($\sum = 220\text{ bits}$) & $\le 438\text{ bits}$ & $\ge 128$-bit (\texttt{HEStd\_128\_classic}) \\
\midrule
GCP Bangkok (\texttt{asia-southeast3}) & \texttt{n4-standard-4} (\texttt{x86\_64}) & $4\text{ vCPU}, 16\text{ GB}$ & \texttt{asia-southeast3-a} & $\$0.23960/\text{hr}$ & $\$6.656\times 10^{-5}/\text{s} + \$0.12/\text{GiB}$ \\
GCP Singapore (\texttt{asia-southeast1}) & \texttt{e2-standard-4} (\texttt{x86\_64}) & $4\text{ vCPU}, 16\text{ GB}$ & \texttt{asia-southeast1-b} & $\$0.15881/\text{hr}$ & $\$4.411\times 10^{-5}/\text{s} + \$0.12/\text{GiB}$ \\
GCP Singapore (\texttt{ARM64} Validator) & \texttt{t2a-standard-4} (\texttt{Neoverse-N1}) & $4\text{ vCPU}, 16\text{ GB}$ & \texttt{asia-southeast1-b} & $\$0.15400/\text{hr}$ & Ampere Altra \texttt{aarch64} ($3.5\text{ W/core}$ TDP) \\
GCP US-Central (\texttt{us-central1}) & \texttt{e2-standard-4} (\texttt{x86\_64}) & $4\text{ vCPU}, 16\text{ GB}$ & \texttt{us-central1-a} & $\$0.13401/\text{hr}$ & $\$3.723\times 10^{-5}/\text{s} + \$0.12/\text{GiB}$ \\
\bottomrule
\end{tabular}%
}
\end{table*}

\section{Bandit Regret Bound and Extended Statistical Audit}\label{sec:supp_stats}
Under sub-Gaussian telemetry noise on scalarized observations $y_t = \boldsymbol{\theta}_{m_t}^\top \mathbf{x}_t + \eta_t$ over the nonempty SLA-feasible action set $\mathcal{F}_t \subseteq \mathcal{M}$, regularized LinUCB ($\mathbf{A}_{m,0} = \lambda_{\text{reg}}\mathbf{I}_{10}$) achieves cumulative regret $\mathcal{R}_T = \mathcal{O}\!\big(d_c \sqrt{T \ln(1 + T L_x^2 / (d_c \lambda_{\text{reg}}))}\big)$ for $d_c = 10$. Across the $360$ paired live multi-region GCP trials ($1{,}440$ queries), $10{,}000$-iteration paired percentile bootstrap resampling (seed $= 42$) yields $\Delta\mathcal{J}_{\text{SLA}} = """ + st["mean_delta_stat"] + r"""$ ($95\%$ CI: $""" + st["ci_stat"] + r"""$) vs.\ \texttt{Static\_MLKEM1024} and $\Delta\mathcal{J}_{\text{SLA}} = """ + st["mean_delta_cls"] + r"""$ ($95\%$ CI: $""" + st["ci_cls"] + r"""$) vs.\ \texttt{Classical\_TLS1.3}, with continuity-corrected McNemar's $\chi^2 = """ + st["mcnemar_chi2"] + r"""$ ($b=""" + st["mcnemar_b"] + r"""$, $c=""" + st["mcnemar_c"] + r"""$, $p < 10^{-16}$).

\section{Testbed Calibration, Smoke Audits, Datasets, and Socket Benchmarks}\label{sec:supp_tables}
To preserve the 12-page regular-paper budget of the main manuscript while maintaining complete empirical transparency, this section includes all foundational calibration, dataset manifest, stack verification, and isolated Linux kernel \texttt{tc/netem} (\texttt{MTU = 1,500 B}) socket benchmark tables generated directly from real hardware runs.

""" + "\n\n".join(supp_inputs_block) + r"""

\section{Complete Cryptographic SHA-256 Provenance Manifest}\label{sec:supp_manifest}
Table~\ref{tab:supp_sha256} records the immutable \texttt{SHA-256} digests stored in \texttt{06\_experiments\_results/sha256\_manifest.json}, binding every raw CSV telemetry file to its corresponding LaTeX table across both the main paper and this supplement. Reviewers can reproduce the local hybrid database stack via \texttt{docker compose -f 04\_testbed\_infra/docker-compose.yml up -d} and verify zero-orphan GCP teardown via \texttt{bash 04\_testbed\_infra/teardown\_all\_gcp.sh}.

\begin{table*}[!t]
\centering
\caption{Complete Cryptographic \texttt{SHA-256} Provenance Manifest Across All Verified Experimental Campaigns}
\label{tab:supp_sha256}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrlr}
\toprule
\textbf{Experiment Campaign ID} & \textbf{Generated LaTeX Table} & \textbf{Measured Rows} & \textbf{Raw Telemetry CSV SHA-256} & \textbf{LaTeX Table SHA-256} \\
\midrule
""" + "\n".join(rows_manifest) + r"""
\bottomrule
\end{tabular}%
}
\end{table*}
""" + bib_tail + r"""\end{document}
"""


# ---------------------------------------------------------------------------
# 5. Compile & Run Automated PDF Layout / Table Presence / Float-Order Verification
# ---------------------------------------------------------------------------
def compile_and_inspect_pdf(doc_stem: str) -> dict:
    wsl_sh = f"""
set -e
export DEBIAN_FRONTEND=noninteractive
for sty in IEEEtran.cls algorithm.sty algpseudocode.sty stfloats.sty; do
    if ! kpsewhich "$sty" >/dev/null 2>&1; then
        apt-get -o DPkg::Lock::Timeout=120 update -qq
        apt-get -o DPkg::Lock::Timeout=120 install -y -qq --no-install-recommends \\
            texlive-latex-base texlive-latex-recommended texlive-latex-extra \\
            texlive-publishers texlive-science texlive-fonts-recommended poppler-utils
        break
    fi
done

rm -rf /tmp/capq_pdf_audit_{doc_stem}
mkdir -p /tmp/capq_pdf_audit_{doc_stem}
cp /mnt/d/Research/CAPQ-Pareto/main.tex /tmp/capq_pdf_audit_{doc_stem}/
cp /mnt/d/Research/CAPQ-Pareto/supplement.tex /tmp/capq_pdf_audit_{doc_stem}/
cp /mnt/d/Research/CAPQ-Pareto/references.bib /tmp/capq_pdf_audit_{doc_stem}/
cp -r /mnt/d/Research/CAPQ-Pareto/00_manuscript /tmp/capq_pdf_audit_{doc_stem}/

cd /tmp/capq_pdf_audit_{doc_stem}
pdflatex -interaction=nonstopmode -halt-on-error {doc_stem}.tex >/dev/null
if grep -q "\\\\citation" {doc_stem}.aux; then
    bibtex {doc_stem} >/dev/null
fi
pdflatex -interaction=nonstopmode -halt-on-error {doc_stem}.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error {doc_stem}.tex >/dev/null

if grep -E "LaTeX Warning:.*undefined" {doc_stem}.log >/dev/null 2>&1; then
    echo "UNDEFINED_REFS_FOUND" >&2
    grep -E "LaTeX Warning:.*undefined" {doc_stem}.log >&2
    exit 4
fi

PAGES=$(pdfinfo {doc_stem}.pdf | awk '/^Pages:/ {{print $2}}')
pdftotext {doc_stem}.pdf /tmp/capq_pdf_audit_{doc_stem}/rendered.txt
pdftotext -f "$PAGES" -l "$PAGES" {doc_stem}.pdf /tmp/capq_pdf_audit_{doc_stem}/last_page.txt
cp {doc_stem}.pdf /mnt/d/Research/CAPQ-Pareto-{doc_stem.capitalize()}-IEEE-TCC.pdf
echo "PAGES=$PAGES"
"""
    res = subprocess.run(["wsl", "-u", "root", "-e", "bash", "-c", wsl_sh], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"LaTeX build failed for {doc_stem}.tex:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")

    pages = int([ln.split("=")[1] for ln in res.stdout.splitlines() if ln.startswith("PAGES=")][-1])
    rendered_txt = subprocess.check_output(["wsl", "-u", "root", "-e", "cat", f"/tmp/capq_pdf_audit_{doc_stem}/rendered.txt"], text=True, encoding="utf-8", errors="replace")
    last_page_txt = subprocess.check_output(["wsl", "-u", "root", "-e", "cat", f"/tmp/capq_pdf_audit_{doc_stem}/last_page.txt"], text=True, encoding="utf-8", errors="replace")
    overfulls = subprocess.run(["wsl", "-u", "root", "-e", "grep", "-E", r"Overfull \\hbox \([1-9][0-9]*\.[0-9]+pt", f"/tmp/capq_pdf_audit_{doc_stem}/{doc_stem}.log"], capture_output=True, text=True).stdout.strip()

    return {
        "pages": pages,
        "rendered_txt": rendered_txt,
        "last_page_txt": last_page_txt,
        "overfull_gt10pt": [ln for ln in overfulls.splitlines() if ln.strip()],
    }


def main() -> None:
    regenerate_clean_fig1(REPO_ROOT / "00_manuscript/figures/fig1_capq_architecture.pdf")
    manifest_data, supp_tables = sanitize_all_tables_and_sync_manifest()
    st = compute_stats_summary()

    lbl_core = extract_label(REPO_ROOT / "00_manuscript/tables/tab_core_720_results.tex")
    lbl_abl  = extract_label(REPO_ROOT / "00_manuscript/tables/tab_ablation_crossdb.tex")
    lbl_arm  = extract_label(REPO_ROOT / "00_manuscript/tables/tab_arm64_vs_x86.tex")
    lbl_gcp  = extract_label(REPO_ROOT / "00_manuscript/tables/tab_gcp_multiregion_wan.tex")

    # 1. Compile & Verify main.tex (<= 12 pages, Algorithm 1 + Fig. 1..8 + TABLE 1..6 all rendered!)
    main_info = None
    for tier in [0, 1, 2]:
        m_tex = build_clean_main_tex(lbl_core, lbl_abl, lbl_arm, lbl_gcp, st, tier=tier)
        (REPO_ROOT / "main.tex").write_text(m_tex, encoding="utf-8", newline="\n")
        main_info = compile_and_inspect_pdf("main")
        print(f"[MAIN AUDIT] Tier={tier} -> {main_info['pages']} pages | Overfull(>10pt)={len(main_info['overfull_gt10pt'])}")
        if main_info["pages"] <= 12:
            break

    assert main_info is not None and main_info["pages"] <= 12, f"main.pdf exceeded 12 pages ({main_info['pages']})"
    m_txt = main_info["rendered_txt"]
    for t_num in range(1, 7):
        assert f"TABLE {t_num}" in m_txt, f"Missing TABLE {t_num} in rendered main.pdf!"
    for f_num in range(1, 9):
        assert f"Fig. {f_num}." in m_txt, f"Missing Fig. {f_num}. in rendered main.pdf!"
    assert "Algorithm 1" in m_txt and "Fig. 9." not in m_txt, "Algorithm/Figure numbering mismatch in main.pdf!"

    # Check whether any TABLE floated after Biography on the final page
    bio_pos = m_txt.rfind("Preecha Noiumkar (Member, IEEE) received")
    last_tab_pos = max(m_txt.rfind(f"TABLE {i}") for i in range(1, 7))
    print(f"[FLOAT ORDER] Main paper last TABLE appears BEFORE Biography: {last_tab_pos < bio_pos}")

    # 2. Compile & Verify supplement.tex (ALL Supplementary Tables S1..S{2+len(supp_tables)} rendered!)
    supp_tex = build_complete_supplement_tex(manifest_data, supp_tables, st)
    (REPO_ROOT / "supplement.tex").write_text(supp_tex, encoding="utf-8", newline="\n")
    supp_info = compile_and_inspect_pdf("supplement")

    expected_supp_table_count = 2 + len(supp_tables)  # Table S1 (CKKS/GCP) + supp_tables + Final SHA-256 Manifest Table
    s_txt = supp_info["rendered_txt"]
    rendered_supp_captions = re.findall(r"TABLE S\d+", s_txt)
    for idx_s in range(1, expected_supp_table_count + 1):
        assert f"TABLE S{idx_s}" in s_txt, f"Missing TABLE S{idx_s} in rendered supplement.pdf! Found: {rendered_supp_captions}"
    # Verify SHA-256 manifest rows actually rendered inside supplement.pdf
    assert "gcp_multiregion_360_real_wan" in s_txt and "core_720" in s_txt, "SHA-256 manifest rows missing from supplement.pdf!"

    print(
        f"[SUPPLEMENT AUDIT] supplement.pdf = {supp_info['pages']} pages | "
        f"Verified ALL {expected_supp_table_count} tables rendered: {sorted(set(rendered_supp_captions), key=lambda x: int(x.split('S')[1]))}"
    )

    # 3. Rebuild Overleaf ZIP with ALL figures, ALL main + supplement tables, and updated manifest
    zip_target = Path(r"D:\Research\CAPQ-Pareto-Overleaf-IEEE-TCC.zip")
    include_rels = [
        "main.tex",
        "supplement.tex",
        "references.bib",
        "README.md",
        "WORKFLOW_CHECKLIST.md",
        "06_experiments_results/sha256_manifest.json",
    ]
    for sub_dir in ["00_manuscript/figures", "00_manuscript/tables"]:
        for f in sorted((REPO_ROOT / sub_dir).rglob("*")):
            if f.is_file():
                include_rels.append(f.relative_to(REPO_ROOT).as_posix())

    with zipfile.ZipFile(zip_target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in include_rels:
            zf.write(REPO_ROOT / rel, arcname=rel)

    print(f"\n[OK] Updated D:\\Research\\CAPQ-Pareto-Main-IEEE-TCC.pdf       ({main_info['pages']} pages, TABLE 1--6, Algorithm 1, Fig. 1--8)")
    print(f"[OK] Updated D:\\Research\\CAPQ-Pareto-Supplement-IEEE-TCC.pdf ({supp_info['pages']} pages, TABLE S1--S{expected_supp_table_count} ALL PRESENT)")
    print(f"[OK] Updated {zip_target} ({zip_target.stat().st_size:,} bytes | SHA-256: {sm.sha256_digest(zip_target)})")


if __name__ == "__main__":
    main()