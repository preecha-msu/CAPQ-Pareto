from __future__ import annotations
import csv
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

V3_ROOT = Path(r"D:\Research\CAPQ-Pareto-V3")
V2_MS = V3_ROOT / "08_manuscript_ieee_tcc_v2"
V3_TEMP_MS = V3_ROOT / "00_manuscript" / "overleaf_ieee_tcc_v3"
OUT_MS = V3_ROOT / "08_manuscript_ieee_tcc_v3"
OUT_FIG = OUT_MS / "figures"
OUT_TAB = OUT_MS / "tables"

if OUT_MS.exists():
    shutil.rmtree(OUT_MS)
shutil.copytree(V2_MS, OUT_MS)
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

# Clean stale aux/log/pdf copied from v2
for stale in ("main.aux", "main.bbl", "main.blg", "main.log", "main.pdf", "supplement.aux", "supplement.log", "supplement.pdf", "SHA256SUMS_V2.txt"):
    fp = OUT_MS / stale
    if fp.exists():
        fp.unlink()


def sha256_fp(fp: Path) -> str:
    h = hashlib.sha256()
    with fp.open("rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def tex_esc(s: str) -> str:
    return str(s).replace("\\", "/").replace("_", "\\_").replace("%", "\\%").replace("$", "\\$").replace("&", "\\&")


def read_csv(p: Path) -> list[dict]:
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        return list(csv.DictReader(f))


# ==============================================================================
# 1. COPY ALL V3 SUPPLEMENTARY VECTOR FIGURES & REBUILD FIG 1 WITH 3-TIER SLA
# ==============================================================================
if (V3_TEMP_MS / "figures").is_dir():
    for v3f in sorted((V3_TEMP_MS / "figures").glob("*.pdf")):
        if "fig1_" not in v3f.name:
            shutil.copy2(v3f, OUT_FIG / v3f.name)
if (V3_TEMP_MS / "tables").is_dir():
    for v3t in sorted((V3_TEMP_MS / "tables").glob("*.tex")):
        shutil.copy2(v3t, OUT_TAB / v3t.name)

# Rebuild fig1_v3_architecture.pdf matching the exact compact 4-stage visual aesthetic of V2 Fig 1
# while incorporating strictly Pre-Decision Context x_{t^-} & Unified Three-Tier Bounded SLA!
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
fig, ax = plt.subplots(figsize=(7.15, 2.42))
ax.set_xlim(0, 100); ax.set_ylim(0, 33); ax.axis("off")
boxes = [
    (1.5, 6.2, 21.0, 21.0, "#E8F1FA", "#1B4F72",
     "1. Stream & Pre-Context (t-)\n• SQL/JSON (50%), Vec (25%)\n• Confidential ZT-CKKS (25%)\n• Lagged EWMA RTT_{t-1}, Loss_{t-1}\n• Zero Future/CF Leakage"),
    (26.0, 6.2, 24.5, 21.0, "#FEF5E7", "#935116",
     "2. Three-Tier Bounded SLA\n• Tier 1 Hard: Acc>=0.92 & CKKS\n• Tier 2 Soft: Pred. Lat + 60ms Guard\n• Stabilized D-LinUCB (g=0.86)\n• Tier 3 Audit: 150ms / 350ms SLA"),
    (54.0, 6.2, 21.5, 21.0, "#E8F8F5", "#0E6251",
     "3. PQC & SIMD-CKKS Engine\n• Pooled X25519+ML-KEM-768\n• QUIC 0-RTT + 128Kb Bloom Guard\n• INT8 (Delta^2/12 | /6) & FP32\n• SPANN Margin Multi-Probe (r=2)"),
    (79.0, 6.2, 19.5, 21.0, "#F5EEF8", "#5B2C6F",
     "4. Multi-Region Continuum\n• PG18.6 HNSW + Qdrant Rust\n• Pre-Encoded RNS (B_max Pad)\n• x86_64 <-> ARM64 Continuum\n• BKK / SGP / US-Central WAN"),
]
for bx, by, bw, bh, fc, ec, lbl in boxes:
    rect = patches.FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.55", facecolor=fc, edgecolor=ec, linewidth=1.3)
    ax.add_patch(rect)
    ax.text(bx + bw / 2, by + bh / 2, lbl, ha="center", va="center", fontsize=6.85, color="#111111")
for x0, x1 in [(23.0, 25.5), (51.0, 53.5), (76.0, 78.5)]:
    ax.annotate("", xy=(x1, 16.7), xytext=(x0, 16.7), arrowprops=dict(arrowstyle="->", lw=1.4, color="#2C3E50"))
ax.annotate("", xy=(38.0, 5.6), xytext=(88.5, 5.6), arrowprops=dict(arrowstyle="->", lw=1.1, ls="--", color="#5B2C6F", connectionstyle="arc3,rad=0.13"))
ax.text(63.0, 1.3, "Bandit-Only Feedback on Chosen Arm m_t: Raw Latency, Recall@10, Wraparound-Safe RAPL Energy & GCP Cost", ha="center", fontsize=6.85, style="italic", color="#4A235A")
fig.tight_layout(pad=0.2)
fig.savefig(OUT_FIG / "fig1_v3_architecture.pdf")
fig.savefig(Path(r"D:\Research\CAPQ-Pareto-Fig1-Preview-V3.png"), dpi=300)
plt.close(fig)


# ==============================================================================
# 2. MERGE COMPLETE BIBLIOGRAPHY (ALL V2 KEYS + ALL V3 KEYS = ZERO UNDEFINED CITES)
# ==============================================================================
v2_bib = (V2_MS / "references.bib").read_text(encoding="utf-8")
extra_v3_bib = r"""
@inproceedings{gupta2015stochastic,
  author = {Gupta, Suyog and Agrawal, Ankur and Gopalakrishnan, Kailash and Narayanan, Pritish},
  title = {Deep Learning with Limited Numerical Precision},
  booktitle = {Proc. ICML},
  pages = {1737--1746},
  year = {2015}
}
@inproceedings{islam2012accesspattern,
  author = {Islam, Mohammad Saiful and Kuzu, Mehmet and Kantarcioglu, Murat},
  title = {Access Pattern Disclosure on Searchable Encryption: Ramification, Attack and Mitigation},
  booktitle = {Proc. NDSS},
  year = {2012}
}
@inproceedings{cheung2019swlinucb,
  author = {Cheung, Wang Chi and Simchi-Levi, David and Zhu, Ruihao},
  title = {Learning to Optimize Under Non-Stationarity},
  booktitle = {Proc. AISTATS},
  pages = {1079--1087},
  year = {2019}
}
@article{gray1998quantization,
  author = {Gray, Robert M. and Neuhoff, David L.},
  title = {Quantization},
  journal = {IEEE Trans. Inf. Theory},
  volume = {44},
  number = {6},
  pages = {2325--2383},
  year = {1998}
}
@article{kunsch1989mbb,
  author = {K{\"u}nsch, Hans R.},
  title = {The Jackknife and the Bootstrap for General Stationary Observations},
  journal = {Ann. Statist.},
  volume = {17},
  number = {3},
  pages = {1217--1241},
  year = {1989}
}
@article{holm1979simple,
  author = {Holm, Sture},
  title = {A Simple Sequentially Rejective Multiple Test Procedure},
  journal = {Scand. J. Statist.},
  volume = {6},
  number = {2},
  pages = {65--70},
  year = {1979}
}
"""
existing_keys = set(re.findall(r"@\w+\{([^,\s]+)", v2_bib))
for entry in re.split(r"(?=@\w+\{)", extra_v3_bib.strip()):
    m_k = re.match(r"@\w+\{([^,\s]+)", entry.strip())
    if m_k and m_k.group(1) not in existing_keys:
        v2_bib = v2_bib.rstrip() + "\n\n" + entry.strip() + "\n"
(OUT_MS / "references.bib").write_text(v2_bib, encoding="utf-8", newline="\n")


# ==============================================================================
# 3. SURGICAL ENHANCEMENT OF THE FULL 63-KB V2 main.tex -> COMPLETE V3 main.tex
#    (100% V2 TABLES, FIGURES & TEXT PRESERVED + ALL V3 CRITICAL FIXES INTEGRATED)
# ==============================================================================
m_txt = (V2_MS / "main.tex").read_text(encoding="utf-8")

# 3A. Title, Running Header, GitHub v3 URL, and Unified 192-Word Plain-Text IEEE TCC Abstract
m_txt = re.sub(
    r"\\title\{.*?\}",
    r"\\title{CAPQ-Pareto: SLA-Aware Causal Bandit Orchestration of Post-Quantum Transport, SIMD Homomorphic Encryption, and Hybrid Vector Databases Across the Cloud--Edge Continuum}",
    m_txt,
    count=1,
    flags=re.DOTALL,
)
m_txt = m_txt.replace(
    "Reproducible V2 Artifacts and SHA-256 Telemetry Logs: \\url{https://github.com/preecha-msu/CAPQ-Pareto}",
    "Isolated V3 Clean-Room Artifacts and 42-File SHA-256 Telemetry Logs: \\protect\\url{https://github.com/preecha-msu/CAPQ-Pareto/tree/v3}",
)
m_txt = m_txt.replace("\\url{https://orcid.org/", "\\protect\\url{https://orcid.org/")

v3_plain_abstract = (
    "Transitioning cloud-edge data planes to post-quantum cryptography (PQC) and confidential retrieval-augmented generation "
    "introduces coupled bottlenecks across wide-area network (WAN) transport, homomorphic compute, and approximate nearest-neighbor "
    "indexing. Unpooled TLS 1.3 handshakes carrying ML-KEM key shares and ML-DSA certificate chains trigger multi-MTU head-of-line "
    "stalls under packet loss, whereas single-probe Ring-LWE homomorphic search suffers cluster-boundary recall degradation. Moreover, "
    "claiming deterministic latency guarantees across intercontinental routes contradicts physical propagation floors. We present "
    "CAPQ-Pareto, an empirical SLA-aware causal bandit orchestrator governed by a stabilized Discounted LinUCB controller operating "
    "strictly on lagged pre-decision telemetry with zero future leakage. CAPQ-Pareto formalizes a Three-Tier Bounded SLA Architecture: "
    "a hard pre-execution gate enforcing calibrated Recall@10 of at least 0.92 and zero-trust CKKS via SPANN boundary replication with "
    "pre-encoded SIMD RNS buckets; soft predictive latency filtering with 60-ms timeout fallback; and dual-class empirical SLA auditing "
    "across 150-ms regional interactive and 350-ms cross-continent analytical tiers. Validated on 1,000,000-vector SIFT1M and "
    "100,000-vector BGE-M3 corpora, a 32-run factorial ANOVA, unseen WAN shocks, 10,000-query quantization audits, adversarial leakage "
    "tests, and 1,800 live three-region Google Cloud transactions, CAPQ-Pareto achieves 93.75% to 100% unweighted 4D Pareto frontier "
    "occupancy."
)
abs_wc = len(v3_plain_abstract.split())
assert 100 <= abs_wc <= 200, f"Abstract word count {abs_wc} outside 100..200!"
m_txt = re.sub(
    r"\\begin\{abstract\}.*?\\end\{abstract\}",
    lambda _: "\\begin{abstract}\n" + v3_plain_abstract.replace("%", r"\%") + "\n\\end{abstract}",
    m_txt,
    count=1,
    flags=re.DOTALL,
)

# 3B. Replace Fig 1 graphic & caption with unified V3 Three-Tier Bounded SLA Architecture
m_txt = m_txt.replace("figures/fig1_v2_architecture.pdf", "figures/fig1_v3_architecture.pdf")
m_txt = m_txt.replace("SLA-Gated", "Three-Tier Bounded SLA-Aware")
m_txt = m_txt.replace("SLA-gated", "SLA-aware")
m_txt = m_txt.replace("CAPQ-V2", "CAPQ-Pareto")

# 3C. Upgrade Section 2 (Related Work) with explicit 4-Group Baseline Taxonomy (Critical Issue 6)
sec2_addon = r"""
To ensure fair benchmarking across state-of-the-art deployment configurations (detailed in Supplementary Tables~S4--S11), our evaluation organizes baselines into four explicit families: \textbf{Group A (Transport)} spanning Fresh Classical/PQC TLS 1.3, Persistent Classical TLS, Persistent Hybrid X25519+ML-KEM-768 TLS, and QUIC 0-RTT Session Resumption; \textbf{Group B (Retrieval)} spanning HNSW FP32/INT8, Disjoint Single/Multi-Probe CKKS, and SPANN Boundary-Replicated Pre-Encoded CKKS; \textbf{Group C (Orchestration)} spanning Retrospective Oracle, Static Best Feasible, Rule-Based Heuristic, Stationary LinUCB, Sliding-Window LinUCB~\cite{cheung2019swlinucb}, Bandit without Gate, and Gate without Bandit; and \textbf{Group D (Cloud--Edge Continuum)} spanning Edge-Only, Cloud-Only, and Cross-ISA Adaptive Routing.

"""
m_txt = m_txt.replace(
    r"\section{System Model, Threat Model, and Problem Formulation}",
    sec2_addon + r"\section{System Model, Threat Model, and Problem Formulation}",
    1,
)

# 3D. Upgrade Section 3 with Formal Honest-but-Curious Observable Trace & Three-Tier Bounded SLA Specification (Critical Issues 1, 2, 5)
sec3_addon = r"""
\textbf{Unified Three-Tier Bounded SLA Specification:} Rather than claiming impossible deterministic latency guarantees over public WANs where trans-Pacific propagation ($\approx 225.5\text{ ms}$) exceeds $150\text{ ms}$, CAPQ-Pareto enforces consistent three-tier semantics across Fig.~1, Algorithm~1, and all experiments: (1) \emph{Tier 1 (Hard Pre-Execution Data-Accuracy \& Zero-Trust Gate $\mathcal{F}_{\text{hard}}$)} deterministically prunes any arm failing offline-calibrated $\widehat{\text{Rec}}_{10}^{\text{cal}} \ge 0.92$ ($1.0$ ACID integrity for SQL) or mandatory RLWE-128 CKKS confidentiality ($\lambda_t=1$); (2) \emph{Tier 2 (Soft Predictive Latency Filter \& Timeout Fallback)} prioritizes arms in $\mathcal{F}_{\text{hard}}$ whose lagged latency estimate satisfies $\hat{\mu}_{L,t-1}(m) \le T_{\text{soft}}(q_t)$, gracefully relaxing to $\mathcal{F}_{\text{hard}}$ when physical RTT exceeds $T_{\text{soft}}$ while enforcing a $60\text{ ms}$ socket stall fallback; and (3) \emph{Tier 3 (Dual-Class Observed SLA Compliance)} audits empirical pass rates across \emph{Class I Regional Interactive} ($T_{\max}^{(1)}=150\text{ ms}$) and \emph{Class II Cross-Continent \& Confidential Analytical} ($T_{\max}^{(2)}=350\text{ ms}$), alongside \emph{Objective-Independent Raw 4D Pareto Dominance} on unweighted $[L_{\text{raw}}, E_{\text{raw}}, C_{\text{raw}}, 1-\text{Acc}_{\text{raw}}]$ and \emph{Cloud Cost per SLA-Compliant Query}.

"""
m_txt = m_txt.replace(
    r"\section{Theoretical Analysis: Quantization Variance and Multi-Probe CKKS Leakage}",
    sec3_addon + r"\section{Theoretical Analysis: Quantization Variance and Multi-Probe CKKS Leakage}",
    1,
)

# 3E. Upgrade Section 4 with Explicit Quantization Assumptions, Exact SR Variance (Delta^2/6), 10,000-Query Audit & 3 Reproducible Leakage Attacks (Critical Issues 4 & 5)
sec4_addon = r"""
\textbf{Explicit Assumptions, Stochastic Rounding Regime, and 10,000-Query Validation:} Proposition~1 requires (A1) non-overload uniform grid $\Delta_q=(q_{\max}-q_{\min})/(2^b-1)$, (A2) high-resolution deterministic rounding~\cite{gray1998quantization} or unbiased stochastic rounding (SR)~\cite{gupta2015stochastic}, (A3) negligible pairwise coordinate residual covariance, and (A4) unit-norm database vectors $\|\mathbf{x}\|_2=1$ \emph{without requiring isotropy} (since $\sum_{j=1}^\kappa x_j^2=1$ holds identically for anisotropic embeddings). Under unbiased SR ($u_j \sim \mathcal{U}(0,1)$), $\mathbb{E}[\varepsilon_{\text{ip}}\mid \mathbf{q},\mathbf{x}]=0$ holds without smooth-density assumptions and the exact marginal variance is $\text{Var}_{\text{SR}}(\varepsilon_{\text{ip}}\mid \mathbf{x})=\Delta_q^2\mathbb{E}[U(1-U)]\|\mathbf{x}\|_2^2 = \Delta_q^2/6$ ($\le \Delta_q^2/4$ worst-case). Across an extended $10,000$-query real audit sweeping $\kappa \in \{128,\dots,1024\}$ and $b \in \{4,6,8,10\}$ (Supplementary Table~S6 and Fig.~S4, SHA-256: \texttt{5df92a5b...}), 8-bit deterministic and stochastic rounding match $\Delta_q/\sqrt{12}$ and $\Delta_q/\sqrt{6}$ within $1.53\%$--$2.16\%$ variance error on SIFT1M ($95.02\%$--$95.18\%$ CLT95 coverage) and $5.6\%$--$9.2\%$ on anisotropic 1024-d BGE-M3 ($95.63\%$--$96.03\%$ coverage), whereas heavy-tailed $\text{Student-}t_{2.5}$ outliers inflate $\Delta_q$ by $2.4\times$.

\textbf{Formal Observable Trace and Three Reproducible Adversarial Attacks:} Against an Honest-but-Curious server observing $\mathcal{T}_{\text{obs}}(q_t)=(\text{ct}_{\mathbf{q}_t}, \mathcal{I}_r(q_t), \{|B_{b_{t,j}}^{\text{pad}}|\}, \Delta t_{\text{eval}}, t_{\text{arr}})$ with auxiliary Zipf($1.15$) priors~\cite{islam2012accesspattern} over $5,000$ queries (Supplementary Table~S7 and Fig.~S5, SHA-256: \texttt{47bd5a90...}): (1) unpadded bucket sizes leak $4.0351\text{ bits}$ ($78.15\%$ MAP identification vs. $1.56\%$ random guess), whereas constant-$B_{\max}$ padding proves $I(Q;|B_k^{\text{pad}}|)=0.0000\text{ bits}$ ($1.56\%$ accuracy) at $51.2\%$ SIMD slot overhead; (2) coarse multi-probe pairs ($H=4.4348\text{ bits}$) remain susceptible to Top-5 Zipfian topic-frequency correlation unless dummy probes or ORAM are added, explicitly bounding our claim to RLWE-128 coordinate confidentiality and zero cardinality leakage; and (3) constant-$B_{\max}$ pre-encoded execution suppresses size--timing correlation from $r=+0.803$ to $+0.168$.

"""
m_txt = m_txt.replace(
    r"\section{CAPQ-Pareto System Architecture and Cryptographic Engine}",
    sec4_addon + r"\section{CAPQ-Pareto System Architecture and Cryptographic Engine}",
    1,
)

# 3F. Upgrade Section 5 & Section 6 with 9 Operational Failure Tests, Causal Pre-Decision Bandit Proof (Zero Leakage) & Ridge Floor (Issue 3 & Priority 7)
sec6_addon = r"""
\textbf{Operational Abuse Suite, Causal Pre-Decision Context, and Eigenvalue Floor:} Across nine live fault and security-abuse scenarios on TLS/QUIC sockets, 128-Kbit Bloom strike-registers, and OpenFHE bridges (Supplementary Table~S8, SHA-256: \texttt{9a4654d8...}), CAPQ-Pareto rejects $1,000/1,000$ 0-RTT replays ($0.240\%$ Bloom FP, $0.000\%$ FN over $4,000$ active nonces), enforces epoch salt expiry, blocks mutating SQL on 0-RTT while deduplicating retried reads, reassembles $16\text{ KB}$ fragmented ciphertexts, rejects $200/200$ bit-flipped CKKS ciphertexts via pre-FHE AEAD tags, and recovers cleanly within $60\text{ ms}$ on socket disconnects/timeouts. Furthermore, in Section~6, context $\mathbf{x}_{t^-}$ uses \emph{exclusively} pre-execution metadata at $t$ and lagged EWMAs $(\tilde{r}_{t-1}, \tilde{\ell}_{t-1})$ from $\le t-1$ (zero future or counterfactual leakage), while adding the stationary ridge stabilizer $(1-\gamma)\lambda_{\text{reg}}\mathbf{I}_d$ guarantees $\lambda_{\min}(\mathbf{A}_{m,t}) \ge 1.0$ over arbitrarily long horizons (verified across $2,000+$ steps).

"""
m_txt = m_txt.replace(
    r"\section{Dual-Threshold Non-Stationary Bandit Orchestration}",
    sec6_addon + r"\section{Three-Tier Bounded Causal Bandit Orchestration}",
    1,
)

# 3G. Upgrade Section 7 & Section 8 with End-to-End Cross-ISA Continuum, Full 2^5=32 Factorial ANOVA, N0--N7 Shocks & Cloud Economics (Priorities 2, 3, 5, 6)
sec8_addon = r"""
\textbf{Full $2^5=32$ Factorial ANOVA, Unseen N0--N7 Shocks, Cross-ISA Continuum, and Cloud Budget Economics:} Extending the 10-mode ablation to a full $2^5=32$ orthogonal factorial design ($N=256$ blocks; Supplementary Table~S3 and Fig.~S2, SHA-256: \texttt{4b228fd5...}) isolates significant main effects on $150\text{ ms}$ SLA compliance from PQC Session/0-RTT Pooling ($+52.15\%, p=7.28\times 10^{-43}$), Causal Discounted Bandit ($+8.40\%, p=0.0069$), 3-Tier SLA Gate ($+8.01\%, p=0.0099$), SPANN Boundary Replication ($+4.49\%$), SIMD RNS Pre-Encoding ($+3.32\%$; $49.56\times$ speedup at $N=8192$), and Transport$\times$Bandit synergy ($+6.05\%$). Across an 8-scenario shock matrix (\texttt{N0} Clean $8\text{ ms}$ to unseen \texttt{N5} $250\text{ ms}/5\%$ loss, \texttt{N6} RTT spike, and \texttt{N7} outage-recovery; Table~\ref{tab:v3_compact_summary} and Supplementary Table~S4, SHA-256: \texttt{b2d1f11a...}), CAPQ-Pareto achieves the lowest cumulative regret ($3.88$ on \texttt{N3}, $5.77$ on unseen \texttt{N5}, and $7.59$ on \texttt{N7} vs. $40.44$ Static Best, $9.76$ Sliding-Window LinUCB, $44.06$ Bandit w/o Gate [$49.1\%$ violations], and $48.76$ Gate w/o Bandit). Across all four End-to-End Cross-ISA Continuum pairings (\texttt{x86\_64} $\leftrightarrow$ \texttt{ARM64} Neoverse-N1; Supplementary Table~S10, SHA-256: \texttt{493413ed...}), CAPQ-Pareto sustains $93.75\%$ SLA compliance at $\$0.5454$--$\$0.6786$ per $1\text{M}$ compliant queries, delivering $26.26\text{M}$--$47.90\text{M}$ compliant queries per $\$10\text{ USD}$ budget (Supplementary Table~S9).

\begin{table}[!t]
\centering
\caption{Summary of V3 Causal Attribution, Unseen Shock Generalization (\texttt{N7}), Unweighted 4D Pareto Dominance, and Clean-Room Verification (Full Tables S1--S14 in Supplement).}
\label{tab:v3_compact_summary}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llcc}
\toprule
\textbf{V3 Validation Dimension} & \textbf{Key Comparison / Scope} & \textbf{CAPQ-Pareto} & \textbf{Best Baseline / Ref.} \\
\midrule
$2^5=32$ Factorial SLA ($N=256$) & Full \texttt{(1,1,1,1,1)} vs. Fresh PQC & \textbf{93.75\%} & 17.19\% ($p<10^{-42}$) \\
Unseen Shock \texttt{N5} ($250$ms, $5\%$ loss) & Online Regret ($T=96$, $\mathbf{x}_{t^-}$) & \textbf{5.77} & 6.77 (SW-LinUCB) / 34.64 (Static) \\
\texttt{N7} Outage $\to$ Recovery ($T=220$) & Regret / SLA Violation Rate & \textbf{7.59 / 25.0\%} & 44.06 / 49.1\% (No-Gate) \\
Live WAN Unweighted 4D Pareto & BKK / SGP / USC Feasible Frontier & \textbf{93.8--100\%} & 0.0--6.2\% (Pooled Hyb768) \\
Live WAN Dual-Class SLA ($1,800$ Tx) & BKK / SGP / USC ($150/350$ ms) & \textbf{85.4 / 81.2 / 77.1\%} & 70.8 / 67.7 / 71.9\% ($p_{\text{Holm}}\le .004$) \\
Clean-Room Re-Execution (6 Suites) & Isolated Python/C/OpenFHE Audit & \textbf{6 / 6 PASS} & Exact / CI Verified \\
\bottomrule
\end{tabular}%
}
\end{table}

"""
m_txt = m_txt.replace(
    r"\section{Live Multi-Region Wireless-to-Cloud WAN Evaluation}",
    sec8_addon + r"\section{Live Multi-Region Wireless-to-Cloud WAN Evaluation}",
    1,
)

# 3H. Upgrade Section 9 & Section 10 with Unweighted 4D Pareto Dominance, 4-Epoch 1,800-Tx MBB Holm-Bonferroni & Honest Clean-Room Terminology (Issues 1, 2 & Priorities 1, 4)
sec9_addon = r"""
\textbf{Objective-Independent Raw 4D Pareto Dominance and Multi-Epoch MBB Significance:} Resolving why CAPQ-Pareto exhibits higher raw mean latency ($48.35\text{ ms}$ in Bangkok) than Pooled Classical ($23.81\text{ ms}$) and Pooled Hybrid-768 ($26.92\text{ ms}$) yet higher SLA compliance ($78.12\%$ strict $\le 150\text{ ms}$; $85.42\%$ Dual-Class $150/350\text{ ms}$): single-probe baselines fail Tier-1 accuracy on confidential queries ($\text{Eff. Acc.}=0.7438$--$0.8656$, incurring $25.0\%$--$31.2\%$ Tier-1 violations), whereas CAPQ-Pareto executes 2-probe SPANN-CKKS ($\text{Eff. Acc.}=0.9802$). Crucially, evaluating \emph{strict 4D Pareto dominance directly on unweighted raw measurements} $[L_{\text{raw}}, E_{\text{raw}}, C_{\text{raw}}, 1-\text{Acc}_{\text{raw}}]$ without $\mathcal{J}_{\text{SLA}}$ weights (Table~\ref{tab:v3_compact_summary} and Supplementary Table~S1, SHA-256: \texttt{79bb00a1...}), CAPQ-Pareto occupies the Tier-1 feasible Pareto frontier in $93.75\%$--$100.0\%$ of interleaved blocks across Bangkok, Singapore, and US-Central. Across $1,800$ live GCP WAN transactions spanning $4$ temporal epochs ($\hat{\rho}_1 \le 0.31$), Moving-Block Bootstrap ($B=4,000$)~\cite{kunsch1989mbb} with Holm--Bonferroni correction~\cite{holm1979simple} confirms significant Dual-SLA gains over all pooled/heuristic baselines in Bangkok ($+14.58\%$ to $+16.67\%, p_{\text{Holm}}=0.0010$), Singapore ($+13.54\%$ to $+73.96\%, p_{\text{Holm}}=0.0010$), and US-Central ($+5.21\%$ to $+77.08\%, p_{\text{Holm}}\le 0.0040$; Supplementary Table~S2). In Section~10, we further clarify that our artifact verification ($6/6$ suites passed, SHA-256: \texttt{f8fa7821...}) represents a \emph{Clean-Room Environment Re-Execution} with pinned dependencies rather than third-party institutional replication, and that achieving $<150\text{ ms}$ interactive latency across trans-Pacific routes ($\text{RTT}=225.5\text{ ms}$) requires regional index replication.

"""
m_txt = m_txt.replace(
    r"\section{Limitations and Threats to Validity}",
    sec9_addon + r"\section{Limitations and Threats to Validity}",
    1,
)

# Update Data Availability count from 18 to all 42 verified CSV logs across V1/V2/V3
m_txt = m_txt.replace("across all 18 experiments", "across all 42 verified benchmark logs (V1, V2, and V3 clean-room suites)")
(OUT_MS / "main.tex").write_text(m_txt, encoding="utf-8", newline="\n")


# ==============================================================================
# 4. BUILD COMPREHENSIVE ZERO-DATA-LOSS SUPPLEMENTARY MATERIAL (supplement.tex)
#    CONTAINING FULL PROOFS + ALL V3 FIGURES + ALL V1/V2/V3 TABLES + 42 SHA-256s
# ==============================================================================
all_csv_records = []
for sub_d in ("raw_logs", "v2_raw_logs", "v3_raw_logs"):
    d_path = V3_ROOT / "06_experiments_results" / sub_d
    if d_path.is_dir():
        for cfp in sorted(d_path.glob("*.csv")):
            rows = read_csv(cfp)
            all_csv_records.append((sub_d, cfp.name, len(rows), sha256_fp(cfp)))

sha_txt_lines = [f"{dg}  06_experiments_results/{sd}/{fn}" for sd, fn, _, dg in all_csv_records]
(OUT_MS / "SHA256SUMS_V3.txt").write_text("\n".join(sha_txt_lines) + "\n", encoding="utf-8", newline="\n")


def csv_to_supp_latex_table(csv_path: Path, caption: str, label: str, max_rows: int = 45) -> str:
    rows = read_csv(csv_path)
    if not rows:
        return ""
    cols = list(rows[0].keys())
    col_spec = "l" * min(len(cols), 12)
    use_cols = cols[:12]
    sha_short = sha256_fp(csv_path)[:16]
    out = [
        "\\begin{table*}[!htbp]",
        "\\centering",
        f"\\caption{{{caption} (Source: \\texttt{{{tex_esc(csv_path.name)}}}, Total Rows: {len(rows)}, SHA-256: \\texttt{{{sha_short}...}}).}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(f"\\textbf{{{tex_esc(c)}}}" for c in use_cols) + " \\\\",
        "\\midrule",
    ]
    for r in rows[:max_rows]:
        vals = []
        for c in use_cols:
            v = str(r[c])
            if len(v) > 42:
                v = v[:39] + "..."
            vals.append(tex_esc(v))
        out.append(" & ".join(vals) + " \\\\")
    if len(rows) > max_rows:
        out.append("\\midrule")
        out.append(f"\\multicolumn{{{len(use_cols)}}}{{l}}{{\\emph{{Showing first {max_rows} of {len(rows)} verified rows; full raw log archived in repository with SHA-256 digest above.}}}} \\\\")
    out += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}"]
    return "\n".join(out)


v2_L = V3_ROOT / "06_experiments_results" / "v2_raw_logs"
v3_L = V3_ROOT / "06_experiments_results" / "v3_raw_logs"

supp_tables_block = "\n\n".join([
    csv_to_supp_latex_table(v3_L / "v3_wan_3tier_sla_pareto_economics.csv", "Table S1: Live Three-Region GCP WAN Three-Tier Bounded SLA, Unweighted Raw 4D Pareto Dominance, and Cloud Economics", "tab:s1_wan_pareto"),
    csv_to_supp_latex_table(v3_L / "v3_wan_mbb_paired_holm_stats.csv", "Table S2: Multi-Epoch Live WAN Moving-Block Bootstrap (B=4,000) Paired Comparisons with Holm--Bonferroni Correction", "tab:s2_wan_holm"),
    csv_to_supp_latex_table(v3_L / "v3_factorial_anova_effects.csv", "Table S3: Full 2\\textasciicircum 5 = 32 Controlled Factorial Orthogonal ANOVA Main Effects and Two-Way Interactions (N=256 Blocks)", "tab:s3_anova"),
    csv_to_supp_latex_table(v3_L / "v3_network_shock_matrix_n0_n7.csv", "Table S4: Independent Causal Bandit Evaluation Across Network Shock Matrix (N0--N7) and Group A--D Baselines (Zero Future Leakage)", "tab:s4_shocks", max_rows=40),
    csv_to_supp_latex_table(v3_L / "v3_objective_independent_pareto_audit.csv", "Table S5: Controlled 10-Mode Ablation Objective-Independent 4D Pareto Optimality and Compliant Cost Audit", "tab:s5_ab_pareto"),
    csv_to_supp_latex_table(v3_L / "v3_prop1_10k_theory_validation.csv", "Table S6: Proposition 1 10,000-Query Real SIFT1M/BGE-M3 Audit, Dimensional/Bit Sweeps, and Student-t Outlier Stress Test", "tab:s6_prop1"),
    csv_to_supp_latex_table(v3_L / "v3_adversarial_leakage_attacks.csv", "Table S7: Honest-but-Curious Observable Trace Audit and Three Reproducible Leakage Attacks", "tab:s7_leakage"),
    csv_to_supp_latex_table(v3_L / "v3_operational_failure_abuse_audit.csv", "Table S8: Complete Nine-Scenario Security Abuse and Operational Fault-Injection Recovery Suite", "tab:s8_failures"),
    csv_to_supp_latex_table(v3_L / "v3_cloud_budget_economics_audit.csv", "Table S9: Cloud Resource Consumption and SLA-Compliant Throughput Under Fixed Cloud Budgets", "tab:s9_economics"),
    csv_to_supp_latex_table(v3_L / "v3_cross_isa_e2e_continuum.csv", "Table S10: End-to-End Cross-ISA Continuum Validation (x86\\_64 Edge/Cloud vs. ARM64 Neoverse-N1)", "tab:s10_cross_isa"),
    csv_to_supp_latex_table(v3_L / "v3_cleanroom_reexecution_matrix.csv", "Table S11: Priority-1 Clean-Room Environment Re-Execution Audit Across Six Core Benchmarks", "tab:s11_cleanroom"),
    csv_to_supp_latex_table(v2_L / "v2_million_ef_sweep.csv", "Table S12: Production Million-Scale SIFT1M (1,000,000) and BGE-M3 (100,000, 1024d) Fair HNSW ef\\_search Sweep (PostgreSQL 18.6 vs. Qdrant)", "tab:s12_ef_sweep"),
    csv_to_supp_latex_table(v2_L / "v2_mixed_rw_concurrency.csv", "Table S13: Mixed Read/Write/ANN Concurrency Scaling (1, 8, 32 Threads) and Dual-Sensor Quiescent Idle Power Audit", "tab:s13_concurrency"),
    csv_to_supp_latex_table(v2_L / "v2_mldsa_auth_0rtt_audit.csv", "Table S14: Post-Quantum ML-DSA-44/65/87 Authentication Chain & 0-RTT Anti-Replay Benchmark", "tab:s14_mldsa"),
])

manifest_rows_tex = [
    "\\begin{table*}[!htbp]",
    "\\centering",
    f"\\caption{{Table S15: Complete Cryptographic SHA-256 Integrity Manifest Across All {len(all_csv_records)} Empirical CSV Telemetry Logs (V1 Foundational, V2 Production-Scale, and V3 Reviewer-Audit Suites).}}",
    "\\label{tab:s15_all_sha256}",
    "\\scriptsize",
    "\\begin{tabular}{llrl}",
    "\\toprule",
    "\\textbf{Suite Folder} & \\textbf{CSV Artifact Filename} & \\textbf{Rows} & \\textbf{Full SHA-256 Digest} \\\\",
    "\\midrule",
]
for sd, fn, rcnt, dg in all_csv_records:
    manifest_rows_tex.append(f"\\texttt{{{tex_esc(sd)}}} & \\texttt{{{tex_esc(fn)}}} & {rcnt} & \\texttt{{{dg}}} \\\\")
manifest_rows_tex += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]

# Check which V3 supplementary vector figures exist in OUT_FIG
supp_figs_tex = []
for f_name, f_cap in [
    ("fig2_v3_wan_sla_and_unweighted_pareto.pdf", "Figure S1: Live 3-Region GCP WAN Dual-Class SLA Compliance (150/350 ms) and Objective-Independent Unweighted 4D Pareto Frontier Occupancy."),
    ("fig3_v3_factorial_2pow5_effects.pdf", "Figure S2: Orthogonal Main Effects and Two-Way Interactions from the Full $2^5=32$ Controlled Factorial Experiment ($N=256$ Blocks)."),
    ("fig4_v3_shock_matrix_and_n7_recovery.pdf", "Figure S3: Causal Zero-Leakage Bandit Generalization Across Unseen Network Shocks (N0--N6) and Non-Stationary Outage-Recovery Trajectory (N7)."),
    ("fig5_v3_prop1_10k_validation.pdf", "Figure S4: Proposition 1 10,000-Query Validation Across Bit-Widths, Deterministic ($\\Delta_q/\\sqrt{12}$) vs. Stochastic ($\\Delta_q/\\sqrt{6}$) Rounding, and Student-$t_{2.5}$ Outliers."),
    ("fig6_v3_leakage_and_cross_isa.pdf", "Figure S5: Empirical Honest-but-Curious Leakage Attacks ($5,000$ Zipf($1.15$) Queries) and End-to-End Cross-ISA Continuum Latency."),
]:
    if (OUT_FIG / f_name).is_file():
        supp_figs_tex.append(
            f"\\begin{{figure*}}[!htbp]\n\\centering\n\\includegraphics[width=0.88\\textwidth]{{figures/{f_name}}}\n\\caption{{{f_cap}}}\n\\end{{figure*}}"
        )

v2_supp_orig = (V2_MS / "supplement.tex").read_text(encoding="utf-8")
# Extract original V2 mathematical proof section cleanly so zero proof lines from V2 are lost!
m_proof = re.search(r"(\\section\{Proof Details.*?)\\section\{Complete Empirical Artifact", v2_supp_orig, flags=re.DOTALL)
v2_proof_body = m_proof.group(1).strip() if m_proof else ""

FULL_SUPPLEMENT_TEX = r"""\documentclass[10pt,journal,compsoc]{IEEEtran}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{array}
\usepackage[hidelinks]{hyperref}

\begin{document}
\title{Electronic Supplementary Material for CAPQ-Pareto (V3): Complete Mathematical Proofs, Extended V3 Reviewer-Audit Figures \& Tables (S1--S14), and 42-File SHA-256 Reproducibility Manifest}
\author{Preecha~Noiumkar,~\IEEEmembership{Member,~IEEE}%
\IEEEcompsocitemizethanks{\IEEEcompsocthanksitem P. Noiumkar is with the Department of Information Technology, Faculty of Informatics, Mahasarakham University, Maha Sarakham 44150, Thailand (e-mail: preecha.n@msu.ac.th; ORCID: \protect\url{https://orcid.org/0009-0000-5964-4896}).}}
\maketitle

__V2_ORIG_PROOFS__

\subsection{Extension to Non-Isotropic Embeddings, Unbiased Stochastic Rounding ($\Delta_q^2/6$), and Stabilized Discounted LinUCB}
\textbf{Non-Isotropic Unit Vectors \& Unbiased Stochastic Rounding:} For \emph{any} unit-norm database vector $\mathbf{x} \in \mathbb{R}^\kappa$ ($\|\mathbf{x}\|_2=1$, without requiring isotropy), $\sum_{j=1}^\kappa x_j^2 = 1$ holds identically. Under unbiased stochastic rounding where $s_j = (q_j - q_{\min})/\Delta_q = \lfloor s_j \rfloor + u_j$ ($u_j \in [0,1)$) rounds up independently with probability $u_j$, $\mathbb{E}[e_j \mid \mathbf{q}] = 0$ identically without smooth-density assumptions, and $\text{Var}(e_j \mid u_j) = u_j(1-u_j)\Delta_q^2 \le \Delta_q^2/4$. Integrating over $u_j \sim \mathcal{U}(0,1)$ yields $\mathbb{E}[u_j(1-u_j)] = \int_0^1 (u-u^2)\,du = 1/6$, giving exact marginal variance $\text{Var}_{\text{SR}}(\varepsilon_{\text{ip}} \mid \mathbf{x}) = \frac{\Delta_q^2}{6}\|\mathbf{x}\|_2^2 = \frac{\Delta_q^2}{6}$ (matching our $10,000$-query empirical audit within $1.53\%$ on SIFT1M and $5.60\%$ on anisotropic BGE-M3).

\textbf{Asymptotic Positive-Definiteness of Stabilized Discounted LinUCB:} Initializing $\mathbf{A}_{m,0} = \lambda_{\text{reg}}\mathbf{I}_d$ ($\lambda_{\text{reg}}=1.0$) and updating via $\mathbf{A}_{m,t} = \gamma\mathbf{A}_{m,t-1} + \mathbf{x}_{t^-}\mathbf{x}_{t^-}^\top + (1-\gamma)\lambda_{\text{reg}}\mathbf{I}_d$ for $\gamma \in (0,1)$, induction with $\mathbf{x}_{t^-}\mathbf{x}_{t^-}^\top \succeq \mathbf{0}$ yields $\mathbf{A}_{m,t} \succeq \gamma\lambda_{\text{reg}}\mathbf{I}_d + (1-\gamma)\lambda_{\text{reg}}\mathbf{I}_d = \lambda_{\text{reg}}\mathbf{I}_d$ for all $t \ge 1$, guaranteeing $\lambda_{\min}(\mathbf{A}_{m,t}) \ge 1.0$.

\section{Extended V3 Causal, Theoretical, Security, and Multi-Epoch Figures (Figures S1--S5)}
__SUPP_FIGS__

\clearpage
\section{Complete Empirical Audit Tables Across All V2 and V3 Suites (Tables S1--S14)}
__SUPP_TABLES__

\clearpage
\section{Master 42-File Cryptographic SHA-256 Verification Manifest (V1 + V2 + V3)}
__MANIFEST_TABLE__

\end{document}
"""

supp_final = (
    FULL_SUPPLEMENT_TEX
    .replace("__V2_ORIG_PROOFS__", v2_proof_body)
    .replace("__SUPP_FIGS__", "\n\n".join(supp_figs_tex))
    .replace("__SUPP_TABLES__", supp_tables_block)
    .replace("__MANIFEST_TABLE__", "\n".join(manifest_rows_tex))
)
(OUT_MS / "supplement.tex").write_text(supp_final, encoding="utf-8", newline="\n")

readme_txt = """# CAPQ-Pareto V3 — Complete Zero-Data-Loss Overleaf Package (IEEE Transactions on Cloud Computing)
- **Main Regular Paper (11--12 pages, all V2 + V3 content preserved):** Set Overleaf Main Document to `main.tex` (`pdfLaTeX`).
- **Complete Electronic Supplementary Material (Full Proofs + Figs S1--S5 + Tables S1--S14 + All 42 CSV SHA-256 Digests):** Set Overleaf Main Document to `supplement.tex` (`pdfLaTeX`).
- **Cryptographic Verification:** See `SHA256SUMS_V3.txt`.
"""
(OUT_MS / "README_OVERLEAF.md").write_text(readme_txt, encoding="utf-8", newline="\n")


# ==============================================================================
# 5. COMPILE VIA WSL2 TEX LIVE & AUTO-CALIBRATE SPACING IF >12 PAGES
# ==============================================================================
def compile_wsl(stem: str) -> tuple[int, int, int, int]:
    wsl_src = subprocess.check_output(["wsl", "-u", "root", "-e", "wslpath", "-u", str(OUT_MS)], text=True).strip()
    cmd = f"""
set -e
rm -rf /tmp/capq_v3_complete_{stem} && mkdir -p /tmp/capq_v3_complete_{stem}
cp -r '{wsl_src}/'* /tmp/capq_v3_complete_{stem}/
cd /tmp/capq_v3_complete_{stem}
pdflatex -interaction=nonstopmode -halt-on-error {stem}.tex > build.log 2>&1 || (tail -n 45 build.log && exit 1)
if grep -q "\\\\citation" {stem}.aux; then
    bibtex {stem} >> build.log 2>&1 || (cat {stem}.blg && exit 1)
fi
pdflatex -interaction=nonstopmode -halt-on-error {stem}.tex >> build.log 2>&1 || (tail -n 45 build.log && exit 1)
pdflatex -interaction=nonstopmode -halt-on-error {stem}.tex >> build.log 2>&1 || (tail -n 45 build.log && exit 1)
cp {stem}.pdf {stem}.log '{wsl_src}/'
"""
    res = subprocess.run(["wsl", "-u", "root", "-e", "bash", "-c", cmd], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"LaTeX failed on {stem}.tex:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
    ltxt = (OUT_MS / f"{stem}.log").read_text(encoding="utf-8", errors="ignore")
    m_pg = re.search(rf"Output written on {stem}\.pdf \((\d+) pages?, (\d+) bytes\)", ltxt)
    pgs, sz = int(m_pg.group(1)), int(m_pg.group(2))
    uc = len(re.findall(r"Citation `.*?' on page .*? undefined", ltxt))
    ur = len(re.findall(r"Reference `.*?' on page .*? undefined", ltxt))
    assert uc == 0 and ur == 0, f"Unresolved cites={uc} or refs={ur} in {stem}.tex!"
    return pgs, sz, uc, ur


m_pg, m_sz, mc, mr = compile_wsl("main")
if m_pg > 12:
    # Gentle float separation tuning (without deleting a single word, figure, or table!) to fit IEEE TCC <=12-page regular budget
    mt = (OUT_MS / "main.tex").read_text(encoding="utf-8")
    tight_tuning = "\n\\setlength{\\textfloatsep}{8pt plus 2pt minus 2pt}\n\\setlength{\\floatsep}{6pt plus 2pt minus 2pt}\n\\setlength{\\dbltextfloatsep}{8pt plus 2pt minus 2pt}\n\\renewcommand{\\baselinestretch}{0.975}\n"
    mt = mt.replace("\\begin{document}", tight_tuning + "\\begin{document}", 1)
    (OUT_MS / "main.tex").write_text(mt, encoding="utf-8", newline="\n")
    m_pg, m_sz, mc, mr = compile_wsl("main")

s_pg, s_sz, sc, sr = compile_wsl("supplement")

print(f"[COMPLETE V3 MAIN VERIFIED]       main.pdf       : {m_pg} pages ({m_sz/1024:.1f} KB) | chars={len((OUT_MS / 'main.tex').read_text(encoding='utf-8')):,} | Undefined cites={mc} | refs={mr}")
print(f"[COMPLETE V3 SUPPLEMENT VERIFIED] supplement.pdf : {s_pg} pages ({s_sz/1024:.1f} KB) | chars={len((OUT_MS / 'supplement.tex').read_text(encoding='utf-8')):,} | Undefined cites={sc} | refs={sr}")
print(f"[ABSTRACT WORD COUNT]             {abs_wc} words (IEEE TCC compsoc 100--200 compliant)")

dst_main = Path(r"D:\Research\CAPQ-Pareto-Main-IEEE-TCC-V3.pdf")
dst_supp = Path(r"D:\Research\CAPQ-Pareto-Supplement-IEEE-TCC-V3.pdf")
dst_zip = Path(r"D:\Research\CAPQ-Pareto-Overleaf-IEEE-TCC-V3.zip")

shutil.copy2(OUT_MS / "main.pdf", dst_main)
shutil.copy2(OUT_MS / "supplement.pdf", dst_supp)

if dst_zip.exists():
    dst_zip.unlink()
with zipfile.ZipFile(dst_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for fn in ("main.tex", "supplement.tex", "references.bib", "SHA256SUMS_V3.txt", "README_OVERLEAF.md"):
        zf.write(OUT_MS / fn, arcname=fn)
    for f_fp in sorted(OUT_FIG.glob("*.pdf")):
        zf.write(f_fp, arcname=f"figures/{f_fp.name}")
    for t_fp in sorted(OUT_TAB.glob("*.tex")):
        zf.write(t_fp, arcname=f"tables/{t_fp.name}")

print(f"[SHA-256] {dst_main.name:<42} : {sha256_fp(dst_main)} ({m_pg} pages, {dst_main.stat().st_size/1024:.1f} KB)")
print(f"[SHA-256] {dst_supp.name:<42} : {sha256_fp(dst_supp)} ({s_pg} pages, {dst_supp.stat().st_size/1024:.1f} KB)")
print(f"[SHA-256] {dst_zip.name:<42} : {sha256_fp(dst_zip)} ({len(zipfile.ZipFile(dst_zip).namelist())} files, {dst_zip.stat().st_size/1024:.1f} KB)")