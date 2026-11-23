import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(r"D:\Research\CAPQ-Pareto")
ZIP_TARGET = Path(r"D:\Research\CAPQ-Pareto-Overleaf-IEEE-TCC.zip")

MAIN_TEX = r"""\documentclass[10pt,journal,compsoc]{IEEEtran}

\ifCLASSOPTIONcompsoc
  \usepackage[nocompress]{cite}
\else
  \usepackage{cite}
\fi

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{xcolor}
\usepackage[hidelinks]{hyperref}

\graphicspath{{00_manuscript/figures/workflow/}{00_manuscript/figures/pareto_4d_plots/}}

\begin{document}

\title{CAPQ-Pareto: Context-Aware Post-Quantum and Homomorphic Encryption Orchestration for Hybrid Structured-Unstructured Databases Across the Cloud-Edge Continuum}

\author{Preecha~Noiumkar,~\IEEEmembership{Member,~IEEE}%
\IEEEcompsocitemizethanks{\IEEEcompsocthanksitem P. Noiumkar is with the Department of Information Technology, Faculty of Informatics, Mahasarakham University, Maha Sarakham 44150, Thailand (e-mail: preecha.n@msu.ac.th; ORCID: \url{https://orcid.org/0009-0000-5964-4896}).\protect\\
Corresponding author: Preecha Noiumkar (e-mail: preecha.n@msu.ac.th).}%
\thanks{This research project was financially supported by the Faculty of Informatics, Mahasarakham University. Reproducible Artifacts and SHA-256 Telemetry Logs: \url{https://github.com/preecha-msu/CAPQ-Pareto}.}}

\markboth{IEEE TRANSACTIONS ON CLOUD COMPUTING}%
{NOIUMKAR: CAPQ-PARETO ORCHESTRATION FOR HYBRID CLOUD-EDGE DATABASES}

\IEEEtitleabstractindextext{%
\begin{abstract}
Cloud-edge applications such as Retrieval-Augmented Generation (RAG) and clinical analytics increasingly rely on hybrid databases that jointly serve lossless structured transactions (SQL/JSON) and approximate vector similarity queries across wide-area networks (WAN). However, securing these pipelines against quantum adversaries introduces severe cloud-edge bottlenecks: standardized Post-Quantum Cryptography (NIST FIPS 203 ML-KEM) causes handshake expansion and WAN packet fragmentation, whereas zero-trust vector evaluation via approximate Fully Homomorphic Encryption (CKKS) incurs substantial latency, edge energy drain, and cloud billing costs while precluding server-side graph traversal. We propose CAPQ-Pareto, a context-aware orchestration framework that dynamically routes hybrid database workloads across heterogeneous cryptographic and compression regimes. We formulate joint encryption-quantization selection as a constrained multi-objective Pareto optimization problem minimizing tail latency, hardware-measured edge energy, and per-second cloud compute and egress costs under strict security, deadline, and retrieval accuracy (Recall@K) bounds. An online SLA-gated Contextual Bandit scheduler dispatches batches across micro-batched QUIC 0-RTT with hybrid X25519/ML-KEM-768, precision-quantized vector transport over lightweight PQC, and coarse-bucketed CKKS inner-product search. All evaluations enforce strict real execution across bare-metal Intel RAPL edge containers and multi-region Google Cloud x86\_64 and ARM64 clusters running PostgreSQL (pgvector) and Milvus with SHA-256 verified telemetry logs.
\end{abstract}

\begin{IEEEkeywords}
Cloud-edge continuum, post-quantum cryptography, ML-KEM, homomorphic encryption, hybrid vector databases, multi-objective optimization, query processing, tail latency.
\end{IEEEkeywords}}

\maketitle
\IEEEdisplaynontitleabstractindextext
\IEEEpeerreviewmaketitle

\IEEEraisesectionheading{\section{Introduction and Problem Statement}\label{sec:introduction}}
\IEEEPARstart{C}{loud-edge} data infrastructures supporting enterprise Retrieval-Augmented Generation (RAG), smart healthcare analytics, and industrial telemetry jointly execute transactional metadata filters (structured SQL/JSON requiring $100\%$ lossless relational integrity) and approximate nearest-neighbor (ANN) similarity searches over high-dimensional embeddings ($128$--$1024$ dimensions) across hybrid engines such as PostgreSQL with \texttt{pgvector} and Milvus~\cite{wang2021milvus, malkov2020hnsw}. Concurrently, protecting long-lived cloud-edge traffic against \textit{Harvest-Now, Decrypt-Later} quantum adversaries and honest-but-curious cloud operators requires transitioning from classical elliptic-curve key exchange (X25519) to NIST Module-Lattice Key-Encapsulation Mechanism (ML-KEM)~\cite{nist_fips203_2024, stebila2016liboqs} and approximate Fully Homomorphic Encryption (CKKS)~\cite{cheon2017ckks, badawi2022openfhe}.

\subsection{Critical Cloud-Edge Bottlenecks}
Static cryptographic policies across hybrid database streams introduce three systems bottlenecks:
\begin{enumerate}
    \item \textbf{PQC Handshake Expansion and WAN Tail Latency:} Static PQC handshakes (e.g., ML-KEM-1024 over TCP/TLS 1.3) expand key-exchange payloads beyond standard Ethernet Maximum Transmission Units (MTU), inducing packet fragmentation and tail-latency spikes under WAN packet loss~\cite{paquin2020pqctls}.
    \item \textbf{Indirect Security--Accuracy Coupling:} Effective retrieval accuracy degrades via three operational mechanisms: (i) lossy edge vector quantization (\texttt{FP32} $\rightarrow$ \texttt{INT8}/1-bit) used to reduce PQC bandwidth overhead, (ii) rescaling noise in CKKS inner products coupled with the inability to traverse encrypted HNSW graphs on the server, and (iii) hard Service-Level Agreement (SLA) query timeouts under bursty arrivals.
    \item \textbf{Unmodeled 4D Cloud-Edge Economics and Energy:} Existing benchmarks evaluate PQC handshakes or standalone FHE search in isolation without jointly optimizing retrieval accuracy, tail latency, physical edge energy (mJ), and per-second cloud billing (USD).
\end{enumerate}

\subsection{Key Scientific Contributions}
\begin{itemize}
    \item \textbf{Empirical Cloud-Edge Bottleneck Characterization:} Real-execution characterization of NIST ML-KEM over QUIC~\cite{iyengar2021quic} and OpenFHE CKKS~\cite{badawi2022openfhe} across hybrid SQL and vector workloads~\cite{cooper2010ycsb, tpc_c_spec, jegou2011sift, chen2024bgem3} on x86\_64 and ARM64 nodes.
    \item \textbf{Constrained 4D Cost-Energy-Accuracy Formulation:} Exact unit-consistent mathematical formulation integrating wraparound-safe Intel RAPL hardware energy counters~\cite{khan2018rapl} and per-second Google Cloud Platform (GCP) compute/egress billing under security, deadline, and $\text{Recall@}K$ bounds.
    \item \textbf{SLA-Gated Contextual Bandit Crypto-Scheduler:} Regularized Disjoint LinUCB orchestrator~\cite{li2010linucb} guarded by a deterministic SLA feasibility gate across three cryptographic regimes.
    \item \textbf{Zero-Estimation SHA-256 Audited Evaluation:} End-to-end real execution protocol verified via cryptographic SHA-256 log manifests, two-sided Wilcoxon signed-rank tests with Holm--Bonferroni family-wise error control ($\alpha_{\text{FWER}} = 0.05$), bootstrap confidence intervals, and exact/continuity-corrected McNemar tests.
\end{itemize}

\section{Related Work and Positioning}
Table~\ref{tab:gap_analysis} contrasts \textbf{CAPQ-Pareto} against existing encrypted database and post-quantum networking paradigms.

\begin{table*}[t]
\centering
\caption{Comparison of Existing Cloud-Edge Encrypted Database Paradigms vs. Proposed CAPQ-Pareto Framework}
\label{tab:gap_analysis}
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{lcccccc}
\toprule
\textbf{Paradigm / Research Scope} & \textbf{Hybrid DB} & \textbf{NIST PQC} & \textbf{Zero-Trust} & \textbf{Tail-SLA \&} & \textbf{Hardware RAPL} & \textbf{Per-Second Cloud} \\
 & \textbf{(SQL + Vector)} & \textbf{QUIC/TLS} & \textbf{FHE Search} & \textbf{Recall@$K$ Gate} & \textbf{Energy Audit} & \textbf{Billing Model} \\
\midrule
Classical Cloud DB over TLS 1.3~\cite{wang2021milvus} & $\checkmark$ & $\times$ & $\times$ & $\times$ & $\times$ & Partial \\
Static PQC-TLS / QUIC Benchmarks~\cite{stebila2016liboqs, paquin2020pqctls} & $\times$ & $\checkmark$ & $\times$ & $\times$ & Partial & $\times$ \\
Standalone Encrypted ANN / FHE~\cite{cheon2017ckks, badawi2022openfhe} & $\times$ & $\checkmark$ & $\checkmark$ & Partial & $\times$ & $\times$ \\
Heuristic Edge-Cloud Task Offloading & Partial & $\times$ & $\times$ & $\checkmark$ & $\checkmark$ & $\checkmark$ \\
\midrule
\textbf{CAPQ-Pareto (Proposed Framework)} & $\checkmark$ & $\checkmark$ & $\checkmark$ & $\checkmark$ & $\checkmark$ & $\checkmark$ \\
\bottomrule
\end{tabular}
\end{table*}

\section{System Architecture and Threat Model}
\begin{figure*}[t]
\centering
\IfFileExists{00_manuscript/figures/workflow/capq_pareto_workflow.jpg}{%
    \includegraphics[width=0.96\textwidth]{capq_pareto_workflow.jpg}%
}{%
    \fbox{\parbox{0.94\textwidth}{\centering\vspace{1.2em}
    \textbf{[Figure 1 Placeholder: CAPQ-Pareto 4-Layer Architecture \& Q1 Guarantees Workflow]}\\
    \footnotesize Place \texttt{capq\_pareto\_workflow.jpg} in \texttt{00\_manuscript/figures/workflow/} to render automatically.
    \vspace{1.2em}}}%
}
\caption{Overview of the CAPQ-Pareto multi-regime post-quantum cryptographic routing and hybrid Cloud-Edge database workflow.}
\label{fig:architecture}
\end{figure*}

As shown in Fig.~\ref{fig:architecture}, \textbf{CAPQ-Pareto} comprises four layers: (1) Hybrid Workload \& Telemetry Layer, (2) Context-Aware Orchestration Fabric, (3) Multi-Regime Crypto Pipeline Layer, and (4) Cloud-Edge Hardware \& Hybrid DB Testbed. We address two adversary models: (i) a \textit{Quantum-Capable WAN Adversary} mitigated via hybrid X25519 + NIST ML-KEM key establishment; and (ii) an \textit{Honest-but-Curious Cloud Database Operator} mitigated in Regime~C by keeping CKKS secret keys exclusively at the Edge Gateway.

\section{Mathematical Problem Formulation}
Let $d_i = (\tau_i, s_i, \kappa_i, S_{\text{req}}(d_i), A_{\min}(d_i), T_{\max})$ denote incoming batch $i$, where $\tau_i \in \{0, 1\}$ encodes workload modality ($\text{SQL}=0, \text{VEC}=1$), $s_i$ is payload size in bytes, and $\kappa_i$ is vector dimensionality. Let $\mathcal{M}$ denote the discrete regime space $m = (\text{proto}, \text{kem}, \text{sym}, q, N_{\text{poly}}, b)$.

\subsection{Constrained 4D Pareto Optimization}
To prevent scale dominance and division-by-zero under degenerate calibration bounds, each metric $X \in \{L, E_{\text{edge}}, C_{\text{gcp}}\}$ is normalized with $\epsilon = 10^{-9}$:
\begin{equation}
\hat{X}(d_i, m) = \frac{X(d_i, m) - X_{\min}}{\max\!\left(X_{\max} - X_{\min},\, \epsilon\right)}.
\label{eq:norm}
\end{equation}
Given WAN state $\boldsymbol{\omega}_t = (\text{BW}_t, \text{RTT}_t, \text{Loss}_t)$ and arrival rate $\lambda_t$, CAPQ-Pareto minimizes:
\begin{equation}
\min_{m \in \mathcal{M}} \quad \mathcal{J}(d_i, m \mid \boldsymbol{\omega}_t) = \alpha \hat{L}(d_i, m) + \beta \hat{E}_{\text{edge}}(d_i, m) + \gamma \hat{C}_{\text{gcp}}(d_i, m)
\label{eq:objective}
\end{equation}
subject to $\alpha + \beta + \gamma = 1$ ($\alpha, \beta, \gamma \ge 0$) and hard feasibility constraints:
\begin{align}
S(m) &\ge S_{\text{req}}(d_i), \label{eq:c_sec} \\
A_{\text{eff}}(d_i, m) &= A_{\text{raw}}(d_i, m) \cdot \mathbb{I}\!\left[L(d_i, m) \le T_{\max}\right] \ge A_{\min}(d_i), \label{eq:c_acc} \\
L_{\text{p95}}(d_i, m) &\le T_{\max}. \label{eq:c_lat}
\end{align}

\subsection{Unit-Consistent Latency, RAPL Energy, and GCP Cost Models}
End-to-end latency (in milliseconds) decomposes into:
\begin{equation}
L(d_i, m) = T_{\text{prep}}^{\text{edge}} + T_{\text{crypto}}^{\text{edge}} + T_{\text{wan}}(s_i \cdot \eta(m), \boldsymbol{\omega}_t) + T_{\text{crypto}}^{\text{cloud}} + T_{\text{db}}.
\label{eq:latency}
\end{equation}

Physical edge energy counters $E_{\text{hw}}(t) = E_{\text{pkg}}(t) + E_{\text{dram}}(t) \in [0, E_{\max})$ sum package and DRAM rails while excluding redundant core/uncore sub-planes (\texttt{PP0}/\texttt{PP1}) to prevent double-counting bias. With unit factor $\kappa_E = 10^{-3}\text{ mJ}/\mu\text{J}$ (Linux \texttt{powercap}) or $\kappa_E = 3.6 \times 10^{-6}\text{ mJ}/\text{pWh}$ (Windows Kernel EMI RAPL) and calibrated idle power $P_{\text{idle\_mW}}$ across $\Delta t_{\text{s}} = t_1 - t_0$ seconds, wraparound-safe net edge energy (in millijoules) is:
\begin{equation}
\Delta E_{\text{raw}} = \begin{cases} E_{\text{hw}}(t_1) - E_{\text{hw}}(t_0), & E_{\text{hw}}(t_1) \ge E_{\text{hw}}(t_0), \\ \left(E_{\max} - E_{\text{hw}}(t_0)\right) + E_{\text{hw}}(t_1), & \text{otherwise}, \end{cases}
\label{eq:rapl_wrap}
\end{equation}
\begin{equation}
E_{\text{edge}}(d_i, m) = \max\!\left(0,\, \kappa_E \Delta E_{\text{raw}} - P_{\text{idle\_mW}} \cdot \Delta t_{\text{s}}\right).
\label{eq:energy}
\end{equation}

With cloud processing time $T_{\text{cloud\_ms}} = T_{\text{crypto}}^{\text{cloud}} + T_{\text{db}}$ (in milliseconds) and outbound payload $B_{\text{out\_bytes}}(d_i, m)$ (in bytes), per-query GCP cost (in USD) is:
\begin{equation}
C_{\text{gcp}}(d_i, m) = \left(p_{\text{vcpu}} n_{\text{vcpu}} + p_{\text{mem}} M_{\text{gb}}\right)\!\left(10^{-3} T_{\text{cloud\_ms}}\right) + p_{\text{egress}}\!\left(2^{-30} B_{\text{out\_bytes}}(d_i, m)\right).
\label{eq:cost}
\end{equation}

\section{Adaptive Crypto-Scheduler and Regimes}
\begin{itemize}
    \item \textbf{Regime A (Structured PQC-QUIC):} Lossless SQL/JSON batches over \texttt{QUIC 0-RTT} with \texttt{X25519\_ML-KEM-768}, \texttt{AES-256-GCM}, and adaptive batching $b \in \{1, 16, 64\}$.
    \item \textbf{Regime B (Unstructured Quantized Light-PQC):} Edge quantization (\texttt{FP32}, \texttt{INT8}, \texttt{1-bit}) over \texttt{QUIC + ML-KEM-512} paired with hardware-aware AEAD selection ($\{\texttt{AES-256-GCM}, \texttt{ChaCha20-Poly1305}\}$).
    \item \textbf{Regime C (Zero-Trust Coarse-Bucketed CKKS):} Edge K-Means coarse routing ($K_c$ centroids) paired with \texttt{OpenFHE CKKS} ($N_{\text{poly}} \in \{4096, 8192, 16384\}$) encrypted inner products over top-$r$ buckets.
\end{itemize}

\begin{algorithm}[t]
\caption{CAPQ-Pareto SLA-Gated Regularized LinUCB Scheduler}
\label{alg:scheduler}
\begin{algorithmic}[1]
\Require Weights $(\alpha, \beta, \gamma)$, exploration $\nu > 0$, ridge factor $\lambda_{\text{reg}} = 1.0$, action set $\mathcal{M}$, matrices $\{\mathbf{A}_m = \lambda_{\text{reg}}\mathbf{I}_{d_c}, \mathbf{b}_m = \mathbf{0}\}_{m \in \mathcal{M}}$ ($d_c = 10$)
\Ensure Optimal feasible regime $m^*_t$ for batch $d_t$
\For{each incoming hybrid batch $d_t$ under arrival rate $\lambda_t$}
    \State Build standardized context $\mathbf{x}_t \leftarrow [1, \tau_t, \tilde{s}_t, \tilde{\kappa}_t, \tilde{\lambda}_t, \widetilde{\text{BW}}_t, \widetilde{\text{RTT}}_t, \widetilde{\text{Loss}}_t, \alpha, \beta]^\top \in \mathbb{R}^{10}$
    \State $\mathcal{M}_{\text{sec}} \leftarrow \{m \in \mathcal{M} : S(m) \ge S_{\text{req}}(d_t)\}$
    \State $\mathcal{M}_{\text{sla}} \leftarrow \{m \in \mathcal{M}_{\text{sec}} : \hat{A}(d_t, m) \ge A_{\min} \land \hat{L}_{\text{p95}}(d_t, m) \le T_{\max}\}$
    \If{$\mathcal{M}_{\text{sla}} = \emptyset$}
        \State $\mathcal{M}_{\text{sla}} \leftarrow \{\arg\min_{m \in \mathcal{M}_{\text{sec}}} \hat{L}_{\text{p95}}(d_t, m)\}$
    \EndIf
    \For{each candidate $m \in \mathcal{M}_{\text{sla}}$}
        \State Solve $\mathbf{A}_m \hat{\boldsymbol{\theta}}_m = \mathbf{b}_m$; \quad $u_m \leftarrow \hat{\boldsymbol{\theta}}_m^\top \mathbf{x}_t - \nu \sqrt{\mathbf{x}_t^\top \mathbf{A}_m^{-1} \mathbf{x}_t}$
    \EndFor
    \State Dispatch $d_t$ via $m^*_t \leftarrow \arg\min_{m \in \mathcal{M}_{\text{sla}}} u_m$
    \State Measure real $(L_t, E_{\text{edge}, t}, C_{\text{gcp}, t}, A_{\text{eff}, t})$, compute $\mathcal{J}_t$, and update $\mathbf{A}_{m^*_t} \leftarrow \mathbf{A}_{m^*_t} + \mathbf{x}_t \mathbf{x}_t^\top$, $\mathbf{b}_{m^*_t} \leftarrow \mathbf{b}_{m^*_t} + \mathcal{J}_t \mathbf{x}_t$
\EndFor
\end{algorithmic}
\end{algorithm}

\section{Testbed Architecture and Zero-Estimation Statistical Protocol}
Evaluation adheres strictly to zero-estimation real execution across (1) Hardware-instrumented Edge containers (\texttt{2 vCPU, 2 GB RAM} quota on \texttt{Intel Core Ultra 9 185H}) with physical \texttt{Intel RAPL / EMI} hardware energy counters, and (2) Multi-Region GCP database instances (\texttt{e2-standard-4} x86\_64 and \texttt{t2a-standard-2} ARM64) across Bangkok, Singapore, and Iowa (Table~\ref{tab:testbed}). Every raw CSV log and generated \LaTeX{} result table is cryptographically bound via a SHA-256 verification manifest.

\textbf{Edge-Case-Safe Non-Parametric Protocol:} Each configuration executes $30$ independent measured trials after $50$ discarded warm-up queries. Continuous metrics are tested via paired two-sided Wilcoxon signed-rank tests (handling zero-difference pairs via Pratt's method and returning $p=1.0$ if all paired differences equal zero), adjusted via monotonic Holm--Bonferroni step-down correction ($p_{\text{Holm},(k)} = \max(p_{\text{Holm},(k-1)}, \min(1, (M-k+1)p_{(k)}))$ at $\alpha_{\text{FWER}} = 0.05$) alongside $10,000$-iteration bootstrap $95\%$ confidence intervals. Paired binary SLA compliance is tested via exact binomial McNemar tests when discordant pairs $n_{01}+n_{10} < 25$ (returning $p=1.0$ if $n_{01}+n_{10}=0$) and continuity-corrected McNemar $\chi^2$ otherwise.

\begin{table}[t]
\centering
\caption{IEEE TCC Real-Execution Benchmark Matrix}
\label{tab:testbed}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{ll}
\toprule
\textbf{Dimension} & \textbf{Real Execution Specification} \\
\midrule
\textbf{Crypto Stack} & OpenSSL 3.x + \texttt{oqs-provider} (\texttt{liboqs}), QUIC, OpenFHE v1.6+ \\
\textbf{Hybrid DBs (2)} & PostgreSQL 18 + \texttt{pgvector}; Milvus v2.6+ \\
\textbf{Structured Data} & TPC-C~\cite{tpc_c_spec} \& YCSB-A JSON ($256\text{ B}$--$4\text{ KB}$)~\cite{cooper2010ycsb} \\
\textbf{Vector Data} & SIFT1M ($128\text{d}$)~\cite{jegou2011sift} \& RAG BGE-M3 ($1024\text{d}$)~\cite{chen2024bgem3} \\
\textbf{Baselines (4)} & Classical TLS 1.3, Static ML-KEM-1024, Rule-Heuristic, \textbf{CAPQ-Pareto} \\
\textbf{Audit \& Integrity} & Real hardware telemetry + SHA-256 manifest per table ($\alpha_{\text{FWER}} = 0.05$) \\
\bottomrule
\end{tabular}%
}
\end{table}

\IfFileExists{00_manuscript/tables/tab_hw_calibration.tex}{\input{00_manuscript/tables/tab_hw_calibration.tex}}{}

\section*{Data Availability and Acknowledgment}
All source code, \texttt{experiment\_matrix.yaml}, raw telemetry logs, SHA-256 verification manifests, and \LaTeX{} table generators are maintained at \url{https://github.com/preecha-msu/CAPQ-Pareto}. This research project was financially supported by the Faculty of Informatics, Mahasarakham University. In accordance with IEEE policy, generative AI tools were utilized strictly for manuscript language polishing and \LaTeX{} formatting verification; all software execution, hardware measurements, and statistical analyses are real and cryptographically verified.

\bibliographystyle{IEEEtran}
\bibliography{references}

\begin{IEEEbiographynophoto}{Preecha Noiumkar}
(Member, IEEE) received the M.Sc. degree in information technology from the Faculty of Information Technology, King Mongkut's Institute of Technology North Bangkok (KMITNB), Bangkok, Thailand. He is currently a Lecturer and Researcher with the Department of Information Technology, Faculty of Informatics, Mahasarakham University, Maha Sarakham, Thailand. His research interests include machine learning, deep learning architecture design, cloud-edge systems security, post-quantum database optimization, and enterprise data engineering.
\end{IEEEbiographynophoto}

\end{document}
"""

SUPPLEMENT_TEX = r"""\documentclass[10pt,journal,compsoc]{IEEEtran}

\ifCLASSOPTIONcompsoc
  \usepackage[nocompress]{cite}
\else
  \usepackage{cite}
\fi

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs}
\usepackage[hidelinks]{hyperref}

\begin{document}

\title{Supplementary Material for ``CAPQ-Pareto: Context-Aware Post-Quantum and Homomorphic Encryption Orchestration for Hybrid Structured-Unstructured Databases Across the Cloud-Edge Continuum''}

\author{Preecha~Noiumkar,~\IEEEmembership{Member,~IEEE}%
\IEEEcompsocitemizethanks{\IEEEcompsocthanksitem P. Noiumkar is with the Department of Information Technology, Faculty of Informatics, Mahasarakham University, Maha Sarakham 44150, Thailand (e-mail: preecha.n@msu.ac.th; ORCID: \url{https://orcid.org/0009-0000-5964-4896}).}}

\maketitle

\section{Security and Entropy Leakage Bounds of Coarse-Bucketed CKKS (Regime C)}
In Regime~C, query embeddings $\mathbf{q} \in \mathbb{R}^\kappa$ are encrypted under the Ring-Learning-With-Errors (RLWE) assumption using CKKS across $N_{\text{poly}} \in \{4096, 8192, 16384\}$ strictly parameterized within Homomorphic Encryption Standard $\ge 128$-bit post-quantum security bounds~\cite{cheon2017ckks, badawi2022openfhe}: $Q_{4096}=[38,28,38]$ bits ($\sum=104 \le 109$, $\Delta=2^{28}$), $Q_{8192}=[60,40,60]$ bits ($\sum=160 \le 218$, $\Delta=2^{40}$), and $Q_{16384}=[60,50,50,60]$ bits ($\sum=220 \le 438$, $\Delta=2^{50}$). Under uniform cluster balance across $K_c$ edge centroids ($1 \le r \le K_c$), revealing $r$ candidate bucket indices yields worst-case Shannon entropy reduction bounded by:
\begin{equation}
\Delta H = \log_2\!\left(\frac{K_c}{r}\right) \text{ bits}, \quad \text{with } \Delta H = 0 \text{ when } r = K_c.
\end{equation}
For non-uniform bucket prior $P(B_j)$, exact mutual information leakage $I(Q; \mathcal{B}_r)$ is computed empirically from real dataset cluster distributions during execution.

\section{Edge-Case Statistical Formulations and SHA-256 Verification Protocol}
Let $d_k = X_k - Y_k$ ($k=1,\dots,N$) denote paired trial differences. If $\sum_{k=1}^N |d_k| = 0$, the two-sided Wilcoxon test deterministically returns $p = 1.0$. For $M$ pairwise baseline comparisons sorted as $p_{(1)} \le \dots \le p_{(M)}$, Holm--Bonferroni adjusted $p$-values enforce strict monotonicity:
\begin{equation}
p_{\text{Holm},(k)} = \max_{j \le k} \min\!\left(1.0,\, (M - j + 1)p_{(j)}\right).
\end{equation}
For paired binary SLA outcomes with discordant counts $(n_{01}, n_{10})$, $p_{\text{McNemar}} = 1.0$ if $n_{01}+n_{10}=0$, exact two-sided binomial probability if $0 < n_{01}+n_{10} < 25$, and continuity-corrected $\chi^2 = \frac{(|n_{01}-n_{10}|-1)^2}{n_{01}+n_{10}}$ if $n_{01}+n_{10} \ge 25$. Every table generated into \texttt{00\_manuscript/tables/} records the SHA-256 digest of its source raw CSV log.

\bibliographystyle{IEEEtran}
\bibliography{references}
\end{document}
"""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    os.chdir(REPO_ROOT)

    # 1. Write main.tex and supplement.tex directly (UTF-8 no BOM, no backups)
    (REPO_ROOT / "main.tex").write_text(MAIN_TEX, encoding="utf-8", newline="\n")
    (REPO_ROOT / "supplement.tex").write_text(SUPPLEMENT_TEX, encoding="utf-8", newline="\n")

    print(f"[SHA-256] main.tex       : {sha256_file(REPO_ROOT / 'main.tex')}")
    print(f"[SHA-256] supplement.tex : {sha256_file(REPO_ROOT / 'supplement.tex')}")

    # 2. Strictly overwrite D:\Research\CAPQ-Pareto-Overleaf-IEEE-TCC.zip on every .tex update
    drive_root = Path(ZIP_TARGET.anchor)
    if not drive_root.exists():
        raise FileNotFoundError(f"Drive {drive_root} is not accessible!")

    ZIP_TARGET.parent.mkdir(parents=True, exist_ok=True)
    if ZIP_TARGET.exists():
        ZIP_TARGET.unlink()

    include_files = ["main.tex", "supplement.tex", "references.bib", "README_OVERLEAF.md"]
    with zipfile.ZipFile(ZIP_TARGET, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in include_files:
            fpath = REPO_ROOT / fname
            if fpath.exists():
                zf.write(fpath, arcname=fname)
        manuscript_dir = REPO_ROOT / "00_manuscript"
        if manuscript_dir.exists():
            for root, _, files in os.walk(manuscript_dir):
                for file in sorted(files):
                    full_p = Path(root) / file
                    rel_p = full_p.relative_to(REPO_ROOT).as_posix()
                    zf.write(full_p, arcname=rel_p)

    if not ZIP_TARGET.is_file():
        raise RuntimeError(f"ZIP creation failed at {ZIP_TARGET}")

    print(f"[OK] Overwrote Overleaf ZIP : {ZIP_TARGET} ({ZIP_TARGET.stat().st_size:,} bytes)")
    print(f"[SHA-256] Overleaf ZIP      : {sha256_file(ZIP_TARGET)}")

    # 3. Replace on GitHub directly without backup files
    subprocess.run(["git", "add", "-A", "main.tex", "supplement.tex", "configs/", "00_manuscript/tables/", "06_experiments_results/sha256_manifest.json", "scripts/", "02_orchestrator/", "05_telemetry_audit/", "tests/"], check=True)
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
    if status.stdout.strip():
        subprocess.run(
            ["git", "commit", "-m", "fix(tex): enforce zero-estimation protocol, exact unit/RAPL equations, and edge-case stats"],
            check=True,
        )
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("[OK] Replaced .tex files on GitHub (origin/main).")
    else:
        print("[INFO] Git working tree already matches latest .tex content.")


if __name__ == "__main__":
    main()