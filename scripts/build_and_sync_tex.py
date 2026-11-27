from __future__ import annotations
import hashlib
import os
import re
import subprocess
import zipfile
from pathlib import Path

REPO_ROOT = Path(r"D:\Research\CAPQ-Pareto-V2")
ZIP_TARGET = Path(r"D:\Research\CAPQ-Pareto-Overleaf-IEEE-TCC.zip")

REFERENCES_BIB = r"""@techreport{nist_fips203_2024,
  author      = {{National Institute of Standards and Technology}},
  title       = {Module-Lattice-Based Key-Encapsulation Mechanism Standard},
  institution = {U.S. Department of Commerce},
  number      = {FIPS 203},
  year        = {2024},
  doi         = {10.6028/NIST.FIPS.203}
}

@inproceedings{stebila2016liboqs,
  author    = {Stebila, Douglas and Mosca, Michele},
  title     = {Post-quantum key exchange for the {Internet} and the {Open Quantum Safe} project},
  booktitle = {Selected Areas in Cryptography (SAC)},
  pages     = {14--37},
  year      = {2016},
  publisher = {Springer}
}

@inproceedings{paquin2020pqctls,
  author    = {Paquin, Christian and Stebila, Douglas and Tamvada, Goutam},
  title     = {Benchmarking post-quantum cryptography in {TLS}},
  booktitle = {International Conference on Post-Quantum Cryptography (PQCrypto)},
  pages     = {72--91},
  year      = {2020},
  publisher = {Springer}
}

@inproceedings{cheon2017ckks,
  author    = {Cheon, Jung Hee and Kim, Andrey and Kim, Miran and Song, Yongsoo},
  title     = {Homomorphic encryption for arithmetic of approximate numbers},
  booktitle = {Advances in Cryptology -- ASIACRYPT 2017},
  pages     = {409--437},
  year      = {2017},
  publisher = {Springer}
}

@inproceedings{badawi2022openfhe,
  author    = {Al Badawi, Ahmad and Bates, Jack and Bergamaschi, Flavio and Cousins, David Bruce and Erabelli, Saroja and Genise, Nicholas and Halevi, Shai and Hunt, Hamish and Kim, Andrey and Lee, Yongwoo and others},
  title     = {{OpenFHE}: Open-source fully homomorphic encryption library},
  booktitle = {Proceedings of the 10th Workshop on Encrypted Computing \& Applied Homomorphic Cryptography (WAHC)},
  pages     = {53--63},
  year      = {2022},
  publisher = {ACM}
}

@article{malkov2020hnsw,
  author  = {Malkov, Yu A. and Yashunin, Dmitry A.},
  title   = {Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {42},
  number  = {4},
  pages   = {824--836},
  year    = {2020}
}

@inproceedings{wang2021milvus,
  author    = {Wang, Jianguo and Yi, Xiaomeng and Guo, Rentong and Jin, Hai and Xu, Peng and Li, Shengjun and Wang, Xiangyu and Guo, Xiangzhou and Li, Chengming and Xu, Xiaohai and others},
  title     = {Milvus: A purpose-built vector data management system},
  booktitle = {Proceedings of the 2021 International Conference on Management of Data (SIGMOD)},
  pages     = {2614--2627},
  year      = {2021},
  publisher = {ACM}
}

@inproceedings{cooper2010ycsb,
  author    = {Cooper, Brian F. and Silberstein, Adam and Tam, Erwin and Ramakrishnan, Raghu and Sears, Russell},
  title     = {Benchmarking cloud serving systems with {YCSB}},
  booktitle = {Proceedings of the 1st ACM Symposium on Cloud Computing (SoCC)},
  pages     = {143--154},
  year      = {2010},
  publisher = {ACM}
}

@techreport{tpc_c_spec,
  author      = {{Transaction Processing Performance Council}},
  title       = {{TPC} Benchmark {C} Standard Specification, Revision 5.11},
  institution = {Transaction Processing Performance Council},
  year        = {2010}
}

@article{jegou2011sift,
  author  = {J{\'e}gou, Herv{\'e} and Douze, Matthijs and Schmid, Cordelia},
  title   = {Product quantization for nearest neighbor search},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {33},
  number  = {1},
  pages   = {117--128},
  year    = {2011}
}

@article{chen2024bgem3,
  author  = {Chen, Jianlv and Xiao, Shitao and Zhang, Peitian and Luo, Kun and Lian, Defu and Liu, Zheng},
  title   = {{BGE M3-Embedding}: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation},
  journal = {arXiv preprint arXiv:2402.03216},
  year    = {2024}
}

@techreport{iyengar2021quic,
  author      = {Iyengar, Jana and Thomson, Martin},
  title       = {{QUIC}: A {UDP}-Based Multiplexed and Secure Transport},
  institution = {Internet Engineering Task Force (IETF)},
  type        = {RFC},
  number      = {9000},
  year        = {2021}
}

@inproceedings{li2010linucb,
  author    = {Li, Lihong and Chu, Wei and Langford, John and Schapire, Robert E.},
  title     = {A contextual-bandit approach to personalized news article recommendation},
  booktitle = {Proceedings of the 19th International Conference on World Wide Web (WWW)},
  pages     = {661--670},
  year      = {2010},
  publisher = {ACM}
}

@article{khan2018rapl,
  author  = {Khan, Kashif Nizam and Hirki, Mikael and Niemi, Tapio and Nurminen, Jukka K. and Ou, Zhonghong},
  title   = {{RAPL} in action: Experiences in using {RAPL} for power measurements},
  journal = {ACM Transactions on Modeling and Performance Evaluation of Computing Systems (TOMPECS)},
  volume  = {3},
  number  = {2},
  pages   = {1--26},
  year    = {2018}
}
"""

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
\thanks{This research project was financially supported by the Faculty of Informatics, Mahasarakham University. Reproducible Artifacts and SHA-256 Verified Telemetry Logs: \url{https://github.com/preecha-msu/CAPQ-Pareto}.}}

\markboth{IEEE TRANSACTIONS ON CLOUD COMPUTING}%
{NOIUMKAR: CAPQ-PARETO ORCHESTRATION FOR HYBRID CLOUD-EDGE DATABASES}

\IEEEtitleabstractindextext{%
\begin{abstract}
Cloud-edge applications such as Retrieval-Augmented Generation and clinical analytics increasingly rely on hybrid databases that jointly serve lossless structured transactions and approximate vector similarity queries across wide-area networks. However, securing these pipelines against quantum adversaries introduces severe bottlenecks: standardized Post-Quantum Cryptography (NIST FIPS 203 ML-KEM) causes handshake expansion and packet fragmentation, whereas zero-trust vector evaluation via approximate Fully Homomorphic Encryption (CKKS) incurs substantial latency, edge energy drain, and cloud compute costs while precluding server-side graph traversal. We propose CAPQ-Pareto, a context-aware orchestration framework that dynamically routes hybrid database workloads across heterogeneous cryptographic and compression regimes. Joint encryption-quantization selection is formulated as a constrained multi-objective Pareto optimization problem minimizing tail latency, hardware-measured edge energy, and per-second cloud compute and egress costs under security, deadline, and retrieval accuracy bounds. An online SLA-gated Contextual Bandit scheduler dispatches queries across micro-batched QUIC 0-RTT with hybrid X25519/ML-KEM-768, precision-quantized vector transport over lightweight ML-KEM-512, and coarse-bucketed OpenFHE CKKS inner-product search. Across 720 real paired executions over kernel-emulated WAN links and hardware RAPL energy counters, CAPQ-Pareto reduces mean latency by 64.5 percent and edge energy by 71.0 percent versus static ML-KEM-1024 while raising joint SLA compliance from 33.3 percent to 74.9 percent.
\end{abstract}

\begin{IEEEkeywords}
Cloud-edge continuum, post-quantum cryptography, ML-KEM, homomorphic encryption, hybrid vector databases, multi-objective optimization, query processing, tail latency.
\end{IEEEkeywords}}

\maketitle
\IEEEdisplaynontitleabstractindextext
\IEEEpeerreviewmaketitle

\IEEEraisesectionheading{\section{Introduction and Problem Statement}\label{sec:introduction}}
\IEEEPARstart{C}{loud-edge} data infrastructures supporting enterprise Retrieval-Augmented Generation (RAG), clinical analytics, and industrial telemetry simultaneously execute transactional metadata filters (structured SQL/JSON requiring $100\%$ lossless relational integrity) and approximate nearest-neighbor (ANN) similarity searches over high-dimensional embeddings ($128$--$1024$ dimensions) across hybrid relational-vector engines such as PostgreSQL with \texttt{pgvector} and native vector stores~\cite{wang2021milvus, malkov2020hnsw}. Concurrently, protecting long-lived cloud-edge traffic against \textit{Harvest-Now, Decrypt-Later} quantum adversaries and honest-but-curious cloud database operators requires transitioning from classical elliptic-curve key exchange (X25519) to NIST Module-Lattice Key-Encapsulation Mechanism (ML-KEM)~\cite{nist_fips203_2024, stebila2016liboqs} and approximate Fully Homomorphic Encryption (CKKS)~\cite{cheon2017ckks, badawi2022openfhe}.

\subsection{Critical Cloud-Edge Bottlenecks}
Enforcing static cryptographic policies across hybrid database streams creates three severe bottlenecks:
\begin{enumerate}
    \item \textbf{WAN MTU Fragmentation vs. Sub-Millisecond CPU Lattice Speed:} While hardware-accelerated ML-KEM lattice operations execute in tens of microseconds on modern CPUs, static non-resumed ML-KEM-1024 handshakes transmit $3,136\text{ B}$ of key-exchange material---exceeding standard $1,500\text{ B}$ Ethernet Maximum Transmission Units (MTU) and triggering severe TCP retransmission tail-latency spikes under WAN packet loss~\cite{paquin2020pqctls}.
    \item \textbf{Indirect Security--Accuracy Coupling:} Effective retrieval accuracy degrades via three operational mechanisms: (i) aggressive edge vector quantization (\texttt{FP32} $\rightarrow$ \texttt{INT8}/\texttt{1-bit}) used to offset PQC bandwidth overhead, (ii) rescaling noise in CKKS homomorphic inner products coupled with the inability to traverse encrypted HNSW proximity graphs on the cloud, and (iii) hard Service-Level Agreement (SLA) query timeouts ($T_{\max} = 150\text{ ms}$).
    \item \textbf{Unmodeled 4D Cloud-Edge Energy and Economics:} Existing benchmarks evaluate PQC handshakes or standalone FHE search in isolation without jointly optimizing retrieval accuracy, tail latency, hardware-measured silicon edge energy (mJ), and per-second cloud compute/egress billing (USD).
\end{enumerate}

\subsection{Key Scientific Contributions}
\begin{itemize}
    \item \textbf{Empirical Cloud-Edge Bottleneck Discovery:} Using real kernel \texttt{tc/netem} sockets (\texttt{MTU = 1500 B}), compiled \texttt{liboqs 0.12.0}, and \texttt{OpenFHE 1.2.3}, we demonstrate that PQC bottlenecks stem from WAN MTU fragmentation rather than CPU lattice arithmetic, and uncover hardware AES-NI energy inversions against stream ciphers on modern hybrid edge silicon.
    \item \textbf{Constrained 4D Cost-Energy-Accuracy Formulation:} We formulate unit-consistent multi-objective optimization integrating wraparound-safe Intel RAPL/EMI silicon energy counters~\cite{khan2018rapl} (excluding redundant sub-plane double counting) and per-second cloud compute/egress billing under security, deadline, and $\text{Recall@}K$ constraints.
    \item \textbf{SLA-Gated Contextual Bandit Crypto-Scheduler:} We design a regularized Disjoint LinUCB orchestrator~\cite{li2010linucb} guarded by an empirical SLA feasibility gate across three specialized cryptographic regimes.
    \item \textbf{Zero-Estimation SHA-256 Verified Evaluation:} Across $720$ real paired executions over hybrid TPC-C, YCSB-A, SIFT-128d, and Wikipedia BGE-M3 (1024d) workloads~\cite{tpc_c_spec, cooper2010ycsb, jegou2011sift, chen2024bgem3}, CAPQ-Pareto achieves statistically significant improvements ($p_{\text{Holm}} \le 4.03 \times 10^{-29}$) with every raw CSV and \LaTeX{} table bound via SHA-256 manifests.
\end{itemize}

\section{Related Work and Positioning}
Table~\ref{tab:gap_analysis} contrasts \textbf{CAPQ-Pareto} against existing encrypted database and post-quantum transport paradigms.

\begin{table*}[t]
\centering
\caption{Comparison of Existing Cloud-Edge Encrypted Database Paradigms vs. Proposed CAPQ-Pareto Framework}
\label{tab:gap_analysis}
\renewcommand{\arraystretch}{1.18}
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
    \includegraphics[width=0.95\textwidth]{00_manuscript/figures/workflow/capq_pareto_workflow.jpg}%
}{%
    \fbox{\parbox{0.94\textwidth}{\centering\vspace{1.0em}
    \textbf{[Figure 1: CAPQ-Pareto 4-Layer Architecture Workflow]}\vspace{1.0em}}}%
}
\caption{Overview of the CAPQ-Pareto multi-regime post-quantum cryptographic routing and hybrid Cloud-Edge database workflow.}
\label{fig:architecture}
\end{figure*}

As illustrated in Fig.~\ref{fig:architecture}, \textbf{CAPQ-Pareto} comprises four layers: (1) Hybrid Workload \& Telemetry Layer, (2) Context-Aware Orchestration Fabric, (3) Multi-Regime Crypto Pipeline Layer, and (4) Cloud-Edge Hybrid Database Testbed. Two threat models are addressed: (i) a \textit{Quantum-Capable WAN Adversary} mitigated via hybrid X25519 + NIST ML-KEM key establishment; and (ii) an \textit{Honest-but-Curious Cloud Database Operator} mitigated in Regime~C by keeping CKKS secret keys strictly at the Edge Gateway while evaluating encrypted inner products over coarse-routed candidate buckets on the cloud.

\section{Mathematical Problem Formulation}
Let $d_i = (\tau_i, s_i, \kappa_i, S_{\text{req}}(d_i), A_{\min}(d_i), T_{\max})$ denote incoming batch $i$, where $\tau_i \in \{0, 1\}$ encodes workload modality ($\text{SQL}=0, \text{VEC}=1$), $s_i$ is payload size in bytes, and $\kappa_i$ is vector dimensionality. Let $\mathcal{M}$ denote the discrete regime space $m = (\text{proto}, \text{kem}, \text{sym}, q, N_{\text{poly}}, b)$.

\subsection{Constrained 4D Pareto Optimization}
Using calibration bounds $[X_{\min}, X_{\max}]$ locked strictly during warm-up with $\epsilon = 10^{-9}$ to prevent division-by-zero:
\begin{equation}
\hat{X}(d_i, m) = \frac{X(d_i, m) - X_{\min}}{\max\!\left(X_{\max} - X_{\min},\, \epsilon\right)}, \quad X \in \{L, E_{\text{edge}}, C_{\text{gcp}}\}.
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
To evaluate regimes that achieve low raw resource cost by violating accuracy or deadline bounds, we define the SLA-penalized Lagrangian score:
\begin{equation}
\mathcal{J}_{\text{SLA}}(d_i, m) = \mathcal{J}(d_i, m) + \left(1 - \mathbb{I}_{\text{SLA}}(d_i, m)\right),
\label{eq:j_sla}
\end{equation}
where $\mathbb{I}_{\text{SLA}}(d_i, m) = \mathbb{I}[S(m) \ge S_{\text{req}} \land A_{\text{eff}} \ge A_{\min} \land L \le T_{\max}]$.

\subsection{Unit-Consistent Latency, Silicon Energy, and Cloud Cost}
End-to-end latency (in ms) sums edge preparation, edge cryptography, WAN socket transport, batching queue delay, and cloud execution:
\begin{equation}
L(d_i, m) = T_{\text{prep}}^{\text{edge}} + T_{\text{crypto}}^{\text{edge}} + T_{\text{wan}}(s_i \cdot \eta(m), \boldsymbol{\omega}_t) + T_{\text{queue}}(b) + T_{\text{cloud}}.
\label{eq:latency}
\end{equation}

Physical edge energy counters $E_{\text{hw}}(t) = E_{\text{pkg}}(t) + E_{\text{dram}}(t) \in [0, E_{\max})$ sum package and DRAM rails while excluding redundant core/uncore sub-planes (\texttt{PP0}/\texttt{PP1}) to prevent double-counting bias. With unit factor $\kappa_E = 10^{-3}\text{ mJ}/\mu\text{J}$ (Linux \texttt{powercap}) or $\kappa_E = 3.6 \times 10^{-6}\text{ mJ}/\text{pWh}$ (Windows Kernel EMI RAPL) and quiescent idle power $P_{\text{idle\_mW}}$ across $\Delta t_{\text{s}} = t_1 - t_0$ seconds, wraparound-safe net edge energy (in mJ) is:
\begin{equation}
\Delta E_{\text{raw}} = \begin{cases} E_{\text{hw}}(t_1) - E_{\text{hw}}(t_0), & E_{\text{hw}}(t_1) \ge E_{\text{hw}}(t_0), \\ \left(E_{\max} - E_{\text{hw}}(t_0)\right) + E_{\text{hw}}(t_1), & \text{otherwise}, \end{cases}
\label{eq:rapl_wrap}
\end{equation}
\begin{equation}
E_{\text{edge}}(d_i, m) = \max\!\left(0,\, \kappa_E \Delta E_{\text{raw}} - P_{\text{idle\_mW}} \cdot \Delta t_{\text{s}}\right).
\label{eq:energy}
\end{equation}

With measured cloud processing time $T_{\text{cloud\_ms}} = T_{\text{crypto}}^{\text{cloud}} + T_{\text{db}}$ (in ms) and outbound bytes $B_{\text{out\_bytes}}(d_i, m)$, parameterized GCP per-query cost (in USD) is:
\begin{equation}
C_{\text{gcp}}(d_i, m) = \left(p_{\text{vcpu}} n_{\text{vcpu}} + p_{\text{mem}} M_{\text{gb}}\right)\!\left(10^{-3} T_{\text{cloud\_ms}}\right) + p_{\text{egress}}\!\left(2^{-30} B_{\text{out\_bytes}}(d_i, m)\right).
\label{eq:cost}
\end{equation}

\section{Adaptive Crypto-Scheduler and Regimes}
\begin{itemize}
    \item \textbf{Regime A (Structured PQC-QUIC):} Lossless SQL/JSON batches over \texttt{QUIC 0-RTT} with hybrid \texttt{X25519\_ML-KEM-768}, \texttt{AES-256-GCM}, and adaptive batching $b \in \{1, 16\}$.
    \item \textbf{Regime B (Unstructured Quantized Light-PQC):} Edge vector quantization (\texttt{FP32}, $\epsilon$-safe asymmetric \texttt{INT8}, or optimal $L_1$-scaled \texttt{1-bit} $\gamma^* = \|\mathbf{v}\|_1/\kappa$) over \texttt{QUIC + ML-KEM-512} paired with hardware-aware AEAD ($\{\texttt{AES-256-GCM}, \texttt{ChaCha20-Poly1305}\}$).
    \item \textbf{Regime C (Zero-Trust Coarse-Bucketed CKKS):} Edge K-Means coarse routing paired with \texttt{OpenFHE 1.2.3 CKKS} (\texttt{HEStd\_128\_classic}, $N_{\text{poly}} \in \{4096, 8192, 16384\}$) encrypted inner-product evaluation over candidate buckets.
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
    \State Measure real $(L_t, E_{\text{edge}, t}, C_{\text{gcp}, t}, A_{\text{eff}, t})$, compute $\mathcal{J}_{\text{SLA}, t}$, and update $\mathbf{A}_{m^*_t} \leftarrow \mathbf{A}_{m^*_t} + \mathbf{x}_t \mathbf{x}_t^\top$, $\mathbf{b}_{m^*_t} \leftarrow \mathbf{b}_{m^*_t} + \mathcal{J}_{\text{SLA}, t} \mathbf{x}_t$
\EndFor
\end{algorithmic}
\end{algorithm}

\section{Testbed Setup and Experimental Results}
\subsection{Real-Execution Testbed and Statistical Protocol}
All experiments enforce strict zero-estimation real execution (Table~\ref{tab:testbed}) on an \texttt{Intel Core Ultra 9 185H} host instrumented via kernel C-API \texttt{pdh.dll} (\texttt{windows\_emi\_pdh\_rapl}, $46.95\,\mu\text{s}$ read overhead) paired with isolated Linux network namespaces (\texttt{unshare -n}, \texttt{tc/netem}, \texttt{MTU = 1500 B}) running compiled \texttt{liboqs 0.12.0}, \texttt{OpenFHE 1.2.3}, \texttt{aioquic}, \texttt{PostgreSQL 18.6} (\texttt{pgvector 0.8.1}), and \texttt{Qdrant} native HNSW. Edge RAPL windows strictly isolate edge cryptography and WAN socket transmission from cloud database/FHE evaluation ($T_{\text{cloud\_ms}}$), which is priced using published GCP \texttt{e2-standard-4} per-second compute and Asia internet egress rates.

\begin{table}[t]
\centering
\caption{IEEE TCC Verified Real-Execution Benchmark Stack}
\label{tab:testbed}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{ll}
\toprule
\textbf{Dimension} & \textbf{Verified Real Execution Specification} \\
\midrule
\textbf{Crypto Stack} & OpenSSL 3.x + \texttt{liboqs 0.12.0} (ML-KEM), \texttt{aioquic}, \texttt{OpenFHE 1.2.3} \\
\textbf{Hybrid DBs (2)} & PostgreSQL 18.6 (\texttt{pgvector 0.8.1}) \& Qdrant Native HNSW \\
\textbf{Structured Data} & TPC-C Rev.~5.11~\cite{tpc_c_spec} \& Exact-Byte YCSB-A ($256\text{--}4096\text{ B}$)~\cite{cooper2010ycsb} \\
\textbf{Vector Data} & Official SIFT ($128\text{d}$)~\cite{jegou2011sift} \& Wikipedia BGE-M3 ($1024\text{d}$)~\cite{chen2024bgem3} \\
\textbf{WAN Profiles (3)} & \texttt{5g\_mec\_good} (12ms), \texttt{wan\_congested} (65ms, 1.5\% loss), \texttt{remote\_iot} (140ms, 3\% loss) \\
\textbf{Audit Protocol} & $720$ core paired runs ($30$ trials/config after $50$-query warm-up) + SHA-256 manifests \\
\bottomrule
\end{tabular}%
}
\end{table}

\IfFileExists{00_manuscript/tables/tab_hw_calibration.tex}{\input{00_manuscript/tables/tab_hw_calibration.tex}}{}

\subsection{Empirical Hardware Calibration and WAN Bottlenecks}
Table~\ref{tab:hw_calib} reveals a critical hardware-software interaction on \texttt{Intel Core Ultra 9 185H}: hardware-accelerated \texttt{AES-256-GCM} consumes $2.15\times$ less net silicon energy ($42.56\,\mu\text{J/op}$ vs. $91.56\,\mu\text{J/op}$, Wilcoxon $p = 0.0020$) than \texttt{ChaCha20-Poly1305}, justifying hardware-aware cipher selection in Regime~B rather than assuming stream ciphers are universally superior on edge devices.

\begin{figure}[t]
\centering
\IfFileExists{00_manuscript/figures/pareto_4d_plots/fig2_wan_fragmentation_bottleneck.pdf}{%
    \includegraphics[width=\columnwidth]{00_manuscript/figures/pareto_4d_plots/fig2_wan_fragmentation_bottleneck.pdf}%
}{}
\caption{Kernel \texttt{tc/netem} (\texttt{MTU = 1500 B}) socket audit demonstrating TCP/TLS 1.3 + ML-KEM-1024 fragmentation tail spikes ($1,038.4\text{ ms}$ p95) vs. batched QUIC 0-RTT ($p_{\text{Holm}} = 0.0015$).}
\label{fig:wan_bottleneck}
\end{figure}

Furthermore, while C-compiled \texttt{ML-KEM-1024} executes in $0.037\text{ ms}$ ($0.61\text{ mJ}$) in CPU memory, Fig.~\ref{fig:wan_bottleneck} proves that transmitting its $3,136\text{ B}$ handshake over a $1,500\text{ B}$ MTU socket inflates mean latency to $206.86\text{ ms}$ under $1.5\%$ WAN loss and p95 latency to $1,038.40\text{ ms}$ under $3.0\%$ constrained loss, whereas Regime~A's resumed QUIC 0-RTT pipeline achieves $42.84\text{ ms}$ and $80.67\text{ ms}$ amortized per-transaction time ($p_{\text{Holm}} = 0.0015$).

\IfFileExists{00_manuscript/tables/tab_core_720_results.tex}{\input{00_manuscript/tables/tab_core_720_results.tex}}{}

\subsection{Core 720-Run Multi-Objective Evaluation}
Table~\ref{tab:core_720_results}, Fig.~\ref{fig:pareto_4d}, and Fig.~\ref{fig:profile_breakdown} present the 720-run paired benchmark across all three WAN profiles and two hybrid workload mixes ($N = 180$ paired trials per orchestrator, quiescent $P_{\text{idle}} = 4,203.1\text{ mW}$):
\begin{itemize}
    \item \textbf{Overcoming Static PQC Fragmentation:} Both \texttt{Classical\_TLS1.3} and \texttt{Static\_MLKEM1024} suffer severe TCP handshake stalls on congested and remote WAN profiles, yielding $33.3\%$ SLA pass rates. \textbf{CAPQ-Pareto} reduces mean end-to-end latency by $64.5\%$ ($98.44\text{ ms}$ vs. $277.15\text{ ms}$) and net edge energy by $71.0\%$ ($33.57\text{ mJ}$ vs. $115.88\text{ mJ}$) compared to \texttt{Static\_MLKEM1024} ($p_{\text{Holm}} = 4.03 \times 10^{-29}$), while keeping mean p95 latency ($149.06\text{ ms}$) within the $150\text{ ms}$ hard deadline.
    \item \textbf{Avoiding the Infeasible Cheapness Trap:} Although context-blind \texttt{Rule\_Heuristic} attains a slightly lower unpenalized resource score ($\mathcal{J} = 0.060$ vs. $0.063$) by aggressively forcing \texttt{1-bit} quantization, it violates the $\text{Recall@10} \ge 0.92$ bound on vector queries, collapsing effective accuracy to $0.323$ and SLA compliance to $32.0\%$. By gating actions through empirical feasibility bounds, \textbf{CAPQ-Pareto} elevates effective accuracy to $0.762$ ($+135.5\%$, $p_{\text{Holm}} = 1.71 \times 10^{-30}$) and achieves the lowest SLA-penalized objective $\mathcal{J}_{\text{SLA}} = \mathbf{0.314}$ versus $0.740$--$0.893$ across all baselines ($p_{\text{Holm}} \le 4.03 \times 10^{-29}$; McNemar $p_{\text{Holm}} \le 6.50 \times 10^{-11}$).
\end{itemize}

\begin{figure}[t]
\centering
\IfFileExists{00_manuscript/figures/pareto_4d_plots/fig3_pareto_4d_tradeoff.pdf}{%
    \includegraphics[width=\columnwidth]{00_manuscript/figures/pareto_4d_plots/fig3_pareto_4d_tradeoff.pdf}%
}{}
\caption{4D Cloud-Edge Pareto trade-off across 720 real runs showing Latency, Silicon Edge Energy, GCP Cost (marker size), and Effective Accuracy (color).}
\label{fig:pareto_4d}
\end{figure}

\begin{figure}[t]
\centering
\IfFileExists{00_manuscript/figures/pareto_4d_plots/fig4_profile_breakdown.pdf}{%
    \includegraphics[width=\columnwidth]{00_manuscript/figures/pareto_4d_plots/fig4_profile_breakdown.pdf}%
}{}
\caption{Per-profile mean latency and joint security-accuracy-deadline SLA compliance across \texttt{5g\_mec\_good}, \texttt{wan\_congested}, and \texttt{remote\_iot\_constrained}.}
\label{fig:profile_breakdown}
\end{figure}

\subsection{Cross-Database BGE-M3 ($1024\text{d}$) and Shock Ablation}
Table~\ref{tab:ablation_crossdb}, Fig.~\ref{fig:shock_weights}, and Fig.~\ref{fig:crossdb_bgem3} evaluate $360$ additional real runs spanning high-dimensional Wikipedia BGE-M3 ($1024\text{d}$) retrieval across \texttt{PostgreSQL 18.6} (\texttt{pgvector 0.8.1}) and \texttt{Qdrant} native HNSW, Pareto weight sensitivity, and sudden WAN degradation shocks ($12\text{ ms} \rightarrow 65\text{ ms} \rightarrow 140\text{ ms}$ RTT). Compressing $1024\text{d}$ vectors via $\epsilon$-safe \texttt{INT8-SQ} shrinks wire payloads by $74.8\%$ ($4,096\text{ B} \rightarrow 1,032\text{ B}$, fitting within a single $1,500\text{ B}$ MTU frame) while preserving $\text{Recall@10} = 0.9513$ on PostgreSQL and $0.9927$ on Qdrant ($\ge 0.92$ SLA), whereas \texttt{1-bit} quantization drops $\text{Recall@10}$ to $0.6983$--$0.7077$ and is properly pruned by CAPQ-Pareto's SLA gate.

\IfFileExists{00_manuscript/tables/tab_ablation_crossdb.tex}{\input{00_manuscript/tables/tab_ablation_crossdb.tex}}{}

\begin{figure}[t]
\centering
\IfFileExists{00_manuscript/figures/pareto_4d_plots/fig5_ablation_weights_and_shock.pdf}{%
    \includegraphics[width=\columnwidth]{00_manuscript/figures/pareto_4d_plots/fig5_ablation_weights_and_shock.pdf}%
}{}
\caption{Online trajectory adaptation under sudden WAN latency/loss shocks and 4-profile Pareto weight sensitivity.}
\label{fig:shock_weights}
\end{figure}

\begin{figure}[t]
\centering
\IfFileExists{00_manuscript/figures/pareto_4d_plots/fig6_crossdb_bgem3_1024d.pdf}{%
    \includegraphics[width=\columnwidth]{00_manuscript/figures/pareto_4d_plots/fig6_crossdb_bgem3_1024d.pdf}%
}{}
\caption{Cross-database BGE-M3 ($1024\text{d}$) $\text{Recall@10}$ and query latency across PostgreSQL 18.6 (\texttt{pgvector 0.8.1}) and Qdrant native HNSW.}
\label{fig:crossdb_bgem3}
\end{figure}

\subsection{Limitations and Validity Considerations}
To uphold strict scientific transparency, two scope boundaries are noted: (1) WAN propagation, bandwidth caps, and packet loss are enforced via isolated Linux kernel \texttt{tc/netem} namespaces (\texttt{MTU = 1500 B}), while cloud expenditure is computed analytically from real measured database/OpenFHE execution durations and wire bytes using published GCP \texttt{e2-standard-4} unit rates rather than live transcontinental VM billing invoices; and (2) under the $140\text{ ms}$ RTT constrained IoT link, zero-trust CKKS queries inevitably exceed the $150\text{ ms}$ deadline due to physical $1\text{-RTT}$ propagation plus homomorphic evaluation ($> 68\text{ ms}$), accounting for the remaining $25.1\%$ SLA misses in extreme WAN regimes. Extending deployment to physical ARM64 edge boards and hardware FHE accelerators represents promising future work.

\section{Conclusion}
We presented \textbf{CAPQ-Pareto}, a context-aware post-quantum and homomorphic encryption orchestrator for hybrid relational-vector cloud-edge databases. By unifying QUIC 0-RTT hybrid ML-KEM transport, precision-gated vector quantization, and coarse-bucketed OpenFHE CKKS evaluation under an SLA-gated regularized LinUCB bandit, CAPQ-Pareto eliminates PQC MTU fragmentation stalls while preserving retrieval accuracy and post-quantum confidentiality.

\section*{Data Availability and Acknowledgment}
All source code, \texttt{experiment\_matrix.yaml}, raw CSV logs, \LaTeX{} table/figure generators, and SHA-256 verification manifests are publicly available at \url{https://github.com/preecha-msu/CAPQ-Pareto}. This research project was financially supported by the Faculty of Informatics, Mahasarakham University. In accordance with IEEE policy, generative AI tools were utilized strictly for manuscript language polishing and \LaTeX{} formatting verification; all software execution, hardware RAPL/EMI measurements, and statistical analyses are real and cryptographically verified.

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
\usepackage{graphicx}
\usepackage[hidelinks]{hyperref}

\begin{document}

\title{Supplementary Material for ``CAPQ-Pareto: Context-Aware Post-Quantum and Homomorphic Encryption Orchestration for Hybrid Structured-Unstructured Databases Across the Cloud-Edge Continuum''}

\author{Preecha~Noiumkar,~\IEEEmembership{Member,~IEEE}%
\IEEEcompsocitemizethanks{\IEEEcompsocthanksitem P. Noiumkar is with the Department of Information Technology, Faculty of Informatics, Mahasarakham University, Maha Sarakham 44150, Thailand (e-mail: preecha.n@msu.ac.th; ORCID: \url{https://orcid.org/0009-0000-5964-4896}).}}

\maketitle

\section{Security and Empirical Entropy Bounds of Coarse-Bucketed CKKS (Regime C)}
In Regime~C, query embeddings $\mathbf{q} \in \mathbb{R}^\kappa$ are encrypted under the Ring-Learning-With-Errors (RLWE) assumption using \texttt{OpenFHE 1.2.3} CKKS strictly enforcing \texttt{HEStd\_128\_classic} ($\ge 128$-bit security~\cite{cheon2017ckks, badawi2022openfhe}): $Q_{4096}=[38,30]$ bits (\texttt{BV} key-switching, $\sum=68 \le 109$), $Q_{8192}=[60,40,60]$ bits, and $Q_{16384}=[60,50,50,60]$ bits. Given empirical cluster prior $p_c = |B_c|/N_{\text{db}}$ with $H(\mathcal{B}) = -\sum_{c=1}^{K_c} p_c \log_2 p_c$ ($4.9585\text{ bits}$ on SIFT-128d and $4.8970\text{ bits}$ on Wikipedia BGE-M3 for $K_c=32$), probing $r_{\text{eff}}$ buckets bounds coarse access-pattern leakage to:
\begin{equation}
\Delta H_{\text{emp}}(r_{\text{eff}}) = \max\!\left(0,\, H(\mathcal{B}) - \log_2(r_{\text{eff}})\right) \text{ bits}.
\end{equation}

\section{Edge-Case Statistical Formulations and SHA-256 Verification Tables}
Let $d_k = X_k - Y_k$ ($k=1,\dots,N$) denote paired trial differences. If $\sum_{k=1}^N |d_k| \le 10^{-15}$, two-sided Wilcoxon deterministically returns $p = 1.0$; otherwise Pratt's zero-handling method is applied. Sorted raw $p$-values $p_{(1)} \le \dots \le p_{(M)}$ undergo monotonic Holm--Bonferroni adjustment $p_{\text{Holm},(k)} = \max_{j \le k} \min(1.0, (M - j + 1)p_{(j)})$. Paired binary SLA outcomes use exact two-sided binomial probability when $0 < n_{01}+n_{10} < 25$ ($p=1.0$ if $n_{01}+n_{10}=0$) and continuity-corrected $\chi^2 = \frac{(|n_{01}-n_{10}|-1)^2}{n_{01}+n_{10}}$ when $n_{01}+n_{10} \ge 25$. Tables~\ref{tab:stack_audit}--\ref{tab:wan_socket_audit} document all SHA-256 verified primitive, dataset, OpenFHE/Qdrant, and kernel \texttt{tc/netem} socket audits.

\IfFileExists{00_manuscript/tables/tab_stack_smoke_audit.tex}{\input{00_manuscript/tables/tab_stack_smoke_audit.tex}}{}
\IfFileExists{00_manuscript/tables/tab_openfhe_qdrant_audit.tex}{\input{00_manuscript/tables/tab_openfhe_qdrant_audit.tex}}{}
\IfFileExists{00_manuscript/tables/tab_datasets_manifest.tex}{\input{00_manuscript/tables/tab_datasets_manifest.tex}}{}
\IfFileExists{00_manuscript/tables/tab_wan_socket_audit.tex}{\input{00_manuscript/tables/tab_wan_socket_audit.tex}}{}

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


def verify_compsoc_abstract(tex_src: str) -> int:
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex_src, re.S)
    if not m:
        raise RuntimeError("Missing abstract in main.tex!")
    body = m.group(1).strip()
    if "\\" in body or "$" in body:
        raise RuntimeError("IEEE compsoc violation: LaTeX macros or math mode found inside plain-text abstract!")
    words = body.split()
    if not (100 <= len(words) <= 200):
        raise RuntimeError(f"IEEE compsoc abstract word count ({len(words)}) outside [100, 200]!")
    return len(words)


def main() -> None:
    os.chdir(REPO_ROOT)
    wc = verify_compsoc_abstract(MAIN_TEX)
    print(f"[AUDIT] IEEE TCC compsoc plain-text abstract verified: {wc} words (strictly within 100--200).")

    (REPO_ROOT / "references.bib").write_text(REFERENCES_BIB, encoding="utf-8", newline="\n")
    (REPO_ROOT / "main.tex").write_text(MAIN_TEX, encoding="utf-8", newline="\n")
    (REPO_ROOT / "supplement.tex").write_text(SUPPLEMENT_TEX, encoding="utf-8", newline="\n")

    print(f"[SHA-256] main.tex       : {sha256_file(REPO_ROOT / 'main.tex')}")
    print(f"[SHA-256] supplement.tex : {sha256_file(REPO_ROOT / 'supplement.tex')}")
    print(f"[SHA-256] references.bib : {sha256_file(REPO_ROOT / 'references.bib')}")

    # Overwrite D:\Research\CAPQ-Pareto-Overleaf-IEEE-TCC.zip on every update
    ZIP_TARGET.parent.mkdir(parents=True, exist_ok=True)
    if ZIP_TARGET.exists():
        ZIP_TARGET.unlink()

    include_files = ["main.tex", "supplement.tex", "references.bib", "WORKFLOW_CHECKLIST.md", "README_OVERLEAF.md"]
    with zipfile.ZipFile(ZIP_TARGET, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in include_files:
            fpath = REPO_ROOT / fname
            if fpath.exists():
                zf.write(fpath, arcname=fname)
        manuscript_dir = REPO_ROOT / "00_manuscript"
        for root, _, files in os.walk(manuscript_dir):
            for file in sorted(files):
                full_p = Path(root) / file
                rel_p = full_p.relative_to(REPO_ROOT).as_posix()
                zf.write(full_p, arcname=rel_p)

    print(f"[OK] Overwrote Overleaf ZIP : {ZIP_TARGET} ({ZIP_TARGET.stat().st_size:,} bytes)")
    print(f"[SHA-256] Overleaf ZIP      : {sha256_file(ZIP_TARGET)}")

    # Replace directly on GitHub without backup files
    subprocess.run(
        [
            "git", "add", "-A",
            ".gitattributes", ".gitignore", "main.tex", "supplement.tex", "references.bib", "WORKFLOW_CHECKLIST.md",
            "configs/", "00_manuscript/", "01_workloads/", "02_orchestrator/",
            "05_telemetry_audit/", "06_experiments_results/plotting_scripts/",
            "06_experiments_results/sha256_manifest.json", "scripts/", "tests/",
        ],
        check=True,
    )
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
    if status.stdout.strip():
        subprocess.run(
            ["git", "commit", "-m", "feat(paper): integrate 720-run real SHA-256 results, 4D Pareto PDFs, and full IEEE TCC manuscript"],
            check=True,
        )
        subprocess.run(["git", "push", "-u", "origin", "v2"], check=True)
        print("[OK] Replaced .tex, figures, tables, and SHA-256 manifest on GitHub (origin/main).")
    else:
        print("[INFO] Git working tree already up to date.")


if __name__ == "__main__":
    main()