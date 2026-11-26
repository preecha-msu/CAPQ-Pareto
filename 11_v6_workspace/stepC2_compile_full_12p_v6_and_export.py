from __future__ import annotations
import hashlib
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(r"D:\Research\CAPQ-Pareto-V6")
V4_MS = ROOT / "08_manuscript_ieee_tcc_v6"
FIG = V4_MS / "figures"
TAB = V4_MS / "tables"


def sha64(fp: Path) -> str:
    d = hashlib.sha256(fp.read_bytes()).hexdigest()
    assert len(d) == 64 and all(c in "0123456789abcdef" for c in d)
    return d


BIB_FULL = r"""@inproceedings{russac2019dlinucb,
  author    = {Yoan Russac and Claire Vernade and Olivier Capp{\'e}},
  title     = {Weighted Linear Bandits for Non-Stationary Environments},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {32},
  pages     = {12040--12049},
  year      = {2019}
}
@inproceedings{henzinger2023tiptoe,
  author    = {Alexandra Henzinger and Emma Dauterman and Henry Corrigan-Gibbs and Nickolai Zeldovich},
  title     = {Private Web Search with {Tiptoe}},
  booktitle = {Proceedings of the 29th ACM Symposium on Operating Systems Principles (SOSP)},
  pages     = {396--416},
  year      = {2023}
}
@inproceedings{chen2021spann,
  author    = {Qi Chen and Bing Zhao and Haidong Wang and Mingqin Li and Chuanjie Liu and Zengzhong Li and Mao Yang and Jingdong Wang},
  title     = {{SPANN}: Highly-efficient Billion-scale Approximate Nearest Neighbor Search},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {34},
  pages     = {5199--5212},
  year      = {2021}
}
@inproceedings{cheon2017ckks,
  author    = {Jung Hee Cheon and Andrey Kim and Miran Kim and Yongsoo Song},
  title     = {Homomorphic Encryption for Arithmetic of Approximate Numbers},
  booktitle = {Advances in Cryptology -- ASIACRYPT},
  pages     = {409--437},
  year      = {2017}
}
@article{albadawi2022openfhe,
  author  = {Ahmad Al Badawi and Jack Bates and Flavio Bergamaschi and David Bruce Cousins and others},
  title   = {{OpenFHE}: Open-Source Fully Homomorphic Encryption Library},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems (TCHES)},
  year    = {2022}
}
@misc{fips203,
  author       = {{National Institute of Standards and Technology}},
  title        = {Module-Lattice-Based Key-Encapsulation Mechanism Standard ({FIPS} 203)},
  howpublished = {U.S. Department of Commerce},
  year         = {2024}
}
@misc{fips204,
  author       = {{National Institute of Standards and Technology}},
  title        = {Module-Lattice-Based Digital Signature Standard ({FIPS} 204)},
  howpublished = {U.S. Department of Commerce},
  year         = {2024}
}
@misc{rfc8879,
  author       = {Alessandro Ghedini and Victor Vasiliev},
  title        = {{TLS} Certificate Compression ({RFC} 8879)},
  howpublished = {Internet Engineering Task Force (IETF)},
  year         = {2020}
}
@misc{rfc9000,
  author       = {Jana Iyengar and Martin Thomson},
  title        = {{QUIC}: A {UDP}-Based Multiplexed and Secure Transport ({RFC} 9000)},
  howpublished = {Internet Engineering Task Force (IETF)},
  year         = {2021}
}
@misc{rfc9002,
  author       = {Jana Iyengar and Ian Swett},
  title        = {{QUIC} Loss Detection and Congestion Control ({RFC} 9002)},
  howpublished = {Internet Engineering Task Force (IETF)},
  year         = {2021}
}
@inproceedings{sikeridis2020pqtls,
  author    = {Dimitrios Sikeridis and Panos Kampanakis and Michael Devetsikiotis},
  title     = {Post-Quantum Authentication in {TLS} 1.3: A Performance Study},
  booktitle = {Network and Distributed System Security Symposium (NDSS)},
  year      = {2020}
}
@misc{bas2024cloudflarepq,
  author       = {Bas Westerbaan and Cefan Daniel Rubin},
  title        = {Defending Against Future Threats: Cloudflare's Post-Quantum {TLS} Deployment at Internet Scale},
  howpublished = {Cloudflare Research Technical Report / IETF PQUIP},
  year         = {2024}
}
@article{malkov2018hnsw,
  author  = {Yu A. Malkov and D. A. Yashunin},
  title   = {Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {42},
  number  = {4},
  pages   = {824--836},
  year    = {2020}
}
@inproceedings{wang2021milvus,
  author    = {Jianguo Wang and Xiaomeng Yi and Rentong Guo and Hai Jin and Peng Xu and others},
  title     = {Milvus: A Purpose-Built Vector Data Management System},
  booktitle = {Proceedings of the ACM SIGMOD International Conference on Management of Data},
  pages     = {2614--2627},
  year      = {2021}
}
@inproceedings{cooper2010ycsb,
  author    = {Brian F. Cooper and Adam Silberstein and Erwin Tam and Raghu Ramakrishnan and Russell Sears},
  title     = {Benchmarking Cloud Serving Systems with {YCSB}},
  booktitle = {Proceedings of the ACM Symposium on Cloud Computing (SoCC)},
  pages     = {143--154},
  year      = {2010}
}
@article{jegou2011sift,
  author  = {Herv{\'e} J{\'e}gou and Matthijs Douze and Cordelia Schmid},
  title   = {Product Quantization for Nearest Neighbor Search},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {33},
  number  = {1},
  pages   = {117--128},
  year    = {2011}
}
@article{chen2024bgem3,
  author  = {Jianlv Chen and Shitao Xiao and Peitian Zhang and Kun Luo and Defu Lian and Zheng Liu},
  title   = {{BGE M3-Embedding}: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings},
  journal = {arXiv preprint arXiv:2402.03216},
  year    = {2024}
}
@inproceedings{li2010linucb,
  author    = {Lihong Li and Wei Chu and John Langford and Robert E. Schapire},
  title     = {A Contextual-Bandit Approach to Personalized News Article Recommendation},
  booktitle = {Proceedings of the 19th International Conference on World Wide Web (WWW)},
  pages     = {661--670},
  year      = {2010}
}
@inproceedings{garivier2011ucb,
  author    = {Aur{\'e}lien Garivier and Eric Moulines},
  title     = {On Upper-Confidence Bound Policies for Switching Bandit Problems},
  booktitle = {Algorithmic Learning Theory (ALT)},
  pages     = {174--188},
  year      = {2011}
}
@inproceedings{lattimore2016causal,
  author    = {Finnian Lattimore and Tor Lattimore and Mark D. Reid},
  title     = {Causal Bandits: Learning Good Interventions via Causal Inference},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {29},
  pages     = {1181--1189},
  year      = {2016}
}
@article{paninski2003entropy,
  author  = {Liam Paninski},
  title   = {Estimation of Entropy and Mutual Information},
  journal = {Neural Computation},
  volume  = {15},
  number  = {6},
  pages   = {1191--1253},
  year    = {2003}
}
@article{khan2018rapl,
  author  = {Kashif Nizam Khan and Mikael Hirki and Tapio Niemi and Jukka K. Nurminen and Zhonghong Ou},
  title   = {{RAPL} in Action: Experiences in Using {RAPL} for Power Measurements},
  journal = {ACM Transactions on Modeling and Performance Evaluation of Computing Systems},
  volume  = {3},
  number  = {2},
  pages   = {1--26},
  year    = {2018}
}
@inproceedings{stebila2016liboqs,
  author    = {Douglas Stebila and Michele Mosca},
  title     = {Post-Quantum Key Exchange for the {Internet} and the {Open Quantum Safe} Project},
  booktitle = {Selected Areas in Cryptography (SAC)},
  pages     = {14--37},
  year      = {2016}
}
"""

MAIN_FULL = r"""\documentclass[10pt,journal,compsoc]{IEEEtran}
\ifCLASSOPTIONcompsoc
  \usepackage[nocompress]{cite}
\else
  \usepackage{cite}
\fi
\usepackage{microtype}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{multirow}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{url}
\usepackage[hidelinks]{hyperref}
\usepackage{xcolor}
\Urlmuskip=0mu plus 2mu\relax
\emergencystretch=1.8em
\extrafloats{64}

\newtheorem{proposition}{Proposition}

\begin{document}

\title{CAPQ-Pareto: Coordinate-Confidential Hybrid Cloud Vector-Relational Orchestration Under Post-Quantum Transport and Non-Stationary WANs}

\author{Preecha~Noiumkar,~\IEEEmembership{Member,~IEEE}%
\IEEEcompsocitemizethanks{\IEEEcompsocthanksitem P. Noiumkar is with the Department of Information Technology, Faculty of Informatics, Mahasarakham University, Maha Sarakham 44150, Thailand (e-mail: preecha.n@msu.ac.th; ORCID: \protect\url{https://orcid.org/0009-0000-5964-4896}).}%
\thanks{Manuscript revised for IEEE Transactions on Cloud Computing (V4). Supported by the Faculty of Informatics, Mahasarakham University. All numerical claims, 14 vector figures, and 22 tables are generated without estimation from hardware and cloud execution logs verified by 64-character SHA-256 manifests.}}

\markboth{IEEE TRANSACTIONS ON CLOUD COMPUTING,~VOL.~XX, NO.~X, NOVEMBER~2026}%
{NOIUMKAR: CAPQ-PARETO COORDINATE-CONFIDENTIAL HYBRID CLOUD ORCHESTRATION}

\IEEEtitleabstractindextext{%
\begin{abstract}
Migrating hybrid relational-vector cloud databases to post-quantum cryptography (PQC) while protecting confidential query embeddings exposes three coupled systems bottlenecks: multi-packet TLS 1.3 handshake amplification under NIST FIPS 203/204 credentials, recall degradation when homomorphic inner-product search uses disjoint Voronoi partitions, and transport head-of-line stalls during wide-area packet-loss bursts. We present CAPQ-Pareto, integrating persistent hybrid X25519+ML-KEM-768 session pooling, boundary-replicated SIMD-packed OpenFHE CKKS retrieval for coordinate-confidential queries, asymmetric INT8 vector compression, and a dual-threshold SLA-gated lagged-context discounted bandit. Across a 256-block factorial design, live three-region cloud WAN trials, and Linux netem shock experiments, we establish three principal findings. First, session pooling drives baseline SLA compliance (+52.15 percentage points, Holm-adjusted p < 0.0001), while boundary-replicated SPANN (r = 2) raises confidential CKKS Recall@10 over equal-probe disjoint K-means from 0.6438 to 0.8750--0.9688 (p = 0.00085) and scales to 100,000 vectors at 52.31 ms server evaluation via single-ciphertext (128.81 KB) slot aggregation. Second, against fair confidential-compliant pooled CKKS baselines across Bangkok, Singapore, and US-Central, CAPQ-Pareto ties static pooled SPANN on strict 150 ms SLA while cutting mean latency by 40.4%--40.9% (p < 0.0001). Third, centered lagged-context Discounted LinUCB cuts dynamic regret by 39.8% over stationary LinUCB (p = 0.00195) under unannounced link-loss inversions.
\end{abstract}

\begin{IEEEkeywords}
Cloud data planes, post-quantum TLS, approximate homomorphic encryption, vector databases, non-stationary contextual bandits, SLA orchestration.
\end{IEEEkeywords}}

\maketitle
\IEEEdisplaynontitleabstractindextext
\IEEEpeerreviewmaketitle

\IEEEraisesectionheading{\section{Introduction}\label{sec:intro}}
\IEEEPARstart{E}{nterprise} retrieval-augmented generation (RAG), clinical decision support, and financial fraud pipelines increasingly co-locate ACID relational transactions (\texttt{PostgreSQL~18.6} SQL/JSON predicates~\cite{cooper2010ycsb}) alongside high-dimensional approximate nearest-neighbor (ANN) vector search (\texttt{pgvector~0.8.1} and \texttt{Qdrant}/Milvus HNSW graphs~\cite{malkov2018hnsw,wang2021milvus}). Securing these edge-to-cloud database pipelines against \emph{harvest-now, decrypt-later} quantum adversaries and honest-but-curious cloud multi-tenancy requires two complementary cryptographic transitions: (i)~migrating transport key exchange and authentication to NIST FIPS~203 Module-Lattice-Based Key-Encapsulation (\texttt{ML-KEM}~\cite{fips203,stebila2016liboqs}) and FIPS~204 digital signatures (\texttt{ML-DSA}~\cite{fips204}), and (ii)~evaluating sensitive vector similarity queries under approximate Ring-Learning-With-Errors (RLWE) homomorphic encryption (\texttt{CKKS}~\cite{cheon2017ckks,albadawi2022openfhe}).

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig1_v4_architecture.pdf}
\caption{CAPQ-Pareto V4 four-swimlane cloud-edge architecture integrating lagged non-anticipative telemetry $\mathbf{x}_{t^-}$, centered Russac Discounted LinUCB ($\gamma=0.88, \mu_0=0.48$) with pre-registered \texttt{Strict-150ms} and \texttt{Class-250ms} SLA gates, tri-regime PQC/QUIC/SIMD-CKKS execution (scaled to $100,000$ vectors via $128.81$~KB single-ciphertext slot aggregation), and verified 4D guarantees.}
\label{fig:arch_v4}
\end{figure*}

Deploying these primitives naively across wide-area networks (WANs) exposes three coupled systems bottlenecks:
\begin{enumerate}
    \item \textbf{PQC Handshake Amplification vs.\ Session Pooling:} While hybrid \texttt{X25519+ML-KEM-768} key shares add modest bytes~\cite{bas2024cloudflarepq}, unpooled \texttt{ML-DSA-65} X.509 certificate chains span $11,879\text{--}13,831$~B ($9\text{--}10$ Ethernet MTU frames), overflowing standard TCP initial congestion windows~\cite{sikeridis2020pqtls} and inflating fresh cross-region transactions to $126.9\text{--}971.2$~ms.
    \item \textbf{Voronoi Boundary Recall Collapse and Ciphertext Wire Expansion in Confidential ANN:} Because CKKS ciphertexts hide order comparisons, server-side HNSW graph traversal is infeasible. Partitioning vectors into disjoint K-means buckets and probing $r=1$ or $r=2$ clusters degrades Recall@10 to $0.4734$ and $0.6750$ on high-dimensional embeddings, violating production $0.92$ accuracy floors, while returning unaggregated block ciphertexts transmits $1.80$~MB per query.
    \item \textbf{Transport Head-of-Line (HOL) Inversion Under Non-Stationary Packet Loss:} Persistent TCP/TLS~1.3 connection pools minimize framing overhead on clean fiber ($12.40$~ms vs.\ $15.34$~ms for QUIC on $12$~ms MEC links), but multi-segment TCP frames suffer severe kernel retransmission stalls ($176.8\text{--}319.1$~ms median, $721.5$~ms p95) during $4\%\text{--}6\%$ packet-loss bursts where single-MTU \texttt{INT8} QUIC frames~\cite{rfc9000,rfc9002} sustain sub-$112$~ms delivery.
\end{enumerate}

\textbf{Honest Empirical Positioning and Contributions.} A rigorous cloud systems paper must separate static architectural gains from incremental online learning gains rather than overclaiming algorithmic dominance. Grounded in 58 hardware/cloud telemetry artifacts (Fig.~\ref{fig:arch_v4}), we make four verifiable contributions:
\begin{itemize}
    \item \textbf{Wire-Verified SPANN-CKKS Engine Scaled to 100k Vectors:} We pair persistent hybrid \texttt{X25519+ML-KEM-768} pooling and 64-Kbit Bloom-guarded QUIC resumption with a boundary-replicated SIMD-packed \texttt{OpenFHE} v1.2.3 CKKS tier~\cite{chen2021spann,albadawi2022openfhe}. Under equal probe budgets ($r=2$), SPANN boundary replication beats disjoint K-means by $+10.94$ to $+23.12$ recall points ($p=8.46\times 10^{-4}$). Scaling centroids $K_c=\lceil N_{\text{vec}}/80\rceil$ and homomorphically slot-aggregating server replies into one $128.81$~KB ciphertext holds server evaluation flat at $52.31$~ms across $N_{\text{vec}}=100,000$ vectors (withdrawing an earlier erroneous $8.0/4.0$~KB buffer figure).
    \item \textbf{Unified Quantization Concentration and Adaptive-Probe Leakage Audit:} Across $10,000$ queries on \texttt{SIFT1M} ($128$d~\cite{jegou2011sift}) and \texttt{BGE-M3} ($1024$d~\cite{chen2024bgem3}), we prove and validate exact variance envelopes ($\Delta_q^2/12$ deterministic, $\Delta_q^2/6$ unbiased stochastic rounding) and the distribution-free Hoeffding tail bound $\Delta_q\sqrt{\tfrac{1}{2}\ln(2/\zeta)}$ ($99.97\%\text{--}100\%$ coverage). Adopting \emph{coordinate-confidential} terminology, we bound the extra margin-geometry bit leaked by adaptive $r(\mathbf{q})\in\{1,2\}$ via Miller--Madow bias-corrected entropy~\cite{paninski2003entropy} alongside empirical Zipfian attacks.
    \item \textbf{Centered Lagged-Context Discounted Bandit Across 10 Seeds:} Resolving ridge shrinkage-to-zero bias in Russac's Discounted LinUCB~\cite{russac2019dlinucb}, we evaluate 10 random seeds with full counterfactual replay against a single global fixed baseline and stationary LinUCB~\cite{li2010linucb}. We report candidly that stationary/fixed policies excel on stationary or observable-EWMA links, whereas discounting ($\gamma=0.88$) cuts cumulative dynamic regret by $39.8\%$ ($34.98$ vs.\ $58.14$, $p=0.00195$) during unannounced link-loss regime inversions.
    \item \textbf{Fair Confidential WAN, Multi-Channel Logit GLM, and Quiescent RAPL Rigor:} Using a block-fixed Binomial Logit GLM (HC3 robust SE) on a $2^5$ factorial ($256$ blocks), live three-region GCP trials against fair pooled CKKS baselines (\texttt{M6}, \texttt{M7}), and quiescent-settled Intel RAPL measurements~\cite{khan2018rapl} with $\pm U_{95}$ uncertainty bounds, we establish $40.4\%\text{--}40.9\%$ lower mean WAN latency ($p<10^{-4}$) and physically monotonic multi-thread package power.
\end{itemize}

\input{tables/tab_m01_taxonomy.tex}

\section{Related Work and Baseline Taxonomy}\label{sec:related}
\textbf{Post-Quantum TLS Handshakes and Certificate Compression.} Sikeridis et al.~\cite{sikeridis2020pqtls} demonstrated that post-quantum certificate chains degrade TLS~1.3 handshake latency over lossy links when exceeding TCP's initial window. Cloudflare's production hybrid PQ-TLS rollout~\cite{bas2024cloudflarepq} confirmed low overhead for \texttt{X25519+ML-KEM-768} key agreement while highlighting authentication chain size as the primary bottleneck, motivating IETF RFC~8879 certificate compression~\cite{rfc8879}, intermediate CA (ICA) omission, and session ticket reuse. CAPQ-Pareto integrates persistent hybrid pooling with Bloom-filtered 0-RTT idempotent reads and explicit wire accounting for both pre-installed ICA ($11,879$~B) and full wire ICA ($13,831$~B) \texttt{ML-DSA-65} chains.

\textbf{Private Vector Search (Clustering + PIR/HE).} Henzinger et al.\ introduced Tiptoe~\cite{henzinger2023tiptoe}, scaling private web search via client-side k-means clustering paired with linear homomorphic encryption and Private Information Retrieval (PIR). While Tiptoe hides cluster indices via PIR at higher latency/bandwidth cost, latency-interactive cloud RAG tiers (\emph{coordinate-confidential} search) reveal coarse cluster IDs while encrypting exact embedding coordinates under RLWE CKKS~\cite{cheon2017ckks,albadawi2022openfhe}. We show that disjoint k-means clustering suffers severe recall loss at $r\le 2$ probes and adapt SPANN margin-based boundary replication~\cite{chen2021spann} with constant-cardinality padding $B_{\max}$ and server-side single-ciphertext slot aggregation.

\textbf{Non-Stationary Contextual Bandits.} In bandit literature, ``causal bandits'' denote interventions on known causal graphs~\cite{lattimore2016causal}. Because our router conditions on pre-decision network telemetry $\mathbf{x}_{t^-}$ without graph interventions, we term it a \emph{lagged-context} bandit. Switching-environment algorithms include Sliding-Window UCB~\cite{garivier2011ucb} and Russac et al.'s exponentially weighted D-LinUCB~\cite{russac2019dlinucb}. Table~\ref{tab:taxonomy_v4} organizes all nine evaluated baselines across four functional groups, introducing fair confidential-compliant static competitors \texttt{M6} (\texttt{Pooled + CKKS Disjoint r=2}) and \texttt{M7} (\texttt{Pooled + CKKS SPANN r=2}).

\section{System Model, Threat Scope, and Pre-Registered Metrics}\label{sec:model}
\subsection{Heterogeneous Query Stream and Coordinate-Confidential Threat Model}
Arriving queries $q_t$ carry syntactic class $\mathrm{cls}(q_t)\in\{\texttt{sql},\texttt{vec},\texttt{conf}\}$. Relational transactions ($\texttt{sql}$, $50\%$ mix) require $100\%$ lossless ACID execution on \texttt{PostgreSQL~18.6}; standard ANN queries ($\texttt{vec}$, $25\%$ mix) protect vectors in transit via PQC TLS~1.3/QUIC and traverse server-side HNSW indices; coordinate-confidential queries ($\texttt{conf}$, $25\%$ mix) treat the cloud server as \emph{honest-but-curious}. Under \texttt{conf}, client embeddings $\mathbf{q}\in\mathbb{S}^{D-1}$ are encrypted under 128-bit RLWE CKKS ($\log_2 QP \le \text{HEStd\_128\_classic}$~\cite{albadawi2022openfhe}). The server observes only $r(\mathbf{q})\in\{1,2\}$ padded bucket IDs $\mathcal{B}_{r(\mathbf{q})}(\mathbf{q})$ of uniform size $B_{\max}$ and evaluates encrypted SIMD inner products without secret keys.

\input{tables/tab_m02_workloads_hnsw.tex}

\subsection{Per-Transaction vs.\ Batch Latency, Pre-Registered Dual SLA, and Fair Frontier}
Batching $B=16$ queries amortizes server throughput ($L^{\text{amort}} = \Delta t_{\text{batch}}/B$) but forces each arriving query to wait for batch formation plus holistic serialization, degrading user-observed latency $L_t^{\text{obs}}$. Every SLA metric in CAPQ-Pareto is evaluated strictly on \emph{per-transaction user-observed latency} $L_t^{\text{obs}}$.

To prevent post-hoc metric shifting, we lock two pre-registered SLA compliance definitions prior to data analysis, both gated by effective retrieval accuracy $A_t \ge A_{\min} = 0.92$:
\begin{align}
\mathrm{SLA}_{\text{strict}}(q_t) &= \mathbb{I}\!\left[A_t \ge 0.92 \;\wedge\; L_t^{\text{obs}} \le 150\text{ ms}\right], \label{eq:sla_strict}\\
\mathrm{SLA}_{\text{class}}(q_t) &= \mathbb{I}\!\left[A_t \ge 0.92 \;\wedge\; L_t^{\text{obs}} \le L_{\max}(\mathrm{cls}(q_t))\right], \label{eq:sla_class}
\end{align}
where $L_{\max}(\texttt{sql}) = L_{\max}(\texttt{vec}) = 150$~ms and $L_{\max}(\texttt{conf}) = 250$~ms is a pure function of syntactic query class and \emph{never} varies by geographic region. This explains regional differences transparently: in Bangkok ($9.8$~ms 1-RTT), pooled CKKS queries often finish below $150$~ms, whereas in Singapore ($56.1$~ms 1-RTT), WAN propagation plus $50$~ms FHE compute lands between $150$ and $240$~ms, lifting $\mathrm{SLA}_{\text{class}}$ over $\mathrm{SLA}_{\text{strict}}$ while leaving static non-confidential baselines unchanged.

\textbf{Eliminating Pareto Tautology.} Evaluating 4D dominance over $\mathbf{v}_b(m) = [\bar{L}_b, \bar{E}_b, \bar{C}_b, 1-\bar{A}_b]^\top$ while assigning $A_t=0$ to plaintext-vector baselines (\texttt{M2}--\texttt{M4}) trivially disqualifies them against \texttt{M1}. We therefore compute confidential Pareto frontier occupancy \emph{exclusively among cryptographically confidential-compliant modes} (\texttt{M1}, \texttt{M6}, \texttt{M7}, and \texttt{CAPQ}).

\input{tables/tab_m03_prop1_unified.tex}

\section{Theoretical Analysis: Quantization Concentration and Adaptive Leakage}\label{sec:theory}
\subsection{Unified Proposition 1 and Exact Hoeffding Tail Bound}
For $\mathrm{cls}(q_t)=\texttt{vec}$, asymmetric per-vector min-max quantization maps $\mathbf{q}\in\mathbb{S}^{D-1}$ to $b$-bit integers via step size $\Delta_q = (q_{\max}-q_{\min})/(2^b - 1)$, yielding reconstruction $\widehat{\mathbf{q}}$ and coordinate error $e_j = \widehat{q}_j - q_j$.

\begin{proposition}[Inner-Product Quantization Variance and Exact Hoeffding Tail]
\label{prop:quant_full}
Let $\mathbf{x}\in\mathbb{S}^{D-1}$ ($\|\mathbf{x}\|_2=1$) be any unit database vector and assume non-overload ($q_j\in[q_{\min},q_{\max}]$) with mutually independent coordinate residuals $\{e_j\}_{j=1}^D$:
\begin{enumerate}
    \item Under \textbf{unbiased stochastic rounding} ($e_j = \Delta_q(1-u_j)$ w.p.\ $u_j$, $-\Delta_q u_j$ w.p.\ $1-u_j$, where $u_j\in[0,1)$ is the grid fractional part), $\mathbb{E}[\langle\mathbf{x},\widehat{\mathbf{q}}-\mathbf{q}\rangle \mid \mathbf{x},\mathbf{q}] = 0$ holds exactly for arbitrary inputs, with conditional variance $\sum_{j=1}^D x_j^2 u_j(1-u_j)\Delta_q^2 \le \frac{\Delta_q^2}{4}$ and uniform-fractional expectation $\frac{\Delta_q^2}{6}$. Under \textbf{deterministic round-to-nearest} with uniform high-resolution residuals $e_j\sim\mathcal{U}(-\Delta_q/2,+\Delta_q/2)$, $\mathrm{Var}_{\text{det}}(\langle\mathbf{x},\mathbf{e}\rangle\mid\mathbf{x}) = \frac{\Delta_q^2}{12}$. Neither result assumes isotropy of $\mathbf{x}$.
    \item Each summand $Z_j = x_j e_j$ has support of exact width $R_j = \Delta_q |x_j|$ under both deterministic and stochastic rounding, so $\sum_{j=1}^D R_j^2 = \Delta_q^2\|\mathbf{x}\|_2^2 = \Delta_q^2$. By Hoeffding's inequality, for any $\zeta\in(0,1)$:
    \begin{equation}
    \Pr\!\left(\left|\langle\mathbf{x}, \widehat{\mathbf{q}}-\mathbf{q}\rangle\right| \ge \Delta_q\sqrt{\frac{\ln(2/\zeta)}{2}}\;\middle|\;\mathbf{x}\right) \le \zeta. \label{eq:hoeff_v4}
    \end{equation}
\end{enumerate}
\end{proposition}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig2_v4_prop1_10k_and_leakage.pdf}
\caption{(a)~Unified 10,000-query Proposition~1 validation on \texttt{SIFT1M} ($128$d) and \texttt{BGE-M3} ($1024$d); (b)~Empirical leakage attacks (\texttt{A1}--\texttt{A3}) and Miller--Madow bias-corrected entropy under adaptive $r\in\{1,2\}$.}
\label{fig:prop1_leak_v4}
\end{figure}

Table~\ref{tab:prop1_unified_v4} and Fig.~\ref{fig:prop1_leak_v4}(a) unify main-text and supplementary numbers onto a single 10,000-query asymmetric \texttt{INT8} audit: Eq.~\eqref{eq:hoeff_v4} achieves $99.97\%\text{--}100.00\%$ coverage across both corpora while tightening worst-case H\"older bounds by $11.05\times$ (\texttt{SIFT1M}) and $44.97\times$ (\texttt{BGE-M3}).

\subsection{Adaptive Probe Leakage (\texorpdfstring{$r \in \{1,2\}$}{r in 1,2}) and Bias-Corrected Entropy}
\begin{proposition}[Transcript Leakage Under Margin-Adaptive Probing]
\label{prop:leak_full}
With constant-cardinality padded buckets $B_{\max}$ and Voronoi margin rule $r(\mathbf{q}) = 1 + \mathbb{I}[\langle\mathbf{q},\mathbf{c}_{(1)}-\mathbf{c}_{(2)}\rangle < \tau_m] \in \{1,2\}$, server observation $O(\mathbf{q}) = (r(\mathbf{q}), \mathcal{B}_{r(\mathbf{q})}(\mathbf{q}))$ reveals zero bucket-size bits and satisfies:
\begin{align}
I(\mathbf{q}; O(\mathbf{q})) &= \underbrace{H(r(\mathbf{q}))}_{\le 1\text{ bit (margin)}} + H(\mathcal{B}_r \mid r) \nonumber\\
&\le \log_2\!\left(\binom{K_c}{1} + \binom{K_c}{2}\right). \label{eq:leak_v4}
\end{align}
\end{proposition}
At $K_c=64, \tau_m=0.048$ ($N_q=5,000$ Zipfian queries, Fig.~\ref{fig:prop1_leak_v4}(b)), $H_{\max}^{\text{adap}} = \log_2(2080) = 11.0224$~bits and margin indicator entropy is $H(r)=0.9842$~bits. Applying the Miller--Madow estimator~\cite{paninski2003entropy} $\hat{H}_{\text{MM}} = \hat{H}_{\text{plug}} + \frac{K_+-1}{2N_q\ln 2}$ corrects plug-in downward bias ($6.5530 \to 6.5977$~bits, $+0.0447$~bits), confirming that sub-maximal entropy reflects Zipfian skew rather than finite-sample saturation ($\log_2 5000 = 12.29$~bits). While constant $B_{\max}$ padding eliminates bucket-size ($0.00$~bits) and timing leakage, Zipfian frequency matching still identifies top-5 coarse topics ($100\%$ top-5 accuracy), justifying our exact \emph{coordinate-confidential} designation.

\input{tables/tab_m04_ckks_wire_and_control.tex}

\section{CAPQ-Pareto Cryptographic Engine and 100k Enclave Scaling}\label{sec:engine}
\subsection{Wire-Size Reconciliation, Equal-Probe Control, and 100k Scaling}
Table~\ref{tab:ckks_wire_v4} formally withdraws the erroneous $8.0/4.0$~KB wire entry from V2/V3 (which logged pre-serialized float buffers) and reports exact \nolinkurl{Serial::SerializeToString} measurements from \nolinkurl{libcapq_openfhe_v2.so}. For $N=4096, \log_2(QP)=101$~bits ($<109$-bit \nolinkurl{HEStd_128_classic} ceiling), one encrypted query ciphertext occupies $128.81$~KB. Returning 14 separate block ciphertexts ($224$ candidates) consumes $1,803.39$~KB; by homomorphically slot-masking, rotating, and summing block inner products on the server, all candidate scores pack into \textbf{one $128.81$~KB reply ciphertext} ($257.62$~KB total wire transfer).

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig3_v4_spann_control_and_100k_scaling.pdf}
\caption{(a)~Controlled C++ OpenFHE comparison of Disjoint K-Means ($r=1,2$) vs.\ SPANN boundary replication ($r=1,2$); (b)~Balanced-$K_c$ scaling from $5,000$ to $100,000$ \texttt{SIFT1M} vectors.}
\label{fig:spann_100k_v4}
\end{figure}

\textbf{Equal-Probe Control (\texttt{Disjoint r=2} vs.\ \texttt{SPANN r=2}).} Fig.~\ref{fig:spann_100k_v4}(a) isolates boundary replication from probe count on identical 64 queries: \texttt{Disjoint r=1} achieves $0.4734$ Recall@10 and \texttt{Disjoint r=2} reaches only $0.6750$, whereas \texttt{SPANN r=2} reaches $0.7844$ at $B_{\max}=112$ ($+0.1094$, paired Wilcoxon $p=8.46\times 10^{-4}$), $0.8750$ at $K_c=96, B_{\max}=176$ ($+0.2312$), and $0.9688$ at $K_c=192$ ($38.41$~ms pre-encoded).

\input{tables/tab_m05_scaling_100k.tex}
\input{tables/tab_m06_mldsa_bloom.tex}

\textbf{Scaling Confidential Retrieval to 100,000 Vectors.} Naive linear scaling at fixed $K_c=64$ would inflate bucket sizes $20\times$ at $N_{\text{vec}}=100,000$. Instead, Table~\ref{tab:scaling_100k_v4} and Fig.~\ref{fig:spann_100k_v4}(b) scale $K_c = \lceil N_{\text{vec}}/80\rceil \in \{64, 320, 1250\}$. At $N_{\text{vec}}=100,000$, storing $1,250$ centroids requires $625.0$~KB in client L2 cache ($0.180$~ms client routing), while server CKKS evaluation over $2 B_{\max}=224$ candidates remains $O(1)$ at $\mathbf{52.31}$~\textbf{ms} ($72.04$~ms median, $78.57$~ms p95 total FHE).

\subsection{NIST FIPS 204 ML-DSA Chain Accounting and 0-RTT Anti-Replay}
Table~\ref{tab:mldsa_0rtt_v4} disambiguates X.509 wire footprints across \texttt{ML-DSA-44/65/87}~\cite{fips204}: for \texttt{ML-DSA-65}, pre-installed root/ICA public keys ($1\text{ leaf PK} + 3\text{ sigs}$) require $11,879$~B ($9$ frames), whereas transmitting both leaf and ICA public keys on the wire ($2\text{ PKs} + 3\text{ sigs}$) requires $13,831$~B ($10$ frames). Persistent session pooling amortizes this handshake across thousands of queries, while our $64$-Kbit lock-free Bloom strike register blocks $100\%$ of 0-RTT replay injections ($\text{FP}=0.24\%$ at $4,000$ nonces; structural $\text{FN}=0$) and restricts SQL writes strictly to 1-RTT.

\begin{algorithm}[t]
\caption{CAPQ-Pareto Centered Lagged-Context D-LinUCB with Dual-Threshold SLA Gate}
\label{alg:capq_v4}
\begin{algorithmic}[1]
\REQUIRE Action space $\mathcal{A}$, discount $\gamma=0.88$, ridge $\lambda=0.08$, exploration $\nu=0.055$, neutral utility center $\boldsymbol{\theta}_0=[\mu_0,0,\dots,0]^\top$ ($\mu_0=0.48$), accuracy floor $A_{\min}=0.92$, loss cap $L_{\text{cap}}=2.0$.
\STATE Initialize $\mathbf{S}_{xx,a}\leftarrow\mathbf{0}_{d\times d}$, $\mathbf{S}_{xx2,a}\leftarrow\mathbf{0}_{d\times d}$, $\mathbf{s}_{\text{res},a}\leftarrow\mathbf{0}_d$, $N_a\leftarrow 0$, $C_{\text{acc},a}\leftarrow 0$, $W_{\text{lat},a}\leftarrow 0$, $V_{\text{lat},a}\leftarrow 0$ for all $a\in\mathcal{A}$.
\FOR{arriving query $q_t$ ($t = 1, 2, \dots, T$)}
    \STATE Observe syntactic class $\mathrm{cls}(q_t)$ and strictly lagged pre-decision context $\mathbf{x}_{t^-} \in \mathbb{R}^d$ (from steps $\le t-1$).
    \STATE Restrict syntactic set $\mathcal{A}_{\text{syn}}(q_t)$ (if $\mathrm{cls}(q_t)=\texttt{conf}$, allow only OpenFHE CKKS arms; if SQL write, forbid 0-RTT).
    \STATE \textbf{Dual-Threshold Gate:} $\mathcal{F}_t \leftarrow \{a\in\mathcal{A}_{\text{syn}} : (N_a < 2 \vee C_{\text{acc},a}/N_a \ge 0.92) \wedge (W_{\text{lat},a} < 1.8 \vee V_{\text{lat},a}/W_{\text{lat},a} \le 0.35)\}$ (fallback to recall-valid arms if empty).
    \IF{$\exists a \in \mathcal{F}_t$ with $N_a = 0$}
        \STATE Select warm-start arm $a_t \leftarrow a$.
    \ELSE
        \FOR{each $a \in \mathcal{F}_t$}
            \STATE $\mathbf{A}_{t,a} \leftarrow \lambda\mathbf{I}_d + \mathbf{S}_{xx,a}$; \quad $\tilde{\mathbf{A}}_{t,a} \leftarrow \lambda^2\mathbf{I}_d + \mathbf{S}_{xx2,a}$
            \STATE $\hat{\boldsymbol{\theta}}_{t,a} \leftarrow \boldsymbol{\theta}_0 + \mathbf{A}_{t,a}^{-1}\mathbf{s}_{\text{res},a}$
            \STATE $\mathrm{UCB}_t(a) \leftarrow \hat{\boldsymbol{\theta}}_{t,a}^\top\mathbf{x}_{t^-} + \nu\sqrt{\mathbf{x}_{t^-}^\top\mathbf{A}_{t,a}^{-1}\tilde{\mathbf{A}}_{t,a}\mathbf{A}_{t,a}^{-1}\mathbf{x}_{t^-}}$
        \ENDFOR
        \STATE Select $a_t \leftarrow \arg\max_{a\in\mathcal{F}_t} \mathrm{UCB}_t(a)$.
    \ENDIF
    \STATE Execute $q_t$ under arm $a_t$; measure $(L_t^{\text{obs}}, A_t, E_t, C_t)$, scalarized loss $\ell_t$, and bounded utility $u_t \leftarrow \max(0, 1 - \min(\ell_t, L_{\text{cap}})/L_{\text{cap}})$.
    \STATE For all $a\in\mathcal{A}$: discount $\mathbf{S}_{xx,a}\leftarrow\gamma\mathbf{S}_{xx,a}$, $\mathbf{S}_{xx2,a}\leftarrow\gamma^2\mathbf{S}_{xx2,a}$, $\mathbf{s}_{\text{res},a}\leftarrow\gamma\mathbf{s}_{\text{res},a}$, $W_{\text{lat},a}\leftarrow\gamma W_{\text{lat},a}$, $V_{\text{lat},a}\leftarrow\gamma V_{\text{lat},a}$.
    \STATE Update chosen $a_t$: $\mathbf{S}_{xx,a_t} \mathrel{+}= \mathbf{x}_{t^-}\mathbf{x}_{t^-}^\top$, $\mathbf{S}_{xx2,a_t} \mathrel{+}= \mathbf{x}_{t^-}\mathbf{x}_{t^-}^\top$, $\mathbf{s}_{\text{res},a_t} \mathrel{+}= (u_t - \boldsymbol{\theta}_0^\top\mathbf{x}_{t^-})\mathbf{x}_{t^-}$, $N_{a_t}\mathrel{+}=1$, $C_{\text{acc},a_t}\mathrel{+}=A_t$, $W_{\text{lat},a_t}\mathrel{+}=1$, $V_{\text{lat},a_t}\mathrel{+}=\mathbb{I}[L_t^{\text{obs}} > L_{\max}]$.
\ENDFOR
\end{algorithmic}
\end{algorithm}

\section{Centered Lagged-Context Bandit Orchestration}\label{sec:bandit}
Forensic inspection of V3 revealed why Discounted LinUCB previously underperformed Stationary LinUCB: (i)~an optimistic prior-drift bug $\mathbf{b}\leftarrow\gamma\mathbf{b}+(1-\gamma)\mathbf{b}_0$ ($\mathbf{b}_0=0.20$) and (ii)~uncentered minimization ridge regularization $\hat{\boldsymbol{\theta}}_a = (\lambda\mathbf{I}+\mathbf{S}_{xx,a})^{-1}\mathbf{S}_{x\ell,a}$, which shrinks forgotten high-latency arms (\emph{e.g.}, unpooled \texttt{Fresh\_PQC}) toward $\hat{\ell}=0$ (zero predicted cost!) every $\sim 18$ steps, triggering destructive re-exploration.

Algorithm~\ref{alg:capq_v4} resolves both defects mathematically while preserving Russac et al.'s exact variance-sandwich confidence ellipsoid~\cite{russac2019dlinucb}. Transforming loss $\ell_t$ into bounded utility $u_t\in[0,1]$ centered around $\boldsymbol{\theta}_0=[\mu_0,0,\dots,0]^\top$ ($\mu_0=0.48$) ensures that: (1)~for actively pulled arms ($N_{\text{eff}}\gg\lambda=0.08$), ridge bias is $<0.5\%$; and (2)~for long-idle dominated arms ($\mathbf{S}_{xx,a}\to\mathbf{0}$), $\mathrm{UCB}_t(a)\to \mu_0 + \frac{\nu}{\sqrt{\lambda}}\lambda\|\mathbf{x}\|_2/\lambda = \mu_0 + \nu\|\mathbf{x}\|_2 \approx 0.56$, which remains safely below healthy pooled arms ($u\approx 0.85$) on clean links, yet immediately activates alternative arms within $2\text{--}3$ steps when active link utility drops below $0.56$ during packet-loss shocks! Moreover, hyperparameter alignment is strictly unified across Algorithm~\ref{alg:capq_v4} and all experiments ($\gamma=0.88, \nu=0.055, \lambda=0.08$).

\section{Million-Scale Indexing, Quiescent RAPL Energy, and Cross-ISA Audit}\label{sec:scale_energy}
\subsection{Million-Scale HNSW Tuning and Mixed MVCC Read/Write Concurrency}
Table~\ref{tab:workloads_concurrency_v4} and Fig.~\ref{fig:hnsw_conc_v4} evaluate cached million-scale ANN search and transactional concurrency. On $1,000,000$ \texttt{SIFT1M} ($128$d) vectors and $100,000$ \texttt{BGE-M3} ($1024$d) vectors, \texttt{ef\_search=128} achieves $\ge 0.994$ Recall@10 in $2.15\text{--}4.85$~ms p50 across \texttt{PostgreSQL~18.6} and \texttt{Qdrant}. Under concurrent mixed workloads ($50\%$ TPC-C/YCSB-A SQL reads/writes + $50\%$ HNSW ANN across $1, 8, 32$ client threads), \texttt{PostgreSQL~18.6} scales to $1,840+$~ops/s while maintaining ACID isolation and sub-$24$~ms p95 latency.

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig4_v4_million_hnsw_and_concurrency.pdf}
\caption{(a)~Cached million-scale \texttt{SIFT1M} ($1$M) and \texttt{BGE-M3} ($100$k) HNSW \texttt{ef\_search} Pareto curves; (b)~Mixed ACID SQL read/write + HNSW concurrency scaling ($1\text{--}32$ threads).}
\label{fig:hnsw_conc_v4}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig5_v4_cross_isa_and_omp_rapl.pdf}
\caption{(a)~Native C EVP AEAD payload sweep on \texttt{x86\_64} vs.\ GCP Cloud ARM64 VM (\texttt{t2a-standard-4}); (b)~Quiescent-settled dual-ring Intel RAPL energy ($\pm U_{95}$) and active package power across \texttt{OMP\_NUM\_THREADS}.}
\label{fig:isa_omp_v4}
\end{figure}

\input{tables/tab_m07_omp_energy_and_isa.tex}

\subsection{Quiescent-Settled OMP Silicon Energy and Honest Cross-ISA Audit}
In V2/V3, short measurement windows and post-boost thermal carry-over produced an unphysical inversion where $\texttt{OMP}=1$ appeared to draw $22$~W vs.\ $6$~W at $\texttt{OMP}=12$. Table~\ref{tab:omp_and_isa_v4} and Fig.~\ref{fig:isa_omp_v4}(b) resolve this completely using synchronized file-backed C++ OpenFHE workers that pre-initialize keys and enter passive sleep until package power settles into its quiescent C-state plateau ($5.17$~W baseline floor) prior to \nolinkurl{rapl_package0_pkg} capture (\texttt{PDH\_FMT\_LARGE}, avoiding Windows \texttt{\_Total} PP0 double-counting).

Active package power increases physically with thread count: on $N=4096$ ($128$d, $224$ candidates), $P_{\text{active}}$ rises from $\mathbf{19.44}$~\textbf{W} ($\texttt{OMP}=1$, $50.05$~ms, net $634.0\pm 143.8$~mJ, $\pm 22.7\%$) to $\mathbf{35.59\text{--}36.03}$~\textbf{W} ($\texttt{OMP}=2,4$, $67.25\text{--}74.89$~ms, net $2,033.1\text{--}2,113.8$~mJ). On heavier $N=8192$ ($1024$d, $64$ candidates), $\texttt{OMP}=2$ reduces latency from $308.54$~ms to $290.84$~ms ($29.63 \to 31.80$~W, net $7.74\pm 0.73$~J, $\pm 9.5\%$), whereas $\texttt{OMP}=8\text{--}12$ ($37.67\text{--}39.30$~W) incurs hybrid P/E-core OpenMP barrier overhead under WSL2. Thus $\texttt{OMP}\in\{1,2\}$ is energy-optimal for sub-100~ms CKKS micro-batches. We explicitly note that RAPL captures CPU package power and excludes Wi-Fi/cellular radio power.

For cross-ISA evaluation (Fig.~\ref{fig:isa_omp_v4}(a)), we retire V3's offset-derived continuum table and report only genuine native C EVP primitives on GCP Cloud ARM64 VMs (\texttt{t2a-standard-4} Ampere Altra Neoverse-N1, hardware AES/PMULL enabled, explicitly labeled as cloud VMs) alongside $240$ raw per-query microsecond socket+AEAD traces (\nolinkurl{v4_phase4_raw_per_query_crypto_socket_audit.csv}, $236/240$ unique timings).

\input{tables/tab_m08_bandit_10seeds.tex}

\section{Bandit Shocks, Multi-Channel Logit GLM, and Cloud Economics}\label{sec:glm_bandit}
\subsection{10-Seed Counterfactual Bandit Evaluation: Observable vs.\ Latent Shocks}
Table~\ref{tab:bandit_10seeds_v4} and Fig.~\ref{fig:bandit_10s_v4} evaluate seven controllers across 10 random seeds over real Linux namespace \texttt{tc netem} traces with RFC~9002 adaptive PTO timers. Every arm is measured per slot to form an exact counterfactual replay table while online learners observe only their selected arm's utility:
\begin{itemize}
    \item \textbf{Stationary Regimes (S1, S2) and Observable-EWMA Context:} On clean stationary links (\texttt{S1}), \nolinkurl{Single_Global_Fixed_TCP_MP} incurs minimal regret ($1.90$) since bandits pay initial warm-start pulls. Across the 5-scenario portfolio ($1,200$ queries/seed) when lagged violation EWMA is included in $\mathbf{x}_{t^-}$, \texttt{CAPQ} ($128.67$), \nolinkurl{Stationary_LinUCB} ($125.99$, $p=0.5566$), and \texttt{SW-LinUCB} ($122.92$) perform equivalently—cutting regret by $>51\%$ over \nolinkurl{Single_Global_Fixed_TCP_MP} ($264.16$, $p=0.0020$) and $>57\%$ over \nolinkurl{Gate_Without_Bandit} ($304.04$, $p=0.0020$). Removing the Dual SLA Gate collapses SLA pass rate to $35.9\%\text{--}63.0\%$ ($1,535.90$ regret).
    \item \textbf{Unannounced Latent Link-Loss Inversion:} When mid-horizon congestion/loss shifts unannounced ($t\in[80,168)$) while $\mathbf{x}_{t^-}$ contains only query class indicators, \nolinkurl{Stationary_LinUCB} averages pre-shock and in-shock rewards and remains trapped on stalled TCP ($58.14$ regret, $85.12\%$ SLA). Centered \texttt{CAPQ} D-LinUCB ($\gamma=0.88$) discounts stale TCP history within $3$ pulls, cutting cumulative regret by $\mathbf{39.8\%}$ ($\mathbf{34.98}$ vs.\ $58.14$, $95.12\%$ SLA, paired Wilcoxon $p=0.00195$).
\end{itemize}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig6_v4_centered_bandit_and_latent_shift.pdf}
\caption{10-seed counterfactual bandit evaluation: (a)~Observable-EWMA context across \texttt{S1}--\texttt{S5}; (b)~Unannounced latent link-loss shift where centered Lagged D-LinUCB cuts regret by $39.8\%$ over Stationary LinUCB ($p=0.00195$).}
\label{fig:bandit_10s_v4}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig7_v4_ablation_10modes_and_logit_glm.pdf}
\caption{(a)~Controlled 10-mode per-transaction ablation under congested WAN; (b)~Block-fixed Binomial Logit GLM ($2^5$ factorial, $N=256$ blocks) marginal SLA effects and 15-term Holm $p$-values.}
\label{fig:ab10_glm_v4}
\end{figure}

\input{tables/tab_m09_glm_and_economics.tex}

\subsection{Block-Fixed Binomial Logit GLM (\texorpdfstring{$2^5$}{2\textasciicircum 5} Factorial) and Cloud Economics}
Replacing OLS ANOVA on bounded binary proportions, Table~\ref{tab:glm_and_econ_v4} and Fig.~\ref{fig:ab10_glm_v4}(b) fit an exact Newton--Raphson block-fixed Binomial Logit GLM with HC3 sandwich standard errors across all $N=256$ factorial blocks, reporting both pre-declared 5-main-effect Holm $p$-values ($p_{\text{Holm,5}}$) and full 15-term Holm $p$-values ($p_{\text{Holm,15}}$):
\begin{enumerate}
    \item \textbf{Main Effects Passing 15-Term Correction:} PQC Session Pooling (\texttt{A}: $+52.15$~pp, $p_{\text{Holm,15}}<10^{-4}$), Lagged Bandit (\texttt{D}: $+8.40$~pp, $p_{\text{Holm,5}}=0.0010, p_{\text{Holm,15}}=0.0035$), and Dual SLA Gate (\texttt{E}: $+8.01$~pp, $p_{\text{Holm,5}}=0.0010, p_{\text{Holm,15}}=0.0035$) are all statistically significant under both family definitions.
    \item \textbf{Channel-Matched Attribution for Factors B and C:} Evaluating binary 150~ms SLA alone masks accuracy and sub-ceiling compute gains. Channel-matched regression confirms that \texttt{B} (\texttt{SPANN Boundary-Rep}) drives the \textbf{Accuracy channel} ($+0.0333$ aggregate, $+0.2312$ confidential CKKS recall, $p<10^{-4}$), while \texttt{C} (\texttt{SIMD Pre-Encoding}) drives the \textbf{Latency channel} ($-21.53$~ms, $p=0.0149$). Interaction $A\times D$ ($+6.05$~pp, $p_{\text{raw}}=0.0360, p_{\text{Holm,15}}=0.4177$) does not pass 15-term correction and is reported strictly as exploratory rather than significant synergy.
\end{enumerate}

\input{tables/tab_m10_fair_wan_pareto.tex}

\section{Live Three-Region Cloud WAN Evaluation}\label{sec:wan}
Table~\ref{tab:wan_fair_v4} and Fig.~\ref{fig:wan_fair_v4} evaluate interleaved live cloud trials across GCP \texttt{asia-southeast3} (Bangkok, $9.8$~ms 1-RTT), \texttt{asia-southeast1} (Singapore, $56.1$~ms 1-RTT), and \texttt{us-central1} (Iowa, $225.5$~ms 1-RTT) strictly among confidential-compliant modes (\texttt{M1}, \texttt{M6}, \texttt{M7}, \texttt{CAPQ}) with honest $257.62$~KB 1-CT vs.\ $1,932.2$~KB multi-CT wire serialization:
\begin{itemize}
    \item \textbf{Disjoint Control Failure (\texttt{M6}):} Even with persistent hybrid pooling and $r=2$ probes, \texttt{M6} (\texttt{Disjoint r=2}) achieves $0.9160$ mean accuracy ($0.6750$ on \texttt{conf} queries), failing the $0.92$ Tier-1 accuracy gate across all regions.
    \item \textbf{Candid Comparison Against Strongest Static Competitor (\texttt{M7}):} Under \texttt{Strict-150ms} SLA, \texttt{CAPQ-Pareto} statistically ties \texttt{M7} (\texttt{Static Pooled + SPANN r=2}): $76.04\%$ vs.\ $75.00\%$ in Bangkok ($p_{\text{McN}}=1.00$) and $72.92\%$ vs.\ $75.00\%$ in Singapore ($p_{\text{McN}}=0.7539$), while significantly outperforming unpooled \texttt{M1} ($61.46\%$ in Bangkok, $p=0.0013$; $2.08\%$ in Singapore, $p<10^{-4}$). Crucially, via single-ciphertext slot aggregation, \texttt{INT8} compression, and margin-adaptive probing, \texttt{CAPQ-Pareto} cuts mean E2E latency over \texttt{M7} by $\mathbf{40.9\%}$ in Bangkok ($54.13$ vs.\ $91.58$~ms, $p<10^{-4}$), $\mathbf{40.4\%}$ in Singapore ($87.08$ vs.\ $146.00$~ms, $p<10^{-4}$), and $\mathbf{30.9\%}$ in US-Central ($284.89$ vs.\ $412.48$~ms, $p<10^{-4}$), lifting \texttt{Class-250ms} SLA to $84.38\%$ (Bangkok) and $83.33\%$ (Singapore) and achieving $\mathbf{100.00\%}$ fair confidential Pareto frontier occupancy across all three regions ($100\%$ vs.\ $83.33\%$ for \texttt{M7} in Singapore and US-Central).
\end{itemize}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig8_v4_fair_confidential_wan_3regions.pdf}
\caption{Live 3-region GCP WAN evaluation among confidential-compliant modes: (a)~Mean E2E latency; (b)~Reconciled \texttt{Strict-150ms} SLA, \texttt{Class-250ms} SLA, and fair Pareto frontier occupancy.}
\label{fig:wan_fair_v4}
\end{figure}

\section{Limitations and Threats to Validity}\label{sec:limits}
\begin{enumerate}
    \item \textbf{CPU Package RAPL vs.\ Wireless Radio Energy:} Hardware counters (\nolinkurl{rapl_package0_pkg}) capture CPU/SoC package energy ($\pm 4.3\%\text{--}25.1\%$ $U_{95}$) and exclude Wi-Fi/5G RF transceiver power, which can dominate total energy on battery-powered mobile edge devices.
    \item \textbf{Confidential Corpus Scale Beyond 100k Vectors:} Balanced-$K_c$ SPANN scales confidential CKKS evaluation to $100,000$ vectors ($52.31$~ms server eval, $625$~KB client centroids). Scaling to $10^7\text{--}10^9$ confidential vectors will require hierarchical multi-tier centroids or PIR-protected centroid routing~\cite{henzinger2023tiptoe} to bound client metadata and coarse topic leakage.
    \item \textbf{Cloud ARM64 VMs vs.\ Physical Cellular Edge:} Our ARM64 benchmarks ran on GCP \texttt{t2a-standard-4} (Ampere Altra Neoverse-N1) datacenter VMs, and live WAN trials traversed campus/fiber-to-cloud paths across four temporal windows ($1,800$ transactions) rather than multi-week 5G vehicular mobility.
    \item \textbf{Author Clean-Room vs.\ Independent Third-Party Evaluation:} Clean-room verification reflects automated clean-environment re-execution by the author; all raw logs, C++ OpenFHE bridges, and 64-hex SHA-256 manifests are packaged for Zenodo DOI archival to support external artifact evaluation.
\end{enumerate}

\section{Conclusion}\label{sec:conclusion}
We presented CAPQ-Pareto, a coordinate-confidential post-quantum cloud-edge database orchestration framework. By separating static systems contributions—persistent hybrid PQC pooling ($+52.15$~pp SLA), boundary-replicated 1-CT slot-aggregated SPANN-CKKS ($128.81$~KB wire size, $+0.11\text{--}0.23$ recall over disjoint $r=2$, $52.31$~ms server eval at $100,000$ vectors), and asymmetric \texttt{INT8} compression—from online bandit adaptation, we demonstrated $40.4\%\text{--}40.9\%$ lower latency than the strongest static confidential baseline across live cloud WANs and a $39.8\%$ dynamic regret reduction ($p=0.00195$) over stationary LinUCB during unannounced packet-loss regime inversions.

\section*{Data Availability and Acknowledgment}
All C++/Python source code, raw query logs, and 64-character SHA-256 manifests are available at \url{https://github.com/preecha-msu/CAPQ-Pareto}. Supported by the Faculty of Informatics, Mahasarakham University.

\bibliographystyle{IEEEtran}
\bibliography{references}

\begin{IEEEbiographynophoto}{Preecha Noiumkar}
(Member, IEEE) received the M.Sc. degree in information technology from King Mongkut's Institute of Technology North Bangkok (KMITNB), Thailand. He is currently a Lecturer and Researcher with the Department of Information Technology, Faculty of Informatics, Mahasarakham University, Thailand. His research interests include cloud-edge systems security, post-quantum database orchestration, and neuro-symbolic artificial intelligence.
\end{IEEEbiographynophoto}

\end{document}
"""

SUPP_FULL = r"""\documentclass[10pt,journal,compsoc]{IEEEtran}
\usepackage{microtype}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{url}
\usepackage[hidelinks]{hyperref}
\Urlmuskip=0mu plus 2mu\relax
\emergencystretch=1.8em
\extrafloats{64}

\begin{document}
\title{Electronic Supplementary Material for CAPQ-Pareto (V4): Complete Mathematical Proofs, Extended Figures (S1--S6), Full Empirical Tables (S1--S12), and 64-Hex SHA-256 Registry}
\author{Preecha~Noiumkar,~\IEEEmembership{Member,~IEEE}}
\maketitle

\section{Complete Mathematical Proofs}
\subsection{Proof of Proposition 1 (Variance Envelopes, Non-Isotropy, and Exact Hoeffding Constant)}
Let $\mathbf{x}\in\mathbb{S}^{D-1}$ satisfy $\sum_{j=1}^D x_j^2 = 1$ (arbitrary direction; no isotropy or rotational invariance of $\mathbf{x}$ is assumed). For asymmetric $b$-bit quantization on $[q_{\min}, q_{\max}]$ with step $\Delta_q = (q_{\max}-q_{\min})/(2^b-1)$, write $q_j = q_{\min} + (k_j + u_j)\Delta_q$ where $k_j \in \{0,\dots,2^b-2\}$ and $u_j \in [0,1)$.
Under \emph{unbiased stochastic rounding}, $\widehat{q}_j = q_{\min} + (k_j + \xi_j)\Delta_q$ with $\xi_j \sim \mathrm{Bernoulli}(u_j)$ drawn independently across coordinates $j=1,\dots,D$. Then $e_j = (\xi_j - u_j)\Delta_q$ satisfies $\mathbb{E}[e_j \mid q_j] = 0$ identically and $\mathrm{Var}(e_j \mid q_j) = u_j(1-u_j)\Delta_q^2 \le \Delta_q^2/4$. By coordinate independence:
\begin{equation}
\mathrm{Var}_{\text{SR}}(\langle\mathbf{x},\mathbf{e}\rangle \mid \mathbf{x},\mathbf{q}) = \Delta_q^2 \sum_{j=1}^D x_j^2 u_j(1-u_j).
\end{equation}
Taking expectation over $u_j \sim \mathcal{U}(0,1)$ yields $\mathbb{E}[u_j(1-u_j)] = \int_0^1 (u-u^2)\,du = \frac{1}{6}$, so $\mathbb{E}_{u}[\mathrm{Var}_{\text{SR}}] = \frac{\Delta_q^2}{6}\sum_{j=1}^D x_j^2 = \frac{\Delta_q^2}{6}$. Under \emph{deterministic round-to-nearest} with $e_j \sim \mathcal{U}(-\Delta_q/2, +\Delta_q/2)$ independent, $\mathrm{Var}_{\text{det}}(e_j) = \frac{\Delta_q^2}{12}$, giving $\mathrm{Var}_{\text{det}}(\langle\mathbf{x},\mathbf{e}\rangle \mid \mathbf{x}) = \frac{\Delta_q^2}{12}\sum_{j=1}^D x_j^2 = \frac{\Delta_q^2}{12}$.

\textbf{Exact Hoeffding Support Width:} For deterministic rounding, $Z_j = x_j e_j \in [a_j, b_j] = [-\tfrac{1}{2}\Delta_q|x_j|, +\tfrac{1}{2}\Delta_q|x_j|]$, so $b_j - a_j = \Delta_q|x_j|$. For stochastic rounding, if $x_j \ge 0$, $Z_j \in [-u_j\Delta_q x_j, (1-u_j)\Delta_q x_j]$, whose width is $(1-u_j)\Delta_q x_j - (-u_j\Delta_q x_j) = \Delta_q x_j = \Delta_q|x_j|$ (and identically $\Delta_q|x_j|$ when $x_j < 0$). Hence $\sum_{j=1}^D (b_j - a_j)^2 = \Delta_q^2 \sum_{j=1}^D x_j^2 = \Delta_q^2$. Applying Hoeffding's two-sided inequality:
\begin{align}
\Pr\!\left(\left|\sum_{j=1}^D Z_j\right| \ge t\right) &\le 2\exp\!\left(-\frac{2t^2}{\sum_{j=1}^D (b_j-a_j)^2}\right) \nonumber\\
&= 2\exp\!\left(-\frac{2t^2}{\Delta_q^2}\right) \le \zeta \nonumber\\
&\iff t \ge \Delta_q\sqrt{\frac{\ln(2/\zeta)}{2}}. \quad \blacksquare
\end{align}

\subsection{Proof of Proposition 2 (Adaptive Probe Transcript Entropy \& Miller--Madow Bias Correction)}
Because every inverted list is padded to fixed size $B_{\max}$ and homomorphically aggregated into a fixed $128.81$~KB reply ciphertext, server transcript $O(\mathbf{q}) = (r(\mathbf{q}), \mathcal{B}_{r(\mathbf{q})}(\mathbf{q}))$ is a deterministic function of $\mathbf{q}$, implying $H(O(\mathbf{q})\mid\mathbf{q})=0$ and $I(\mathbf{q}; O(\mathbf{q})) = H(O(\mathbf{q}))$. By the entropy chain rule, $H(r, \mathcal{B}_r) = H(r) + H(\mathcal{B}_r \mid r)$, where binary variable $r\in\{1,2\}$ has $H(r) \le 1$~bit and total support cardinality is $|\mathcal{O}| = \binom{K_c}{1} + \binom{K_c}{2}$, bounding $H(O(\mathbf{q})) \le \log_2\big(K_c + K_c(K_c-1)/2\big)$. For finite sample size $N_q$ over $K_+$ observed states, Taylor expansion of $-\sum_o \hat{p}_o\log_2\hat{p}_o$ around true probabilities $p_o$ yields first-order downward bias $-\frac{K_+ - 1}{2N_q\ln 2} + O(N_q^{-2})$, motivating the Miller--Madow correction $\hat{H}_{\text{MM}} = \hat{H}_{\text{plug}} + \frac{K_+ - 1}{2N_q\ln 2}$. $\blacksquare$

\subsection{Derivation of Centered Russac Discounted LinUCB Variance Sandwich}
Suppose true utility satisfies $u_s = \boldsymbol{\theta}_{s,a}^\top\mathbf{x}_{s^-} + \eta_s$ with conditionally sub-Gaussian noise $\eta_s$. Re-parameterizing $\boldsymbol{\delta}_{s,a} = \boldsymbol{\theta}_{s,a} - \boldsymbol{\theta}_0$ on centered observations $\tilde{u}_s = u_s - \boldsymbol{\theta}_0^\top\mathbf{x}_{s^-}$ yields weighted ridge estimator $\hat{\boldsymbol{\delta}}_{t,a} = \mathbf{A}_{t,a}^{-1}\sum_{s<t:a_s=a}\gamma^{t-1-s}\tilde{u}_s\mathbf{x}_{s^-}$, so $\hat{\boldsymbol{\theta}}_{t,a} = \boldsymbol{\theta}_0 + \hat{\boldsymbol{\delta}}_{t,a}$. Under stationary intervals $\boldsymbol{\theta}_{s,a}=\boldsymbol{\theta}_a^*$, estimation error decomposes into regularization bias $-\lambda\mathbf{A}_{t,a}^{-1}(\boldsymbol{\theta}_a^* - \boldsymbol{\theta}_0)$ (bounded by $\lambda\|\boldsymbol{\theta}_a^*-\boldsymbol{\theta}_0\|_2$ rather than shrinking toward zero utility) and martingale noise $\mathbf{A}_{t,a}^{-1}\sum \gamma^{t-1-s}\eta_s\mathbf{x}_{s^-}$ whose exact quadratic covariance is $\sigma_\eta^2 \mathbf{A}_{t,a}^{-1}(\sum \gamma^{2(t-1-s)}\mathbf{x}_{s^-}\mathbf{x}_{s^-}^\top)\mathbf{A}_{t,a}^{-1} \preceq \sigma_\eta^2 \mathbf{A}_{t,a}^{-1}\tilde{\mathbf{A}}_{t,a}\mathbf{A}_{t,a}^{-1}$ by Russac et al.~(Theorem~1). $\blacksquare$

\section{Extended Supplementary Figures (Figures S1--S6)}
\begin{figure*}[t]
\centering
\includegraphics[width=0.92\textwidth]{figures/fig_s1_v4_ckks_rings_and_mldsa.pdf}
\caption{Figure S1: (a)~OpenFHE CKKS ring/packing latency scaling ($N\in\{4096,8192,16384\}$); (b)~NIST FIPS~204 \texttt{ML-DSA-44/65/87} authentication chain payload and Ethernet MTU frame counts.}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=0.92\textwidth]{figures/fig_s2_v4_raw_socket_crypto_distributions.pdf}
\caption{Figure S2: Raw per-query microsecond socket + native AEAD trace ($N=240$ queries, $236/240$ distinct values, zero synthetic offsets): (a)~Sequential trace; (b)~Encrypt/decrypt microsecond boxplots.}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=0.92\textwidth]{figures/fig_s3_v4_netem_profiles_and_9faults.pdf}
\caption{Figure S3: (a)~Real Linux namespace \texttt{tc netem} socket p95 latencies under RFC~9002 adaptive PTO (\texttt{P0}--\texttt{P4}); (b)~Detection and clean recovery rates across the 9-scenario operational fault/abuse suite.}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=0.92\textwidth]{figures/fig_s4_v4_cloud_economics_and_weights.pdf}
\caption{Figure S4: (a)~Cloud billing cost per 1M queries across four operator budget profiles; (b)~Monotonic latency response across scalarization weight profiles.}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=0.92\textwidth]{figures/fig_s5_v4_wan_temporal_windows_mbb.pdf}
\caption{Figure S5: Multi-window temporal stability of CAPQ-Pareto across Bangkok, Singapore, and US-Central live cloud endpoints.}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=0.92\textwidth]{figures/fig_s6_v4_bandit_350_seed_distributions.pdf}
\caption{Figure S6: (a)~10-seed portfolio regret distributions ($350$ controller-scenario runs); (b)~Proposition~1 empirical inner-product error scaling across quantization bit-widths $b\in\{4,6,8,10\}$.}
\end{figure*}

\section{Complete Empirical Tables (Tables S1--S9)}
\input{tables/tab_s01_10mode_ablation.tex}
\input{tables/tab_s02_full_15term_glm.tex}
\input{tables/tab_s03_full_35row_bandit.tex}
\input{tables/tab_s04_prop1_extended.tex}
\clearpage
\input{tables/tab_s05_leakage_and_mm.tex}
\input{tables/tab_s06_operational_faults.tex}
\input{tables/tab_s07_wan_multiepoch.tex}
\input{tables/tab_s08_cross_isa_stats.tex}
\input{tables/tab_s09_weights_and_cleanroom.tex}

\clearpage
\section{Master 64-Hex SHA-256 Verification Manifests and Deprecated Errata Registry (Tables S10--S12)}
\input{tables/tab_s10_sha256_v4_active.tex}
\input{tables/tab_s11_sha256_foundational_active.tex}
\input{tables/tab_s12_deprecated_errata.tex}

\end{document}
"""

COVER_TEX = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{hyperref}
\begin{document}
\noindent November 26, 2026\\[1em]
\textbf{To:} Editor-in-Chief and Associate Editor, \textit{IEEE Transactions on Cloud Computing}\\
\textbf{Subject:} Submission of Revised Regular Paper (V4): \textit{``CAPQ-Pareto: Coordinate-Confidential Hybrid Cloud Vector-Relational Orchestration Under Post-Quantum Transport and Non-Stationary WANs''}\\[1em]
Dear Editor-in-Chief and Associate Editor,\\[0.6em]
Please find enclosed the thoroughly revised V4 manuscript and complete Electronic Supplementary Material for \textbf{CAPQ-Pareto}. In response to the Reviewer's rigorous evaluation of V3, we audited every script, mathematical constant, and hardware log from first principles under a strict \textbf{Zero-Estimation / 100\% Real Execution} protocol:
\begin{enumerate}
    \item \textbf{Bandit Redemption (P0.1):} Identified and fixed prior-shrinkage and minimization ridge-to-zero bugs in V3's D-LinUCB implementation, re-running 10-seed counterfactual replay benchmarks over real Linux \texttt{tc netem} sockets against a single global fixed baseline and stationary LinUCB ($-39.8\%$ dynamic regret under unannounced link-loss inversions, $p=0.00195$, while candidly reporting parity on stationary/observable-EWMA links).
    \item \textbf{Equal-Probe \texttt{Disjoint r=2} Control, Wire Reconciliation, and 100k Scaling (P0.2, P0.5):} Withdrew the erroneous $8.0/4.0$~KB buffer figure in favor of verified C++ OpenFHE serialized sizes ($128.81$~KB request / $128.81$~KB 1-CT slot-aggregated response), added the controlled \texttt{Disjoint r=2} comparison ($+0.11\text{--}0.23$ Recall@10 gain for SPANN, $p=8.46\times 10^{-4}$), scaled confidential CKKS to $100,000$ vectors ($52.31$~ms server eval), and replaced OLS ANOVA with a block-fixed Binomial Logit GLM (HC3).
    \item \textbf{Fair Confidential WAN Baselines \& Quiescent RAPL $\pm U_{95}$ (P0.3, P0.4, P1, P2):} Added fair pooled CKKS baselines (\texttt{M6}, \texttt{M7}), reported pre-registered \texttt{Strict-150ms} and \texttt{Class-250ms} SLAs side-by-side ($40.4\%\text{--}40.9\%$ lower latency than \texttt{M7}, $p<10^{-4}$), unified 10,000-query Proposition~1 audits with the exact $\Delta_q\sqrt{\tfrac{1}{2}\ln(2/\zeta)}$ Hoeffding bound, measured quiescent-settled dual-ring RAPL power, and verified 64-character SHA-256 digests across all 54 active artifacts (with 4 superseded V1/V3 logs segregated into an explicit Errata Registry).
\end{enumerate}
All authors confirm this work is original and not under consideration elsewhere.\\[1em]
Sincerely,\\[0.6em]
\textbf{Preecha Noiumkar} (Member, IEEE)\\
Department of Information Technology, Faculty of Informatics, Mahasarakham University, Thailand\\
E-mail: \texttt{preecha.n@msu.ac.th}
\end{document}
"""


def compile_wsl(tex_name: str) -> tuple[Path, int]:
    stem = tex_name[:-4]
    wsl_v4 = subprocess.check_output(["wsl", "-u", "root", "-e", "wslpath", "-u", str(V4_MS)], text=True).strip()
    sh = (
        f"set -e; rm -rf /tmp/capq_full_{stem} && mkdir -p /tmp/capq_full_{stem}; "
        f"cp -a '{wsl_v4}/.' /tmp/capq_full_{stem}/; cd /tmp/capq_full_{stem}; "
        f"pdflatex -interaction=nonstopmode -halt-on-error {stem}.tex > build.log 2>&1 || (tail -n 50 build.log && exit 1); "
        f"if grep -q '\\\\bibdata' {stem}.aux 2>/dev/null; then bibtex {stem} >> build.log 2>&1 || (cat {stem}.blg && exit 1); fi; "
        f"pdflatex -interaction=nonstopmode -halt-on-error {stem}.tex >> build.log 2>&1 || (tail -n 50 build.log && exit 1); "
        f"pdflatex -interaction=nonstopmode -halt-on-error {stem}.tex >> build.log 2>&1 || (tail -n 50 build.log && exit 1); "
        f"cp {stem}.pdf {stem}.log '{wsl_v4}/'"
    )
    res = subprocess.run(["wsl", "-u", "root", "-e", "bash", "-lc", sh], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"LaTeX build failed for {tex_name}:\n{res.stdout}\n{res.stderr}")
    pdf_p = V4_MS / f"{stem}.pdf"
    ltxt = (V4_MS / f"{stem}.log").read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"Output written on .*?\((\d+) pages?", ltxt)
    return pdf_p, (int(m.group(1)) if m else -1)


def verify_gates(pg_main: int, pg_supp: int) -> None:
    abs_m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", MAIN_FULL, re.S)
    assert abs_m is not None
    wc = len(abs_m.group(1).strip().split())
    assert 100 <= wc <= 200, f"Abstract word count {wc} outside 100--200!"

    # Verify all 14 figures & 22 tables are actually referenced/included!
    inc_figs = re.findall(r"\\includegraphics(?:\[.*?\])?\{figures/([^}]+)\}", MAIN_FULL + SUPP_FULL)
    inc_tabs = re.findall(r"\\input\{tables/([^}]+)\}", MAIN_FULL + SUPP_FULL)
    assert len(set(inc_figs)) == 14, f"Expected all 14 V4 figures included, found {len(set(inc_figs))}: {inc_figs}"
    assert len(set(inc_tabs)) == 22, f"Expected all 22 V4 tables included, found {len(set(inc_tabs))}"

    for tf in V4_MS.rglob("*.tex"):
        bad63 = re.findall(r"(?<![0-9a-fA-F])[0-9a-fA-F]{63}(?![0-9a-fA-F])", tf.read_text(encoding="utf-8"))
        assert not bad63, f"63-hex truncation detected in {tf.name}: {bad63}"

    for lf in (V4_MS / "main.log", V4_MS / "supplement.log"):
        ltxt = lf.read_text(encoding="utf-8", errors="ignore")
        over = [float(m.group(1)) for m in re.finditer(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", ltxt)]
        sev = [x for x in over if x > 8.0]
        assert not sev, f"Overfull >8pt in {lf.name}: {sev}"
        assert "LaTeX Warning: Reference" not in ltxt and "Citation `" not in ltxt, f"Undefined ref/cite in {lf.name}!"
        print(f"  [GATE PASS] {lf.name:<16} | Undefined=0 | SevereOverfull(>8pt)=0 (max={max(over, default=0.0):.2f}pt)")

    assert 9 <= pg_main <= 12, f"Unexpected main.pdf page count: {pg_main}"
    assert pg_supp >= 7, f"Unexpected supplement.pdf page count: {pg_supp}"
    print(f"  [COMPLETENESS GATE PASS] Abstract={wc}w | Included Vector Figs=14/14 | Included Tables=22/22 | Main={pg_main}p | Supp={pg_supp}p")


if __name__ == "__main__":
    (V4_MS / "references.bib").write_text(BIB_FULL.strip() + "\n", encoding="utf-8", newline="\n")
    (V4_MS / "main.tex").write_text(MAIN_FULL.strip() + "\n", encoding="utf-8", newline="\n")
    (V4_MS / "supplement.tex").write_text(SUPP_FULL.strip() + "\n", encoding="utf-8", newline="\n")
    (V4_MS / "cover_letter_v4.tex").write_text(COVER_TEX.strip() + "\n", encoding="utf-8", newline="\n")

    print("=== [STEP C.2] COMPILING FULL REGULAR PAPER MAIN, SUPPLEMENT, REBUTTAL & COVER LETTER ===")
    p_main, pg_m = compile_wsl("main.tex")
    if pg_m > 12:
        mt = (V4_MS / "main.tex").read_text(encoding="utf-8")
        tight = "\n\\setlength{\\textfloatsep}{6pt plus 2pt minus 2pt}\n\\setlength{\\floatsep}{5pt plus 2pt minus 2pt}\n\\setlength{\\dbltextfloatsep}{6pt plus 2pt minus 2pt}\n\\renewcommand{\\baselinestretch}{0.965}\n"
        mt = mt.replace("\\begin{document}", tight + "\\begin{document}", 1)
        (V4_MS / "main.tex").write_text(mt, encoding="utf-8", newline="\n")
        p_main, pg_m = compile_wsl("main.tex")
    p_supp, pg_s = compile_wsl("supplement.tex")
    p_reb, pg_r  = compile_wsl("response_to_reviewers_v4.tex")
    p_cov, pg_c  = compile_wsl("cover_letter_v4.tex")
    verify_gates(pg_m, pg_s)

    d_res = Path(r"D:\Research")
    shutil.copy2(p_main, d_res / "CAPQ-Pareto-Main-IEEE-TCC-V6.pdf")
    shutil.copy2(p_supp, d_res / "CAPQ-Pareto-Supplement-IEEE-TCC-V6.pdf")
    shutil.copy2(p_reb,  d_res / "CAPQ-Pareto-Response-to-Reviewers-V6.pdf")
    shutil.copy2(p_cov,  d_res / "CAPQ-Pareto-Cover-Letter-IEEE-TCC-V6.pdf")

    zip_fp = d_res / "CAPQ-Pareto-Overleaf-IEEE-TCC-V6.zip"
    with zipfile.ZipFile(zip_fp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fn in ["main.tex", "supplement.tex", "response_to_reviewers_v4.tex", "cover_letter_v4.tex", "references.bib", "CAPQ-Pareto-SHA256-Manifest-V6.txt"]:
            zf.write(V4_MS / fn, arcname=fn)
        for tf in sorted(TAB.glob("*.tex")):
            zf.write(tf, arcname=f"tables/{tf.name}")
        for ff in sorted(FIG.glob("*.pdf")):
            zf.write(ff, arcname=f"figures/{ff.name}")

    # Append final deliverable hashes into D:\Research\CAPQ-Pareto-SHA256-Manifest-V6.txt
    base_manifest = (V4_MS / "CAPQ-Pareto-SHA256-Manifest-V6.txt").read_text(encoding="utf-8")
    deliv_lines = ["\n# SECTION D: FINAL EXPORTED DELIVERABLES IN D:\\Research\\"]
    deliverables = [
        d_res / "CAPQ-Pareto-Main-IEEE-TCC-V6.pdf",
        d_res / "CAPQ-Pareto-Supplement-IEEE-TCC-V6.pdf",
        d_res / "CAPQ-Pareto-Response-to-Reviewers-V6.pdf",
        d_res / "CAPQ-Pareto-Cover-Letter-IEEE-TCC-V6.pdf",
        d_res / "CAPQ-Pareto-Fig1-Preview-V6.png",
        zip_fp,
    ]
    for dfp in deliverables:
        deliv_lines.append(f"{sha64(dfp)}  {dfp.name}")
    full_manifest_txt = base_manifest + "\n".join(deliv_lines) + "\n"
    (d_res / "CAPQ-Pareto-SHA256-Manifest-V6.txt").write_text(full_manifest_txt, encoding="utf-8", newline="\n")
    (V4_MS / "CAPQ-Pareto-SHA256-Manifest-V6.txt").write_text(full_manifest_txt, encoding="utf-8", newline="\n")

    print(f"\n=== [COMPLETE V4 PACKAGE READY IN D:\\Research\\ | MAIN={pg_m}p | SUPP={pg_s}p | REBUTTAL={pg_r}p | COVER={pg_c}p] ===")
    for dfp in deliverables + [d_res / "CAPQ-Pareto-SHA256-Manifest-V6.txt"]:
        print(f"  {dfp.name:<46} | {dfp.stat().st_size:9,d} B | SHA256={sha64(dfp)}")