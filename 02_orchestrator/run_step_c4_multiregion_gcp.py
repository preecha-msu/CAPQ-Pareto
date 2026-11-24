from __future__ import annotations
import base64
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(r"D:\Research\CAPQ-Pareto-V2")
sys.path.insert(0, str(REPO_ROOT / "05_telemetry_audit"))
sys.path.insert(0, str(REPO_ROOT / "05_telemetry_audit" / "energy_rapl_scaphandre"))
sys.path.insert(0, str(REPO_ROOT / "02_orchestrator" / "pareto_math_model"))

import sha256_manifest as sm
import math_stats_core as msc
from hardware_energy_meter import RealHardwareEnergyMeter

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10,
    "legend.fontsize": 7.8,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ALPHA, BETA, GAMMA = 0.34, 0.33, 0.33
MIN_VEC_RECALL = 0.92

# Published GCP hourly USD rates by machine family & region (converted to exact per-second billing)
HOURLY_USD_TABLE = {
    "gcp_bangkok_ase3": {
        "e2-standard-4": 0.16952, "n4-standard-4": 0.23960, "n2-standard-4": 0.24540,
        "n2d-standard-4": 0.21310, "c3-standard-4": 0.26620, "c4-standard-4": 0.29800,
        "n4-standard-2": 0.11980, "e2-standard-2": 0.08476, "n2-standard-2": 0.12270,
    },
    "gcp_singapore_ase1": {
        "e2-standard-4": 0.15881, "n4-standard-4": 0.22840, "n2-standard-4": 0.23368,
        "n2d-standard-4": 0.20292, "c3-standard-4": 0.25360, "c4-standard-4": 0.28400,
        "n4-standard-2": 0.11420, "e2-standard-2": 0.07941, "n2-standard-2": 0.11684,
    },
    "gcp_iowa_usc1": {
        "e2-standard-4": 0.13401, "n4-standard-4": 0.18976, "n2-standard-4": 0.19424,
        "n2d-standard-4": 0.16868, "c3-standard-4": 0.20880, "c4-standard-4": 0.23760,
        "n4-standard-2": 0.09488, "e2-standard-2": 0.06701, "n2-standard-2": 0.09712,
    },
}

GCP_REGIONS = [
    {
        "id": "gcp_bangkok_ase3",
        "label": "Bangkok (asia-southeast3)",
        "vm_name": "capq-bkk-ase3",
        "zones": ["asia-southeast3-a", "asia-southeast3-b", "asia-southeast3-c"],
        "egress_usd_per_gib": 0.12,
        "sla_deadline_ms": 150.0,
    },
    {
        "id": "gcp_singapore_ase1",
        "label": "Singapore (asia-southeast1)",
        "vm_name": "capq-sg-ase1",
        "zones": ["asia-southeast1-b", "asia-southeast1-a", "asia-southeast1-c"],
        "egress_usd_per_gib": 0.12,
        "sla_deadline_ms": 150.0,
    },
    {
        "id": "gcp_iowa_usc1",
        "label": "US-Central (us-central1)",
        "vm_name": "capq-usa-usc1",
        "zones": ["us-central1-a", "us-central1-b", "us-central1-c"],
        "egress_usd_per_gib": 0.12,
        "sla_deadline_ms": 350.0,
    },
]

PREFERRED_TYPES = [
    "e2-standard-4", "n4-standard-4", "n2-standard-4", "n2d-standard-4",
    "c3-standard-4", "c4-standard-4", "n4-standard-2", "e2-standard-2", "n2-standard-2",
]

ACTIONS = {
    "cls_tls_sql":  {"transport": "tcp_nores", "kem": "X25519",            "cipher": "AES-256-GCM", "q_mode": "none",        "batch_b": 1},
    "cls_tls_vec":  {"transport": "tcp_nores", "kem": "X25519",            "cipher": "AES-256-GCM", "q_mode": "fp32",        "batch_b": 1},
    "stat_pqc_sql": {"transport": "tcp_nores", "kem": "ML-KEM-1024",       "cipher": "AES-256-GCM", "q_mode": "none",        "batch_b": 1},
    "stat_pqc_vec": {"transport": "tcp_nores", "kem": "ML-KEM-1024",       "cipher": "AES-256-GCM", "q_mode": "fp32",        "batch_b": 1},
    "reg_a_b1":     {"transport": "quic_0rtt", "kem": "X25519_ML-KEM-768", "cipher": "AES-256-GCM", "q_mode": "none",        "batch_b": 1},
    "reg_a_b16":    {"transport": "quic_0rtt", "kem": "X25519_ML-KEM-768", "cipher": "AES-256-GCM", "q_mode": "none",        "batch_b": 16},
    "reg_b_fp32":   {"transport": "quic_0rtt", "kem": "ML-KEM-512",        "cipher": "AES-256-GCM", "q_mode": "fp32",        "batch_b": 1},
    "reg_b_int8":   {"transport": "quic_0rtt", "kem": "ML-KEM-512",        "cipher": "AES-256-GCM", "q_mode": "int8_sq",     "batch_b": 1},
    "reg_b_1bit":   {"transport": "quic_0rtt", "kem": "ML-KEM-512",        "cipher": "AES-256-GCM", "q_mode": "binary_1bit", "batch_b": 1},
    "reg_c_ckks4k": {"transport": "quic_0rtt", "kem": "ML-KEM-768",        "cipher": "AES-256-GCM", "q_mode": "ckks_4096",   "batch_b": 1},
}


def wsl_cmd(args: list[str], check: bool = True, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    res = subprocess.run(["wsl", "-u", "root", "-e", *args], input=input_bytes, check=False, capture_output=True)
    if check and res.returncode != 0:
        raise RuntimeError(
            f"Cmd failed ({res.returncode}): {' '.join(args)}\n"
            f"STDOUT: {res.stdout.decode('utf-8', 'replace')}\nSTDERR: {res.stderr.decode('utf-8', 'replace')}"
        )
    return res


def resolve_ubuntu_x86_family() -> str:
    for fam in ["ubuntu-2404-lts-amd64", "ubuntu-2204-lts"]:
        r = wsl_cmd(["gcloud", "compute", "images", "describe-from-family", fam, "--project=ubuntu-os-cloud"], check=False)
        if r.returncode == 0:
            return fam
    return "ubuntu-2204-lts"


def discover_candidates_in_region(zones: list[str], proj: str) -> list[tuple[str, str]]:
    res = wsl_cmd([
        "gcloud", "compute", "machine-types", "list", f"--project={proj}",
        f"--zones={','.join(zones)}", "--format=json(name,zone)",
    ])
    items = json.loads(res.stdout.decode("utf-8"))
    avail_by_zone: dict[str, set[str]] = {z: set() for z in zones}
    for it in items:
        z_short = it["zone"].split("/")[-1]
        if z_short in avail_by_zone:
            avail_by_zone[z_short].add(it["name"])

    candidates: list[tuple[str, str]] = []
    for mtype in PREFERRED_TYPES:
        for z in zones:
            if mtype in avail_by_zone.get(z, set()):
                candidates.append((z, mtype))
    return candidates


def provision_single_region(reg: dict, proj: str, img_family: str, cached_fhe_bytes: bytes | None) -> tuple[dict, bytes | None]:
    vm_name = reg["vm_name"]
    candidates = discover_candidates_in_region(reg["zones"], proj)
    if not candidates:
        raise RuntimeError(f"No standard 2/4-vCPU x86_64 machine types discovered in {reg['zones']}")

    active_zone = None
    chosen_mtype = None
    ext_ip = None
    last_err = ""

    for z, mtype in candidates:
        print(f"  -> Trying {vm_name} ({mtype}) in {z} ...")
        # Omit --boot-disk-type so GCP selects hyperdisk-balanced (for n4/c4) or pd-balanced (for e2/n2) automatically!
        res = wsl_cmd([
            "gcloud", "compute", "instances", "create", vm_name,
            f"--project={proj}", f"--zone={z}",
            f"--machine-type={mtype}",
            f"--image-family={img_family}", "--image-project=ubuntu-os-cloud",
            "--boot-disk-size=30GB",
            "--tags=capq-cloud-node",
            "--metadata=startup-script=shutdown -P +60",
            "--format=json", "--quiet",
        ], check=False)
        if res.returncode == 0:
            info = json.loads(res.stdout.decode("utf-8"))[0]
            ext_ip = info["networkInterfaces"][0]["accessConfigs"][0]["natIP"]
            active_zone = z
            chosen_mtype = mtype
            break
        last_err = res.stderr.decode("utf-8", "replace").strip().splitlines()[-1] if res.stderr else "unknown error"
        print(f"     [SKIP {z}/{mtype}] {last_err}")

    if not active_zone or not chosen_mtype or not ext_ip:
        raise RuntimeError(f"Exhausted all candidate zones/types for {vm_name}. Last error: {last_err}")

    hourly_rate = HOURLY_USD_TABLE[reg["id"]].get(chosen_mtype, 0.20)
    print(f"[PROVISIONED] {reg['label']} | VM={vm_name} ({chosen_mtype} @ ${hourly_rate:.4f}/hr) | Zone={active_zone} | IP={ext_ip}")

    ssh_opts = [
        "--quiet", "--ssh-flag=-T", "--ssh-flag=-o StrictHostKeyChecking=no",
        "--ssh-flag=-o ConnectTimeout=15", "--ssh-flag=-o ServerAliveInterval=25",
    ]
    for _ in range(25):
        r = wsl_cmd(["gcloud", "compute", "ssh", f"capqbench@{vm_name}", f"--zone={active_zone}", f"--project={proj}", *ssh_opts, "--command=sudo mkdir -p /opt/capq_cloud /opt/capq_bridge"], check=False)
        if r.returncode == 0:
            break
        time.sleep(5.0)
    else:
        raise RuntimeError(f"SSH readiness timeout on {vm_name}")

    if cached_fhe_bytes is not None:
        wsl_cmd(["gcloud", "compute", "ssh", f"capqbench@{vm_name}", f"--zone={active_zone}", f"--project={proj}", *ssh_opts, "--command=sudo tee /tmp/openfhe_x86_cache.tar.gz >/dev/null"], input_bytes=cached_fhe_bytes)

    sift_bytes = (REPO_ROOT / "01_workloads/unstructured_vectors/sift1m_rag_embeddings/sift_128d_real_5k_200q.npz").read_bytes()
    wsl_cmd(["gcloud", "compute", "ssh", f"capqbench@{vm_name}", f"--zone={active_zone}", f"--project={proj}", *ssh_opts, "--command=sudo tee /opt/capq_cloud/sift_128d_real_5k_200q.npz >/dev/null"], input_bytes=sift_bytes)

    cpp_bytes = wsl_cmd(["cat", "/opt/capq_bridge/capq_openfhe_bridge.cpp"]).stdout
    wsl_cmd(["gcloud", "compute", "ssh", f"capqbench@{vm_name}", f"--zone={active_zone}", f"--project={proj}", *ssh_opts, "--command=sudo tee /opt/capq_bridge/capq_openfhe_bridge.cpp >/dev/null"], input_bytes=cpp_bytes)

    daemon_bytes = (REPO_ROOT / "04_testbed_infra/cloud_region_daemon.py").read_bytes().replace(b"\r\n", b"\n")
    wsl_cmd(["gcloud", "compute", "ssh", f"capqbench@{vm_name}", f"--zone={active_zone}", f"--project={proj}", *ssh_opts, "--command=sudo tee /opt/capq_cloud/cloud_region_daemon.py >/dev/null"], input_bytes=daemon_bytes)

    boot_bytes = (REPO_ROOT / "04_testbed_infra/bootstrap_cloud_region.sh").read_bytes().replace(b"\r\n", b"\n")
    print(f"[BOOTSTRAPPING] Installing PostgreSQL+pgvector+OpenFHE & starting QUIC/TLS daemon on {vm_name}...")
    wsl_cmd(["gcloud", "compute", "ssh", f"capqbench@{vm_name}", f"--zone={active_zone}", f"--project={proj}", *ssh_opts, "--command=tr -d '\\r' > /tmp/boot.sh && sudo bash /tmp/boot.sh"], input_bytes=boot_bytes)

    if cached_fhe_bytes is None:
        r_b64 = wsl_cmd(["gcloud", "compute", "ssh", f"capqbench@{vm_name}", f"--zone={active_zone}", f"--project={proj}", *ssh_opts, "--command=sudo base64 -w0 /tmp/openfhe_built_bundle.tar.gz"], check=False)
        if r_b64.returncode == 0 and r_b64.stdout.strip():
            b64_line = r_b64.stdout.decode("utf-8", "replace").strip().splitlines()[-1].strip()
            cached_fhe_bytes = base64.b64decode(b64_line)
            print(f"[CACHED] Captured portable OpenFHE binary bundle ({len(cached_fhe_bytes) // 1024} KB) for Regions 2 & 3.")

    reg_live = {
        **reg,
        "zone": active_zone,
        "machine_type": chosen_mtype,
        "vm_usd_per_sec": hourly_rate / 3600.0,
        "ext_ip": ext_ip,
    }
    return reg_live, cached_fhe_bytes


def main() -> None:
    t_all = time.perf_counter()
    proj = wsl_cmd(["gcloud", "config", "get-value", "project"]).stdout.decode("utf-8").strip()
    img_family = resolve_ubuntu_x86_family()
    print(f"=== [STEP C4] SEQUENTIAL 3-REGION GCP LIVE WAN AUDIT | Project={proj} | Image={img_family} ===")

    meter = RealHardwareEnergyMeter()
    edge_proc = None
    trial_rows = []
    query_rows = []
    cached_fhe_bytes: bytes | None = None

    try:
        err_f = open(REPO_ROOT / "06_experiments_results/raw_logs/multiregion_edge_stderr.log", "w", encoding="utf-8")
        edge_proc = subprocess.Popen(
            ["wsl", "-u", "root", "-e", "env", "OMP_WAIT_POLICY=PASSIVE", "python3", "-u", "/mnt/d/Research/CAPQ-Pareto-V2/04_testbed_infra/edge_multiregion_client.py"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err_f, text=True,
        )
        assert edge_proc.stdin is not None and edge_proc.stdout is not None
        if edge_proc.stdout.readline().strip() != "EDGE_CLIENT_READY":
            raise RuntimeError("Local Edge Gateway client failed to start.")

        def edge_call(payload: dict) -> dict:
            assert edge_proc is not None and edge_proc.stdin is not None and edge_proc.stdout is not None
            edge_proc.stdin.write(json.dumps(payload) + "\n")
            edge_proc.stdin.flush()
            line = edge_proc.stdout.readline()
            if not line:
                raise RuntimeError("Edge client EOF")
            resp = json.loads(line)
            if "worker_error" in resp:
                raise RuntimeError(resp["worker_error"])
            return resp

        time.sleep(0.5)
        p_idle_mw = meter.calibrate_idle_power(2.0)
        e_chk = meter.read_raw_counter()
        _ = edge_call({"op": "idle_sleep", "sec": 0.2})
        _, _, _ = meter.measure_window_net_mj(e_chk, time.perf_counter() - 0.2)
        print(f"[PRE-FLIGHT OK] Edge Client + RAPL Meter verified | Idle Power = {p_idle_mw:.2f} mW")

        schedulers = ["Classical_TLS1.3", "Static_MLKEM1024", "Rule_Heuristic", "CAPQ_Pareto"]
        sql_adaptive = ["reg_a_b1", "reg_a_b16"]
        vec_adaptive = ["reg_b_fp32", "reg_b_int8", "reg_b_1bit", "reg_c_ckks4k"]

        for reg_spec in GCP_REGIONS:
            reg_live = None
            try:
                print(f"\n--- [REGION LIFECYCLE START] {reg_spec['label']} ---")
                reg_live, cached_fhe_bytes = provision_single_region(reg_spec, proj, img_family, cached_fhe_bytes)

                _ = meter.calibrate_idle_power(1.5)
                conn_info = edge_call({"op": "set_target", "ip": reg_live["ext_ip"]})
                med_rtt = float(conn_info["median_rtt_ms"])
                t_backend = conn_info["transport_backend"]
                t_max = float(reg_live["sla_deadline_ms"])
                print(f"[CONNECTED] {reg_live['label']} ({reg_live['machine_type']}) | Real WAN RTT = {med_rtt:.2f} ms | Transport = {t_backend} | SLA = {t_max:.0f} ms")

                bandit_sql = msc.RegularizedLinUCBScheduler(sql_adaptive, d_c=10, nu=0.20)
                bandit_vec = msc.RegularizedLinUCBScheduler(vec_adaptive, d_c=10, nu=0.20)

                emp_p95, emp_acc = {}, {}
                w_lats, w_engs, w_csts = [], [], []
                for m_key in ["reg_a_b1", "reg_a_b16", "reg_b_fp32", "reg_b_int8", "reg_b_1bit", "reg_c_ckks4k", "stat_pqc_sql"]:
                    cfg_m = ACTIONS[m_key]
                    kind_m = "sql" if m_key in ("reg_a_b1", "reg_a_b16", "stat_pqc_sql") else "vec"
                    w_tasks = [{"kind": kind_m, "idx": i * 7, **cfg_m} for i in range(3)]
                    e0 = meter.read_raw_counter()
                    t0 = time.perf_counter()
                    w_res = edge_call({"op": "run_batch", "tasks": w_tasks})["results"]
                    net_mj, _, _ = meter.measure_window_net_mj(e0, t0)
                    lats_m = [float(x["e2e_ms"]) for x in w_res]
                    emp_p95[m_key] = float(np.percentile(lats_m, 95))
                    emp_acc[m_key] = float(np.mean([float(x["raw_acc"]) * (1.0 if x["e2e_ms"] <= t_max else 0.0) for x in w_res]))
                    pq_e = (net_mj / len(w_tasks)) / float(cfg_m["batch_b"])
                    for x in w_res:
                        c_u = (
                            reg_live["vm_usd_per_sec"] * (1e-3 * float(x["cloud_ms"]) / float(cfg_m["batch_b"]))
                            + reg_live["egress_usd_per_gib"] * ((2.0 ** -30) * (float(x["wire_out"]) / float(cfg_m["batch_b"])))
                        ) * 1e6
                        w_lats.append(float(x["e2e_ms"])); w_engs.append(pq_e); w_csts.append(c_u)

                l_min, l_max = min(w_lats), max(w_lats)
                e_min, e_max = min(w_engs), max(w_engs)
                c_min, c_max = min(w_csts), max(w_csts)

                for trial in range(1, 31):
                    rng_t = np.random.default_rng(30000 + trial * 19)
                    base_qs = [
                        {"kind": "sql" if q_i % 2 == 0 else "vec", "idx": int(rng_t.integers(0, 200)), "ztrust": (q_i == 3 and med_rtt < 55.0)}
                        for q_i in range(4)
                    ]
                    for sched in schedulers:
                        tasks, chosen = [], []
                        ctx = np.array([1.0, 0.5, 0.125, 0.125, 0.5, 0.5, min(med_rtt / 250.0, 1.5), 0.05, ALPHA, BETA], dtype=np.float64)
                        for bq in base_qs:
                            kind = bq["kind"]
                            if sched == "Classical_TLS1.3":
                                act = "cls_tls_sql" if kind == "sql" else "cls_tls_vec"
                            elif sched == "Static_MLKEM1024":
                                act = "stat_pqc_sql" if kind == "sql" else "stat_pqc_vec"
                            elif sched == "Rule_Heuristic":
                                act = "reg_a_b16" if kind == "sql" else ("reg_c_ckks4k" if bq["ztrust"] else "reg_b_1bit")
                            else:
                                if kind == "sql":
                                    sla_ok = [m for m in sql_adaptive if emp_p95[m] <= t_max]
                                    act = bandit_sql.select_action(ctx, sql_adaptive, sla_ok, emp_p95)
                                else:
                                    pool = ["reg_c_ckks4k"] if bq["ztrust"] else ["reg_b_fp32", "reg_b_int8", "reg_b_1bit"]
                                    sla_ok = [m for m in pool if emp_p95[m] <= t_max and emp_acc[m] >= MIN_VEC_RECALL]
                                    act = bandit_vec.select_action(ctx, pool, sla_ok, emp_p95)
                            chosen.append((act, kind))
                            tasks.append({"kind": kind, "idx": bq["idx"], **ACTIONS[act]})

                        e0 = meter.read_raw_counter()
                        t0 = time.perf_counter()
                        res_list = edge_call({"op": "run_batch", "tasks": tasks})["results"]
                        net_mj, _, _ = meter.measure_window_net_mj(e0, t0)

                        q_lats, q_engs, q_csts, q_accs, q_slas = [], [], [], [], []
                        for j, r_item in enumerate(res_list):
                            act_j, kind_j = chosen[j]
                            bf = float(ACTIONS[act_j]["batch_b"])
                            lj = float(r_item["e2e_ms"])
                            ej = (net_mj / float(len(tasks))) / bf
                            cj = (
                                reg_live["vm_usd_per_sec"] * (1e-3 * float(r_item["cloud_ms"]) / bf)
                                + reg_live["egress_usd_per_gib"] * ((2.0 ** -30) * (float(r_item["wire_out"]) / bf))
                            ) * 1e6
                            ok_dl = lj <= t_max
                            aj = float(r_item["raw_acc"]) if ok_dl else 0.0
                            sla_ok_j = bool(ok_dl and (aj >= (1.0 if kind_j == "sql" else MIN_VEC_RECALL)))

                            jh = msc.compute_pareto_objective(
                                msc.normalize_metric(lj, l_min, l_max),
                                msc.normalize_metric(ej, e_min, e_max),
                                msc.normalize_metric(cj, c_min, c_max),
                                ALPHA, BETA, GAMMA,
                            )
                            if sched == "CAPQ_Pareto":
                                pen = 0.0 if sla_ok_j else 1.0
                                if kind_j == "sql":
                                    bandit_sql.update(act_j, ctx, jh + pen)
                                else:
                                    bandit_vec.update(act_j, ctx, jh + pen)

                            q_lats.append(lj); q_engs.append(ej); q_csts.append(cj); q_accs.append(aj); q_slas.append(sla_ok_j)
                            query_rows.append({
                                "region": reg_live["id"], "machine_type": reg_live["machine_type"],
                                "trial": trial, "scheduler": sched, "action": act_j,
                                "e2e_ms": round(lj, 4), "cloud_ms": round(float(r_item["cloud_ms"]), 4),
                                "cloud_cpu_ms": round(float(r_item["cloud_cpu_ms"]), 4),
                                "wire_bytes": int((r_item["wire_in"] + r_item["wire_out"]) // int(bf)),
                                "sla_pass": int(sla_ok_j),
                            })

                        ml, me, mc = float(np.mean(q_lats)), float(np.mean(q_engs)), float(np.mean(q_csts))
                        sr = float(np.mean(q_slas))
                        j_raw = msc.compute_pareto_objective(
                            msc.normalize_metric(ml, l_min, l_max),
                            msc.normalize_metric(me, e_min, e_max),
                            msc.normalize_metric(mc, c_min, c_max),
                            ALPHA, BETA, GAMMA,
                        )
                        trial_rows.append({
                            "region": reg_live["id"],
                            "machine_type": reg_live["machine_type"],
                            "median_rtt_ms": med_rtt,
                            "transport_backend": t_backend,
                            "trial": trial,
                            "scheduler": sched,
                            "mean_latency_ms": round(ml, 4),
                            "edge_energy_mj": round(me, 4),
                            "gcp_cost_usd_1m": round(mc, 4),
                            "effective_accuracy": round(float(np.mean(q_accs)), 4),
                            "sla_compliance_rate": round(sr, 4),
                            "trial_sla_pass": int(sr >= 0.75),
                            "j_sla_score": round(j_raw + (1.0 - sr), 5),
                        })

                print(f"[COMPLETED] {reg_live['label']} | 120 paired trials (480 live WAN queries) recorded.")
            finally:
                if reg_live is not None:
                    print(f"[TEARDOWN REGION] Deleting {reg_live['vm_name']} in {reg_live['zone']} immediately...")
                    wsl_cmd(["gcloud", "compute", "instances", "delete", reg_live["vm_name"], f"--zone={reg_live['zone']}", f"--project={proj}", "--quiet"], check=False)

    finally:
        if edge_proc is not None:
            try:
                if edge_proc.stdin:
                    edge_proc.stdin.write("QUIT\n")
                    edge_proc.stdin.flush()
            except Exception:
                pass
            edge_proc.terminate()
        meter.close()
        wsl_cmd(["bash", "/mnt/d/Research/CAPQ-Pareto-V2/04_testbed_infra/teardown_all_gcp.sh"], check=False)

    assert len(trial_rows) == 360 and len(query_rows) == 1440
    out_csv = REPO_ROOT / "06_experiments_results/raw_logs/gcp_multiregion_360_real_wan.csv"
    q_csv = REPO_ROOT / "06_experiments_results/raw_logs/gcp_multiregion_1440_queries.csv"
    for path_p, data_p in [(out_csv, trial_rows), (q_csv, query_rows)]:
        with path_p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data_p[0].keys()))
            w.writeheader()
            w.writerows(data_p)

    schedulers = ["Classical_TLS1.3", "Static_MLKEM1024", "Rule_Heuristic", "CAPQ_Pareto"]
    baselines = ["Classical_TLS1.3", "Static_MLKEM1024", "Rule_Heuristic"]
    capq_j = np.array([float(r["j_sla_score"]) for r in trial_rows if r["scheduler"] == "CAPQ_Pareto"])
    raw_ps = []
    for b in baselines:
        bj = np.array([float(r["j_sla_score"]) for r in trial_rows if r["scheduler"] == b])
        _, pv = msc.paired_wilcoxon_safe(capq_j, bj)
        raw_ps.append(pv)
    holm_ps = msc.holm_bonferroni(raw_ps)

    csv_hash = sm.sha256_digest(out_csv)
    tex_rows = []
    print(f"\n=== STEP C4 COMPLETE: REAL 3-REGION GCP WAN SUMMARY (Elapsed: {time.perf_counter() - t_all:.1f}s) ===")
    for reg in GCP_REGIONS:
        rid = reg["id"]
        short_r = reg["label"].split(" ")[0]
        m_used = next(r["machine_type"] for r in trial_rows if r["region"] == rid)
        for s in schedulers:
            sub_t = [r for r in trial_rows if r["region"] == rid and r["scheduler"] == s]
            sub_q = [float(q["e2e_ms"]) for q in query_rows if q["region"] == rid and q["scheduler"] == s]
            m_lat = float(np.mean(sub_q))
            p95_lat = float(np.percentile(sub_q, 95))
            p99_lat = float(np.percentile(sub_q, 99))
            m_eng = float(np.mean([float(r["edge_energy_mj"]) for r in sub_t]))
            m_cst = float(np.mean([float(r["gcp_cost_usd_1m"]) for r in sub_t]))
            m_acc = float(np.mean([float(r["effective_accuracy"]) for r in sub_t]))
            m_sla = float(np.mean([float(r["sla_compliance_rate"]) for r in sub_t])) * 100.0
            m_jsla = float(np.mean([float(r["j_sla_score"]) for r in sub_t]))
            print(
                f"{short_r:<10} ({m_used}) | {s:<17} | Mean={m_lat:6.1f}ms | p95={p95_lat:6.1f}ms | p99={p99_lat:6.1f}ms "
                f"| E={m_eng:6.2f}mJ | Acc={m_acc:.3f} | SLA={m_sla:5.1f}%"
            )
            safe_s = s.replace("_", r"\_")
            lbl_s = f"\\textbf{{{safe_s}}}" if s == "CAPQ_Pareto" else f"\\texttt{{{safe_s}}}"
            tex_rows.append(
                f"{short_r} (\\texttt{{{m_used}}}) & {lbl_s} & ${m_lat:.1f}$ / ${p95_lat:.1f}$ / $\\mathbf{{{p99_lat:.1f}}}$ & "
                f"${m_eng:.2f}$ & ${m_cst:.3f}$ & ${m_acc:.3f}$ & ${m_sla:.1f}\\%$ & ${m_jsla:.3f}$ \\\\"
            )
        tex_rows.append(r"\midrule")

    tex_path = REPO_ROOT / "00_manuscript/tables/tab_gcp_multiregion_wan.tex"
    tex_lines = [
        r"% Auto-generated strictly from 1,440 live internet queries (360 paired trials) across 3 GCP Regions",
        f"% Source CSV SHA-256: {csv_hash}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Live Multi-Region Google Cloud WAN Audit From Physical Edge Gateway (Thailand) to Bangkok (\texttt{asia-southeast3}), Singapore (\texttt{asia-southeast1}), and Iowa (\texttt{us-central1}) ($N=30$ Paired Trials/Region/Scheduler)}",
        r"\label{tab:gcp_multiregion_wan}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"\textbf{GCP Region (VM)} & \textbf{Orchestrator} & \textbf{Mean / p95 / p99 Lat. [ms]} & \textbf{Edge RAPL [mJ/q]} & \textbf{GCP [\$/1M q]} & \textbf{Eff. Acc.} & \textbf{SLA Pass} & \textbf{$\mathcal{J}_{\text{SLA}}\downarrow$} \\",
        r"\midrule",
        *tex_rows[:-1],
        r"\midrule",
        r"\multicolumn{8}{l}{\footnotesize Overall Paired Wilcoxon $p_{\text{Holm}}$ vs. Classical / Static-PQC / Heuristic: $" + f"{holm_ps[0]:.2e}" + r"$ / $" + f"{holm_ps[1]:.2e}" + r"$ / $" + f"{holm_ps[2]:.2e}" + r"$. Raw CSV SHA-256: \texttt{" + csv_hash + r"}} \\",
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ]
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8", newline="\n")

    fig8_p = REPO_ROOT / "00_manuscript/figures/pareto_4d_plots/fig8_gcp_multiregion_p99_wan.pdf"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 3.1))
    colors = {"Classical_TLS1.3": "#7f8c8d", "Static_MLKEM1024": "#c0392b", "Rule_Heuristic": "#e67e22", "CAPQ_Pareto": "#1e8449"}
    r_ids = [r["id"] for r in GCP_REGIONS]
    r_lbls = ["Bangkok\n(asia-southeast3)", "Singapore\n(asia-southeast1)", "US-Central\n(us-central1)"]
    xr = np.arange(len(r_ids))
    bw_b = 0.19

    for idx, s in enumerate(schedulers):
        off = (idx - 1.5) * bw_b
        p99s = [float(np.percentile([float(q["e2e_ms"]) for q in query_rows if q["region"] == rid and q["scheduler"] == s], 99)) for rid in r_ids]
        slas = [float(np.mean([float(t["sla_compliance_rate"]) * 100.0 for t in trial_rows if t["region"] == rid and t["scheduler"] == s])) for rid in r_ids]
        ax1.bar(xr + off, p99s, bw_b, color=colors[s], label=s.replace("_", "-"))
        ax2.bar(xr + off, slas, bw_b, color=colors[s], label=s.replace("_", "-"))

    ax1.set_xticks(xr)
    ax1.set_xticklabels(r_lbls)
    ax1.set_ylabel("p99 Tail Latency (ms) [Log Scale]")
    ax1.set_yscale("log")
    ax1.set_title("(a) Live GCP Multi-Region p99 Tail Latency")
    ax1.grid(True, which="both", axis="y", linestyle=":", alpha=0.5)

    ax2.set_xticks(xr)
    ax2.set_xticklabels(r_lbls)
    ax2.set_ylim(0, 108)
    ax2.set_ylabel("Joint Regional SLA Pass Rate (%)")
    ax2.set_title("(b) Live Multi-Region SLA Compliance")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax2.legend(loc="lower left", fontsize=7.0, frameon=True)
    fig.tight_layout()
    fig.savefig(fig8_p, bbox_inches="tight")
    plt.close(fig)

    manifest_path = REPO_ROOT / "06_experiments_results/sha256_manifest.json"
    rec = sm.register_verified_table(
        manifest_path=manifest_path,
        raw_csv_path=out_csv,
        latex_table_path=tex_path,
        measured_rows=len(trial_rows),
        experiment_id="gcp_multiregion_360_real_wan",
    )
    print(f"\n[SHA-256] gcp_multiregion_360_real_wan.csv     : {rec['raw_csv_sha256']}")
    print(f"[SHA-256] gcp_multiregion_1440_queries.csv     : {sm.sha256_digest(q_csv)}")
    print(f"[SHA-256] tab_gcp_multiregion_wan.tex          : {rec['latex_table_sha256']}")
    print(f"[SHA-256] fig8_gcp_multiregion_p99_wan.pdf     : {sm.sha256_digest(fig8_p)}")


if __name__ == "__main__":
    main()