#!/usr/bin/env bash
set -e
export DEBIAN_FRONTEND=noninteractive

mkdir -p /opt/capq_cloud /opt/capq_bridge
cloud-init status --wait >/dev/null 2>&1 || true
apt-get -o DPkg::Lock::Timeout=120 update -qq
apt-get -o DPkg::Lock::Timeout=120 install -y -qq \
    build-essential cmake ninja-build git libssl-dev libgomp1 \
    postgresql postgresql-contrib postgresql-server-dev-all \
    python3-dev python3-pip python3-numpy python3-cryptography python3-psycopg2

if python3 -m pip install --help | grep -q "break-system-packages"; then
    python3 -m pip install -q --break-system-packages aioquic
else
    python3 -m pip install -q aioquic
fi

if [ -f /tmp/openfhe_x86_cache.tar.gz ]; then
    tar -xzf /tmp/openfhe_x86_cache.tar.gz -C /
    ldconfig
fi

cd /tmp
rm -rf pgvector
git clone --depth 1 --branch v0.8.0 https://github.com/pgvector/pgvector.git
make -C pgvector with_llvm=no -j"$(nproc)" >/dev/null
make -C pgvector with_llvm=no install >/dev/null

systemctl enable --now postgresql || pg_ctlcluster "$(ls /etc/postgresql | head -n 1)" main start
su - postgres -c "psql -tc \"SELECT 1 FROM pg_roles WHERE rolname='capq_user'\" | grep -q 1 || psql -c \"CREATE USER capq_user WITH SUPERUSER PASSWORD 'capq_real_pass';\""
su - postgres -c "psql -tc \"SELECT 1 FROM pg_database WHERE datname='capq_db'\" | grep -q 1 || psql -c \"CREATE DATABASE capq_db OWNER capq_user;\""

python3 - << 'PYINGEST'
import numpy as np, psycopg2
from psycopg2.extras import execute_values

conn = psycopg2.connect(dbname="capq_db", user="capq_user", password="capq_real_pass", host="127.0.0.1", port=5432)
conn.autocommit = True
cur = conn.cursor()
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
cur.execute("DROP TABLE IF EXISTS capq_tpcc_tx; CREATE TABLE capq_tpcc_tx (tx_id INT PRIMARY KEY, w_id INT, amount DOUBLE PRECISION, tx_type TEXT);")
rng = np.random.default_rng(42)
tpcc_rows = [(i, int(i % 10) + 1, float(rng.uniform(10.0, 5000.0)), "NEW_ORDER" if i % 2 == 0 else "PAYMENT") for i in range(1, 1001)]
execute_values(cur, "INSERT INTO capq_tpcc_tx (tx_id, w_id, amount, tx_type) VALUES %s", tpcc_rows)

cur.execute("DROP TABLE IF EXISTS capq_sift_128d; CREATE TABLE capq_sift_128d (id INT PRIMARY KEY, emb vector(128));")
d = np.load("/opt/capq_cloud/sift_128d_real_5k_200q.npz")
base = d["base"]
vec_rows = [(int(i), "[" + ",".join(f"{x:.6f}" for x in base[i]) + "]") for i in range(base.shape[0])]
execute_values(cur, "INSERT INTO capq_sift_128d (id, emb) VALUES %s", vec_rows, page_size=500)
cur.execute("CREATE INDEX idx_cloud_sift_hnsw ON capq_sift_128d USING hnsw (emb vector_ip_ops) WITH (m = 16, ef_construction = 64);")
conn.close()
print("[CLOUD-DB] Ingested 1,000 TPC-C rows + 5,000 SIFT-128d HNSW vectors.")
PYINGEST

if ! python3 -c "import ctypes; ctypes.CDLL('/opt/capq_bridge/libcapq_openfhe_bridge.so')" >/dev/null 2>&1; then
    mkdir -p /tmp/capq_fhe
    cd /tmp/capq_fhe
    rm -rf openfhe-development
    git clone --depth 1 --recurse-submodules --shallow-submodules --branch v1.2.3 https://github.com/openfheorg/openfhe-development.git
    cmake -S openfhe-development -B openfhe-development/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF \
        -DNATIVE_SIZE=64 -DWITH_NATIVEOPT=OFF -DCMAKE_CXX_FLAGS="-O3 -march=x86-64-v2" -DCMAKE_INSTALL_LIBDIR=lib >/dev/null
    ninja -C openfhe-development/build -j"$(nproc)" install >/dev/null
    ldconfig

    INC_OPENFHE=$(find /usr/local/include/openfhe -maxdepth 1 -type d | sed 's/^/-I/')
    g++ -std=c++17 -O3 -march=x86-64-v2 -fopenmp -fPIC -shared -include cstdint -Wno-error \
        /opt/capq_bridge/capq_openfhe_bridge.cpp \
        $INC_OPENFHE \
        -I/usr/local/include/openfhe/core -I/usr/local/include/openfhe/pke \
        -I/usr/local/include/openfhe/binfhe -I/usr/local/include/openfhe/cereal \
        -L/usr/local/lib -lOPENFHEpke -lOPENFHEcore -Wl,-rpath,/usr/local/lib \
        -o /opt/capq_bridge/libcapq_openfhe_bridge.so

    tar -czf /tmp/openfhe_built_bundle.tar.gz /usr/local/lib/libOPENFHE* /opt/capq_bridge/libcapq_openfhe_bridge.so
fi

pkill -f cloud_region_daemon.py || true
nohup env OMP_WAIT_POLICY=PASSIVE python3 -u /opt/capq_cloud/cloud_region_daemon.py > /var/log/capq_daemon.log 2>&1 < /dev/null &

for i in $(seq 1 30); do
    if grep -q "CLOUD_DAEMON_ONLINE" /var/log/capq_daemon.log 2>/dev/null; then
        echo "[CLOUD-READY] Daemon listening on TCP:8443 and UDP:443/20444"
        exit 0
    fi
    sleep 1
done
cat /var/log/capq_daemon.log || true
exit 1