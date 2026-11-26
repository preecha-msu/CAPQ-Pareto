import hashlib
import pathlib
import shutil

root = pathlib.Path(r"D:\Research\CAPQ-Pareto-V6")
res_dir = pathlib.Path(r"D:\Research")

# คัดลอกสคริปต์ทำงานจาก 10_v5_workspace มายัง 11_v6_workspace พร้อมผูกพาธเป็น V6 ทั้งหมด
v5_work = root / "10_v5_workspace"
v6_work = root / "11_v6_workspace"
v6_work.mkdir(parents=True, exist_ok=True)

if v5_work.exists():
    for py_file in v5_work.glob("*.py"):
        txt = py_file.read_text(encoding="utf-8", errors="ignore")
        txt_v6 = (
            txt.replace(r"D:\Research\CAPQ-Pareto-V5", r"D:\Research\CAPQ-Pareto-V6")
               .replace(r"D:\\Research\\CAPQ-Pareto-V5", r"D:\\Research\\CAPQ-Pareto-V6")
               .replace("/mnt/d/Research/CAPQ-Pareto-V5", "/mnt/d/Research/CAPQ-Pareto-V6")
               .replace("08_manuscript_ieee_tcc_v5", "08_manuscript_ieee_tcc_v6")
               .replace("10_v5_workspace", "11_v6_workspace")
               .replace("v5_raw_logs", "v6_raw_logs")
               .replace("-IEEE-TCC-V5.", "-IEEE-TCC-V6.")
               .replace("-Preview-V5.", "-Preview-V6.")
               .replace("-Reviewers-V5.", "-Reviewers-V6.")
               .replace("-Manifest-V5.", "-Manifest-V6.")
               .replace("SHA256SUMS_V5.txt", "SHA256SUMS_V6.txt")
        )
        out_name = py_file.name.replace("_v5", "_v6").replace("v5_", "v6_")
        (v6_work / out_name).write_text(txt_v6, encoding="utf-8", newline="\n")

# คัดลอกไฟล์ SHA256SUMS ภายในโปรเจกต์มาตั้งต้นเป็นของ V6
v5_sums = root / "SHA256SUMS_V5.txt"
v6_sums = root / "SHA256SUMS_V6.txt"
if v5_sums.exists() and not v6_sums.exists():
    shutil.copy2(v5_sums, v6_sums)

# โคลนไฟล์ส่งมอบ V5 ที่สมบูรณ์มาเป็น Baseline เริ่มต้นของ *-V6.* ใน D:\Research\ (Read-Only บน V5)
mapping = [
    ("CAPQ-Pareto-Main-IEEE-TCC-V5.pdf",         "CAPQ-Pareto-Main-IEEE-TCC-V6.pdf"),
    ("CAPQ-Pareto-Supplement-IEEE-TCC-V5.pdf",   "CAPQ-Pareto-Supplement-IEEE-TCC-V6.pdf"),
    ("CAPQ-Pareto-Response-to-Reviewers-V5.pdf", "CAPQ-Pareto-Response-to-Reviewers-V6.pdf"),
    ("CAPQ-Pareto-Cover-Letter-IEEE-TCC-V5.pdf", "CAPQ-Pareto-Cover-Letter-IEEE-TCC-V6.pdf"),
    ("CAPQ-Pareto-Fig1-Preview-V5.png",          "CAPQ-Pareto-Fig1-Preview-V6.png"),
    ("CAPQ-Pareto-Overleaf-IEEE-TCC-V5.zip",     "CAPQ-Pareto-Overleaf-IEEE-TCC-V6.zip"),
]

manifest_lines = ["# CAPQ-Pareto V6 Initialized Baseline SHA-256 Manifest (Cloned from Verified V5)"]
for src_name, dst_name in mapping:
    src_p = res_dir / src_name
    dst_p = res_dir / dst_name
    if src_p.exists():
        shutil.copy2(src_p, dst_p)
        h = hashlib.sha256(dst_p.read_bytes()).hexdigest()
        manifest_lines.append(f"{h}  {dst_name}")
        print(f"[V6 BASELINE READY] {dst_name:<44} | SHA256={h}")

manifest_p = res_dir / "CAPQ-Pareto-SHA256-Manifest-V6.txt"
manifest_p.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8", newline="\n")
mh = hashlib.sha256(manifest_p.read_bytes()).hexdigest()
print(f"[V6 BASELINE READY] {'CAPQ-Pareto-SHA256-Manifest-V6.txt':<44} | SHA256={mh}")