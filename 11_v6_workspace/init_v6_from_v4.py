import hashlib, pathlib, shutil

root = pathlib.Path(r"D:\Research\CAPQ-Pareto-V6")
res_dir = pathlib.Path(r"D:\Research")

# คัดลอกสคริปต์สร้างรูปและตารางของ V4 มาตั้งต้นใน 11_v6_workspace และเปลี่ยนเป้าหมายเป็น V5 เท่านั้น
v4_audit = root / "09_v4_reviewer_rebuttal_audit"
v5_work  = root / "11_v6_workspace"
v5_work.mkdir(parents=True, exist_ok=True)

if v4_audit.exists():
    for py_file in v4_audit.glob("*.py"):
        txt = py_file.read_text(encoding="utf-8", errors="ignore")
        txt_v5 = (
            txt.replace(r"D:\Research\CAPQ-Pareto-V4", r"D:\Research\CAPQ-Pareto-V6")
               .replace(r"D:\Research\CAPQ-Pareto-V3", r"D:\Research\CAPQ-Pareto-V6")
               .replace("08_manuscript_ieee_tcc_v4", "08_manuscript_ieee_tcc_v6")
               .replace("-IEEE-TCC-V4.", "-IEEE-TCC-V6.")
               .replace("-Preview-V4.", "-Preview-V6.")
               .replace("-Reviewers-V4.", "-Reviewers-V6.")
               .replace("-Manifest-V4.", "-Manifest-V6.")
               .replace("SHA256SUMS_V4.txt", "SHA256SUMS_V6.txt")
        )
        out_name = py_file.name.replace("_v4", "_v5")
        (v5_work / out_name).write_text(txt_v5, encoding="utf-8")

# โคลนไฟล์ส่งมอบ V4 ที่สมบูรณ์มาเป็น Baseline เริ่มต้นของ *-V5.* ใน D:\Research\ (Read-only บน V4)
mapping = [
    ("CAPQ-Pareto-Main-IEEE-TCC-V4.pdf",         "CAPQ-Pareto-Main-IEEE-TCC-V6.pdf"),
    ("CAPQ-Pareto-Supplement-IEEE-TCC-V4.pdf",   "CAPQ-Pareto-Supplement-IEEE-TCC-V6.pdf"),
    ("CAPQ-Pareto-Response-to-Reviewers-V4.pdf", "CAPQ-Pareto-Response-to-Reviewers-V6.pdf"),
    ("CAPQ-Pareto-Cover-Letter-IEEE-TCC-V4.pdf", "CAPQ-Pareto-Cover-Letter-IEEE-TCC-V6.pdf"),
    ("CAPQ-Pareto-Fig1-Preview-V4.png",          "CAPQ-Pareto-Fig1-Preview-V6.png"),
    ("CAPQ-Pareto-Overleaf-IEEE-TCC-V4.zip",     "CAPQ-Pareto-Overleaf-IEEE-TCC-V6.zip"),
]

manifest_lines = ["# CAPQ-Pareto V5 Initialized Baseline SHA-256 Manifest (Cloned from Verified V4)"]
for src_name, dst_name in mapping:
    src_p = res_dir / src_name
    dst_p = res_dir / dst_name
    if src_p.exists():
        shutil.copy2(src_p, dst_p)
        h = hashlib.sha256(dst_p.read_bytes()).hexdigest()
        manifest_lines.append(f"{h}  {dst_name}")
        print(f"[V5 BASELINE READY] {dst_name:<44} | SHA256={h}")

manifest_p = res_dir / "CAPQ-Pareto-SHA256-Manifest-V6.txt"
manifest_p.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
mh = hashlib.sha256(manifest_p.read_bytes()).hexdigest()
print(f"[V5 BASELINE READY] {'CAPQ-Pareto-SHA256-Manifest-V6.txt':<44} | SHA256={mh}")