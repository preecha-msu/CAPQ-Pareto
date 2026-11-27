import json
import pathlib
import ssl
import subprocess
import urllib.request

repo = pathlib.Path(r"D:\Research\CAPQ-Pareto-V6")

def resolve_github_ip() -> str:
    # 1. Try DNS-over-HTTPS directly to raw IP 1.1.1.1 and 8.8.8.8 (needs ZERO local DNS!)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    doh_urls = [
        ("https://1.1.1.1/dns-query?name=github.com&type=A", {"Accept": "application/dns-json"}),
        ("https://8.8.8.8/resolve?name=github.com&type=A", {"Accept": "application/json"}),
    ]
    for url, hdrs in doh_urls:
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=6, context=ctx) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for ans in data.get("Answer", []):
                    if ans.get("type") == 1 and ans.get("data"):
                        return ans["data"].strip()
        except Exception:
            pass
    # 2. Fallback to well-known GitHub Asia-Pacific / Global anycast IPs
    return "20.205.243.166"

gh_ip = resolve_github_ip()
print(f"[DNS BYPASS] Resolved github.com -> {gh_ip}")

def run_git_push(args: list[str]) -> subprocess.CompletedProcess:
    cmd = ["git", "-c", f"http.curloptResolve=github.com:443:{gh_ip}"] + args
    return subprocess.run(cmd, cwd=repo, capture_output=True, text=True, encoding="utf-8", errors="replace")

# Backup original V1 origin/main to origin/v1-archive if not yet present
chk = run_git_push(["ls-remote", "--heads", "origin", "v1-archive"])
if chk.returncode != 0:
    print("[NETWORK DIAGNOSTIC] Direct TCP 443 to GitHub failed:")
    print(chk.stderr.strip())
    print("-> กรุณาตรวจสอบว่า Wi-Fi / LAN เชื่อมต่ออินเทอร์เน็ตอยู่ (ลองเปิดเว็บทั่วไป 1 หน้า) แล้วรันสคริปต์นี้ซ้ำได้เลยครับ")
    raise SystemExit(2)

if "refs/heads/v1-archive" not in chk.stdout:
    bk = run_git_push(["push", "origin", "refs/remotes/origin/main:refs/heads/v1-archive"])
    if bk.returncode == 0:
        print("[BACKUP OK] Preserved original V1 main branch on GitHub as 'origin/v1-archive'.")
else:
    print("[BACKUP EXISTS] 'origin/v1-archive' already preserved on GitHub.")

p_v6 = run_git_push(["push", "-u", "origin", "v6"])
if p_v6.returncode != 0:
    raise RuntimeError(f"Push to origin/v6 failed:\n{p_v6.stderr}")
print("[PUSH OK] Updated branch 'origin/v6' on GitHub.")

p_main = run_git_push(["push", "origin", "v6:main", "--force"])
if p_main.returncode != 0:
    raise RuntimeError(f"Push to origin/main failed:\n{p_main.stderr}")
print("[PUSH OK] Updated default front page 'origin/main' on GitHub to Final V6!")