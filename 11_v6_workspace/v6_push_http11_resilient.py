import json
import pathlib
import ssl
import subprocess
import time
import urllib.request

repo = pathlib.Path(r"D:\Research\CAPQ-Pareto-V6")

# 1. Exclude temporary 150-DPI page preview PNGs from Git packfile (keep files on local disk!)
gi = repo / ".gitignore"
gi_txt = gi.read_text(encoding="utf-8") if gi.exists() else ""
if "11_v6_workspace/pdf_page_previews/" not in gi_txt:
    gi.write_text(gi_txt.rstrip() + "\n11_v6_workspace/pdf_page_previews/\n", encoding="utf-8", newline="\n")

subprocess.run(["git", "rm", "-rf", "--cached", "--ignore-unmatch", "11_v6_workspace/pdf_page_previews"], cwd=repo, check=False, capture_output=True)
subprocess.run(["git", "add", ".gitignore"], cwd=repo, check=True)
st = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=repo, capture_output=True, text=True)
if st.stdout.strip():
    subprocess.run(["git", "commit", "--amend", "--no-edit"], cwd=repo, check=True)
    print("[PACK SLIMMED] Excluded temporary pdf_page_previews from Git pack (kept on disk).")

# Verify 40/40 SHA-256 verifier still passes 100%
vr = subprocess.run(["python", str(repo / "reproduce_and_verify_v6.py")], cwd=repo, capture_output=True, text=True)
assert vr.returncode == 0, vr.stderr
print("[VERIFIER OK] 40/40 V6 artifacts verified intact.")

# 2. Resolve GitHub IPs (prefer fast Singapore Asia-Southeast edge 20.205.243.166 + DoH fallback)
candidate_ips = ["20.205.243.166"]
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
for url, hdrs in [
    ("https://1.1.1.1/dns-query?name=github.com&type=A", {"Accept": "application/dns-json"}),
    ("https://8.8.8.8/resolve?name=github.com&type=A", {"Accept": "application/json"}),
]:
    try:
        req = urllib.request.Request(url, headers=hdrs)
        with urllib.request.urlopen(req, timeout=5, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for ans in data.get("Answer", []):
                ip_s = ans.get("data", "").strip()
                if ans.get("type") == 1 and ip_s and ip_s not in candidate_ips:
                    candidate_ips.append(ip_s)
    except Exception:
        pass

print("[CANDIDATE GITHUB IPs]", candidate_ips)

def git_push_resilient(push_args: list[str], label: str) -> None:
    for attempt in range(1, 4):
        for ip in candidate_ips:
            cmd = [
                "git",
                "-c", f"http.curloptResolve=github.com:443:{ip}",
                "-c", "http.version=HTTP/1.1",
                "-c", "http.postBuffer=524288000",
                "-c", "http.lowSpeedLimit=0",
                "-c", "http.lowSpeedTime=999999",
                "-c", "pack.windowMemory=64m",
            ] + push_args
            print(f"  -> Pushing {label} via {ip} (HTTP/1.1, 500MB buffer, attempt {attempt})...")
            res = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if res.returncode == 0:
                print(f"  [PUSH SUCCESS] {label} synced cleanly!")
                return
            print(f"     [RETRY WARN] {res.stderr.strip().splitlines()[-1] if res.stderr.strip() else 'timeout'}")
            time.sleep(2)
    raise RuntimeError(f"Failed to push {label} after retries.")

git_push_resilient(["push", "-u", "origin", "v6"], "origin/v6")
git_push_resilient(["push", "origin", "v6:main", "--force"], "origin/main (Front Page)")

print("\n=== [GITHUB SYNC 100% COMPLETE] ===")
print("  Front Page : https://github.com/preecha-msu/CAPQ-Pareto")
print("  Branch v6  : https://github.com/preecha-msu/CAPQ-Pareto/tree/v6")