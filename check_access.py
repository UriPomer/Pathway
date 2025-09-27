import os
import sys
import requests
from typing import List

REPO = os.environ.get("MODEL_REPO", "zai-org/CogVideoX1.5-5B-I2V")
TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
BASE = "https://huggingface.co"  # API base

# Candidate large shard paths (common naming patterns observed in error log)
SHARD_PATHS: List[str] = [
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
    "text_encoder/model-00001-of-00004.safetensors",
    "text_encoder/model-00002-of-00004.safetensors",
    "text_encoder/model-00003-of-00004.safetensors",
    "text_encoder/model-00004-of-00004.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
]

if not TOKEN:
    print("[ERROR] No token found in env (HUGGINGFACE_HUB_TOKEN / HF_TOKEN). Exiting.")
    sys.exit(1)

headers = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/octet-stream"}
print(f"[INFO] Checking direct shard accessibility for repo: {REPO}")

ok = True
for rel in SHARD_PATHS:
    url = f"{BASE}/{REPO}/resolve/main/{rel}"
    try:
        r = requests.head(url, headers=headers, timeout=25, allow_redirects=True)
        status = r.status_code
        cl = r.headers.get("Content-Length", "?")
        loc = r.url
        if status == 200:
            print(f"[OK] {rel} -> 200 len={cl}")
        elif status in (302, 301):
            print(f"[REDIR] {rel} -> {status} final={loc}")
        elif status == 401:
            print(f"[FAIL-401] {rel} (Unauthorized) final={loc}")
            ok = False
        elif status == 403:
            print(f"[FAIL-403] {rel} (Forbidden) final={loc}")
            ok = False
        else:
            print(f"[WARN] {rel} -> {status} final={loc}")
    except Exception as e:
        print(f"[ERR] HEAD {rel} : {e}")
        ok = False

if ok:
    print("[SUMMARY] All probed shards accessible (no 401/403). Partial download likely due to network interruptions (503) or CAS fallback issues.")
    print("          Consider disabling accelerated transfer: export HF_HUB_ENABLE_HF_TRANSFER=0 and retry.")
else:
    print("[SUMMARY] Some shards inaccessible. Re-check model access approval or token scope.")
    print("          Try regenerating a classic read token and export HUGGINGFACE_HUB_TOKEN=hf_xxx")
