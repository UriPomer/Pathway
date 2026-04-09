"""Model download utilities. All HF env vars are read automatically by huggingface_hub."""
import logging
import os
import time

# Load .env BEFORE importing huggingface_hub so HF_ENDPOINT etc. take effect
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

from huggingface_hub import snapshot_download

_MAX_RETRIES = 3
_RETRY_BACKOFF = 5

WAN_TASK_REPOS = {"i2v-A14B": "Wan-AI/Wan2.2-I2V-A14B"}
WAN_SENTINEL_FILES = {"i2v-A14B": "models_t5_umt5-xxl-enc-bf16.pth"}
CSD_REPO_ID = "tomg-group-umd/CSD-ViT-L"

# Files in HF repos that should be skipped (e.g. accidentally committed files with 403)
_IGNORE_PATTERNS = ["nohup.out", "*.out"]


def _retry_download(fn, max_retries=_MAX_RETRIES):
    for attempt in range(1, max_retries + 1):
        try:
            return fn(attempt)
        except Exception as e:
            if attempt < max_retries:
                wait = attempt * _RETRY_BACKOFF
                print(f"  [Download] Failed ({type(e).__name__}: {e}), retrying in {wait}s ...")
                time.sleep(wait)
            else:
                raise


def _download_repo(repo_id: str, local_dir: str = None, max_retries: int = _MAX_RETRIES) -> str:
    def _fn(attempt):
        print(f"  [Download] {repo_id} ... (attempt {attempt}/{max_retries})")
        path = snapshot_download(
            repo_id,
            local_dir=local_dir,
            max_workers=1,
            ignore_patterns=_IGNORE_PATTERNS,
        )
        print(f"  [Download] Ready: {path}")
        return path
    return _retry_download(_fn, max_retries)


def ensure_wan_checkpoint(task: str = "i2v-A14B", ckpt_dir: str = None) -> str:
    """Check if wan checkpoint exists, download if not. Return ckpt_dir."""
    assert task in WAN_TASK_REPOS, f"Unknown wan task: {task}"
    if ckpt_dir is None:
        ckpt_dir = os.environ.get("WAN2_CKPT_DIR", "").strip()
    if not ckpt_dir:
        raise ValueError("WAN2_CKPT_DIR is not set. Please configure it in .env")

    # Check if already downloaded
    sentinel = WAN_SENTINEL_FILES.get(task, "models_t5_umt5-xxl-enc-bf16.pth")
    if os.path.isfile(os.path.join(ckpt_dir, sentinel)):
        logging.info(f"[model_utils] Wan checkpoint OK: {ckpt_dir}")
        return ckpt_dir
    low_noise_dir = os.path.join(ckpt_dir, "low_noise_model")
    if os.path.isdir(low_noise_dir) and os.listdir(low_noise_dir):
        logging.info(f"[model_utils] Wan checkpoint OK (subfolders): {ckpt_dir}")
        return ckpt_dir

    return _download_repo(WAN_TASK_REPOS[task], local_dir=ckpt_dir)


def ensure_csd_model() -> str:
    """Ensure CSD model is downloaded. Return local path."""
    return _download_repo(CSD_REPO_ID)


def ensure_all_models(task: str = "i2v-A14B", ckpt_dir: str = None):
    print("=" * 60 + f"\n  Model Download — task={task}\n" + "=" * 60)
    print("\n[1/2] Wan2.2 I2V Checkpoint")
    ensure_wan_checkpoint(task, ckpt_dir)
    print("\n[2/2] CSD Style Model")
    ensure_csd_model()
    print("\n" + "=" * 60 + "\n  All models ready!\n" + "=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    import argparse
    parser = argparse.ArgumentParser(description="Download required models")
    parser.add_argument("--task", default="i2v-A14B", choices=list(WAN_TASK_REPOS.keys()))
    parser.add_argument("--ckpt_dir", default=None)
    ensure_all_models(**vars(parser.parse_args()))
