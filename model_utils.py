"""Model download utilities. All HF env vars are read automatically by huggingface_hub."""
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

# MUST load .env BEFORE importing huggingface_hub so HF_HOME/HF_ENDPOINT
# are visible when huggingface_hub caches HF_HUB_CACHE at import time.
_PROJECT_ENV = Path(__file__).resolve().parent / ".env"
load_dotenv(_PROJECT_ENV, override=False)

from huggingface_hub import snapshot_download, try_to_load_from_cache
from modelscope import snapshot_download as ms_snapshot_download

_MAX_RETRIES = 3
_RETRY_BACKOFF = 5

# Wan models are downloaded via ModelScope (much faster in China).
# CSD model is only on HuggingFace, so it uses HF snapshot_download.
WAN_TASK_REPOS = {
    "i2v-A14B": "Wan-AI/Wan2.2-I2V-A14B",
    "t2v-A14B": "Wan-AI/Wan2.2-T2V-A14B",
}
WAN_SENTINEL_FILES = {"i2v-A14B": "models_t5_umt5-xxl-enc-bf16.pth",
                       "t2v-A14B": "models_t5_umt5-xxl-enc-bf16.pth"}
CSD_REPO_ID = "tomg-group-umd/CSD-ViT-L"

# Wan 2.1 diffusers format models (for Frame Guidance official reproduction)
# These are on both HuggingFace and ModelScope — use ModelScope for speed in China.
WAN21_HF_REPOS = {
    "t2v-1.3B": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "i2v-14B":  "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
}
WAN21_MS_REPOS = {
    "t2v-1.3B": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "i2v-14B":  "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
}

# ModelScope cache dir (on external disk)
_MS_CACHE_DIR = os.environ.get(
    "MS_CACHE_DIR",
    os.path.join(os.environ.get("HF_HOME", "/root/autodl-tmp/huggingface"), "..", "ms_cache"),
)

# Files in HF repos that should be skipped (e.g. accidentally committed files with 403)
_IGNORE_PATTERNS = ["nohup.out", "*.out"]

# T2V only needs transformer weights; T5/VAE/tokenizer are shared with I2V.
# These are the shared files/dirs that can be symlinked from an I2V checkpoint.
_T2V_SHARED_ASSETS = [
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth",
    "google",
]
# Only download transformer subdirs for T2V (saves ~180GB).
_T2V_ALLOW_PATTERNS = [
    "high_noise_model/*",
    "low_noise_model/*",
    "configuration.json",
]


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


def _download_repo(repo_id: str, local_dir: str = None,
                    max_retries: int = _MAX_RETRIES,
                    allow_patterns: list = None) -> str:
    """Download a repo via ModelScope (fast in China)."""
    def _fn(attempt):
        print(f"  [Download/ModelScope] {repo_id} ... (attempt {attempt}/{max_retries})")
        kwargs = dict(
            model_id=repo_id,
            local_dir=local_dir,
            cache_dir=_MS_CACHE_DIR,
            ignore_file_pattern=_IGNORE_PATTERNS,
        )
        if allow_patterns:
            kwargs["allow_patterns"] = allow_patterns
        path = ms_snapshot_download(**kwargs)
        print(f"  [Download/ModelScope] Ready: {path}")
        return path
    return _retry_download(_fn, max_retries)


def _download_repo_hf(repo_id: str, local_dir: str = None,
                      max_retries: int = _MAX_RETRIES) -> str:
    """Download a repo via HuggingFace (for models not on ModelScope)."""
    def _fn(attempt):
        print(f"  [Download/HuggingFace] {repo_id} ... (attempt {attempt}/{max_retries})")
        kwargs = dict(
            repo_id=repo_id,
            local_dir=local_dir,
            max_workers=1,
            ignore_patterns=_IGNORE_PATTERNS,
        )
        path = snapshot_download(**kwargs)
        print(f"  [Download/HuggingFace] Ready: {path}")
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


def ensure_t2v_checkpoint(i2v_ckpt_dir: str, t2v_ckpt_dir: str = None) -> str:
    """Ensure T2V checkpoint is available. Downloads only transformer weights
    and symlinks T5/VAE/tokenizer from the I2V checkpoint to save disk space.

    Args:
        i2v_ckpt_dir: Path to existing I2V checkpoint (for symlinks).
        t2v_ckpt_dir: Path for T2V checkpoint. If None, uses ``<i2v_dir>/../Wan2.2-T2V-A14B``.

    Returns:
        Path to ready-to-use T2V checkpoint directory.
    """
    if t2v_ckpt_dir is None:
        t2v_ckpt_dir = os.path.join(os.path.dirname(i2v_ckpt_dir.rstrip("/")), "Wan2.2-T2V-A14B")

    # Check if already ready (transformer dirs + sentinel symlink)
    low_dir = os.path.join(t2v_ckpt_dir, "low_noise_model")
    sentinel = os.path.join(t2v_ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth")
    if os.path.isdir(low_dir) and os.listdir(low_dir) and os.path.exists(sentinel):
        logging.info(f"[model_utils] T2V checkpoint OK: {t2v_ckpt_dir}")
        return t2v_ckpt_dir

    # Download only transformer weights
    print(f"\n  [T2V] Downloading transformer weights to {t2v_ckpt_dir}")
    print(f"  [T2V] T5/VAE/tokenizer will be symlinked from {i2v_ckpt_dir}")
    _download_repo(
        WAN_TASK_REPOS["t2v-A14B"],
        local_dir=t2v_ckpt_dir,
        allow_patterns=_T2V_ALLOW_PATTERNS,
    )

    # Create symlinks for shared assets (T5, VAE, tokenizer)
    for asset in _T2V_SHARED_ASSETS:
        src = os.path.join(i2v_ckpt_dir, asset)
        dst = os.path.join(t2v_ckpt_dir, asset)
        if os.path.exists(dst):
            continue
        if not os.path.exists(src):
            logging.warning(f"[model_utils] Shared asset not found in I2V dir: {src}")
            continue
        os.symlink(src, dst)
        print(f"  [T2V] Symlinked: {asset} -> {src}")

    logging.info(f"[model_utils] T2V checkpoint ready: {t2v_ckpt_dir}")
    return t2v_ckpt_dir


def ensure_wan21_model(task: str = "i2v-14B") -> str:
    """Ensure Wan 2.1 diffusers-format model is downloaded.

    Tries ModelScope first (fast in China), falls back to HuggingFace mirror.

    These are used by the Frame Guidance official notebooks (keyframe_wan.ipynb,
    others_wan.ipynb) and are separate from the Wan 2.2 native checkpoints.

    Args:
        task: One of "t2v-1.3B" or "i2v-14B".

    Returns:
        Path to the downloaded model directory.
    """
    assert task in WAN21_HF_REPOS, f"Unknown Wan 2.1 task: {task}, choices: {list(WAN21_HF_REPOS.keys())}"

    # Check HF cache first (in case previously downloaded via HF)
    repo_id_hf = WAN21_HF_REPOS[task]
    cached = try_to_load_from_cache(repo_id_hf, "config.json")
    if isinstance(cached, str):
        model_dir = os.path.dirname(cached)
        logging.info(f"[model_utils] Wan 2.1 {task} already cached (HF): {model_dir}")
        return model_dir

    # Download via ModelScope (much faster in China)
    repo_id_ms = WAN21_MS_REPOS[task]
    print(f"\n  [Wan2.1] Downloading {repo_id_ms} via ModelScope ...")
    return _download_repo(repo_id_ms)


def ensure_csd_model() -> str:
    """Ensure CSD model is downloaded (via HuggingFace, not on ModelScope). Return local path."""
    cached = try_to_load_from_cache(CSD_REPO_ID, "pytorch_model.bin")
    if isinstance(cached, str):
        return os.path.dirname(cached)
    return _download_repo_hf(CSD_REPO_ID)


def ensure_all_models(task: str = "i2v-A14B", ckpt_dir: str = None,
                      wan21: str = None):
    print("=" * 60 + f"\n  Model Download — task={task}\n" + "=" * 60)
    print("\n[1/4] Wan2.2 I2V Checkpoint")
    i2v_dir = ensure_wan_checkpoint("i2v-A14B", ckpt_dir)
    print("\n[2/4] Wan2.2 T2V Checkpoint (transformer only)")
    ensure_t2v_checkpoint(i2v_dir)
    print("\n[3/4] CSD Style Model")
    ensure_csd_model()
    if wan21:
        print(f"\n[4/4] Wan 2.1 {wan21} (diffusers, for FG official demos)")
        ensure_wan21_model(wan21)
    else:
        print("\n[4/4] Wan 2.1 (skipped, use --wan21 to download)")
    print("\n" + "=" * 60 + "\n  All models ready!\n" + "=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    import argparse
    parser = argparse.ArgumentParser(description="Download required models")
    parser.add_argument("--task", default="i2v-A14B", choices=list(WAN_TASK_REPOS.keys()))
    parser.add_argument("--ckpt_dir", default=None)
    parser.add_argument("--wan21", default=None, choices=list(WAN21_HF_REPOS.keys()),
                        help="Also download Wan 2.1 diffusers model for FG demos (t2v-1.3B or i2v-14B)")
    ensure_all_models(**vars(parser.parse_args()))
