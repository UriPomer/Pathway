#!/usr/bin/env python3
"""
Car-turn keyframe-guided video generation using Frame Guidance.
Based on the official keyframe_wan.ipynb (turtle example), adapted for car-turn.

Usage:
    cd /home/zhangyixin/Pathway/frame-guidance
    python run_keyframe_demo.py                  # keyframe guidance with car-turn
    python run_keyframe_demo.py --no-guidance     # baseline without guidance
"""
import os
import sys
import argparse
import subprocess

# Load .env from project root (one level up from frame-guidance/)
from dotenv import load_dotenv
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_project_root, ".env"), override=True)


def select_free_gpu():
    """Auto-select the GPU with the most free memory."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        best_idx, best_free = 0, -1
        for line in result.stdout.strip().split("\n"):
            idx, free = line.split(",")
            idx, free = int(idx.strip()), int(free.strip())
            if free > best_free:
                best_idx, best_free = idx, free
        print(f"Auto-selected GPU {best_idx} ({best_free} MiB free)")
        return best_idx
    except Exception:
        return 0


gpu_id = select_free_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

import torch
import numpy as np
from PIL import Image
import decord

# Ensure the frame-guidance pipelines are importable
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, _project_root)

from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_gif, export_to_video
from transformers import CLIPVisionModel
from pipelines.pipeline_wan_i2v import WanImageToVideoPipeline


# ---------- helpers from official notebook ----------
def make_frames(path):
    frames = []
    vr = decord.VideoReader(path)
    for i in range(len(vr)):
        frames.append(Image.fromarray(vr[i].asnumpy()))
    return frames


def parse_line_to_list(line: str) -> list:
    """parse_line_to_list("1,3,4") -> [1, 3, 4]
       parse_line_to_list("1-3,4") -> [1, 2, 3, 4]"""
    result = []
    parts = line.split(",")
    for part in parts:
        part = part.strip()
        if "-" in part:
            start_str, end_str = part.split("-")
            result.extend(range(int(start_str), int(end_str) + 1))
        else:
            result.append(int(part))
    return result


# ---------- model loading ----------
def load_pipe():
    # Use model_utils to find (or download via ModelScope) the local path,
    # so from_pretrained doesn't re-download from slow HF mirror.
    from model_utils import ensure_wan21_model
    model_id = ensure_wan21_model("i2v-14B")
    print(f"Loading model: {model_id}")

    image_encoder = CLIPVisionModel.from_pretrained(
        model_id, subfolder="image_encoder", torch_dtype=torch.float32,
        ignore_mismatched_sizes=True,
    )
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.bfloat16
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16
    )
    pipe = pipe.to("cuda")
    pipe.transformer.enable_gradient_checkpointing()

    print("Model loaded.")
    return pipe


# ---------- car-turn keyframe generation ----------
def run_car_turn(pipe, no_guidance=False):
    """
    Car-turn keyframe guidance.
    3 keyframes: frame 0, 25, 48 -> mapped to 81-frame indices.
    Original keyframes are at 49-frame positions (0, 25, 48).
    For 81-frame generation, we scale: 0->0, 25->42, 48->80.
    """
    # Load keyframe images
    kf_0 = Image.open(os.path.join(script_dir, "examples", "car-turn", "0.png"))
    kf_25 = Image.open(os.path.join(script_dir, "examples", "car-turn", "25.png"))
    kf_48 = Image.open(os.path.join(script_dir, "examples", "car-turn", "48.png"))

    # Build video_init: use first frame as init image for I2V,
    # fill keyframes at scaled positions
    num_frames = 81
    video_init = [None] * num_frames

    # Map 49-frame keyframe positions to 81-frame positions
    # 0 -> 0, 25 -> 42 (≈25/48*80), 48 -> 80
    kf_map = {0: kf_0, 42: kf_25, 80: kf_48}

    # Fill: first half with kf_0, around middle with kf_25, last part with kf_48
    for i in range(num_frames):
        if i <= 21:
            video_init[i] = kf_0
        elif i <= 61:
            video_init[i] = kf_25
        else:
            video_init[i] = kf_48

    prompt = ("A red sports car makes a sharp turn on a winding mountain road, "
              "tires gripping the asphalt, surrounded by lush green hills and "
              "a clear blue sky. Dynamic camera angle captures the motion.")
    negative_prompt = "low quality"

    if no_guidance:
        guidance_lr = [0.0] * 50
        guidance_step = [0] * 50
        travel_time = (-1, -1)
        fixed_frames = []
        save_tag = "car_turn_no_fg"
        print("[CAR-TURN BASELINE] guidance OFF")
    else:
        # Following official turtle example parameters
        guidance_lr = [3e0] * 5 + [3e0] * 45
        guidance_step = [5] * 5 + [3] * 5 + [1] * 10 + [0] * 30
        travel_time = (15, 20)
        # Keyframe positions in 81-frame space (scaled from 49-frame originals)
        fixed_frames = parse_line_to_list("42,80")  # mid and last keyframes
        save_tag = "car_turn_fg"
        print(f"[CAR-TURN FG] fixed_frames={fixed_frames}, travel_time={travel_time}")

    print(f"[CAR-TURN] prompt: {prompt}")

    seed = 0
    video = pipe(
        prompt=prompt,
        video=video_init,
        negative_prompt=negative_prompt,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        guidance_scale=5.0,
        generator=torch.Generator(device="cuda").manual_seed(seed),
        fixed_frames=fixed_frames,
        guidance_step=guidance_step,
        guidance_lr=guidance_lr,
        loss_fn="frame",
        height=480,
        width=832,
        num_frames=num_frames,
        latent_downscale_factor=4,
    ).frames[0]

    save_path = os.path.join(script_dir, "results", f"{save_tag}.gif")
    return video, save_path


# ---------- save ----------
def save_output(video, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    output_ = (video * 255).astype(np.uint8)
    frames = [Image.fromarray(f) for f in output_]
    export_to_gif(frames, save_path, fps=16)
    print(f"Saved gif: {save_path}")
    try:
        mp4_path = save_path.replace(".gif", ".mp4")
        export_to_video(frames, mp4_path, fps=16)
        print(f"Saved mp4: {mp4_path}")
    except Exception:
        pass


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-guidance", action="store_true",
                        help="Disable frame guidance for baseline comparison")
    args = parser.parse_args()

    pipe = load_pipe()
    video, path = run_car_turn(pipe, no_guidance=args.no_guidance)
    save_output(video, path)
    print("Done.")


if __name__ == "__main__":
    main()
