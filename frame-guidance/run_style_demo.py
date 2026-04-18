#!/usr/bin/env python3
"""
Reproduce the official Frame Guidance demos from the notebooks.
Uses Wan 2.1 (diffusers) — exactly matching the notebooks.

Usage:
    cd /root/Pathway/frame-guidance
    python run_style_demo.py style       # style transfer (T2V 1.3B)
    python run_style_demo.py loop        # loop video (T2V 1.3B)
    python run_style_demo.py keyframe    # keyframe-guided (I2V 14B)
    python run_style_demo.py             # defaults to style
"""
import os
import sys
import argparse

# Load .env from project root (one level up from frame-guidance/)
from dotenv import load_dotenv
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_project_root, ".env"), override=True)

import torch
import numpy as np
from PIL import Image

# Ensure the frame-guidance pipelines are importable
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, _project_root)

from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_gif


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_t2v_pipe():
    """Load Wan 2.1 T2V 1.3B pipeline (for style / loop demos)."""
    from pipelines.pipeline_wan import WanPipeline
    from model_utils import ensure_wan21_model

    model_id = ensure_wan21_model("t2v-1.3B")
    print(f"Loading T2V model: {model_id}")
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    pipe.transformer.enable_gradient_checkpointing()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    print("T2V model loaded.")
    return pipe


def load_i2v_pipe():
    """Load Wan 2.1 I2V 14B pipeline (for keyframe demo)."""
    from pipelines.pipeline_wan_i2v import WanImageToVideoPipeline
    from transformers import CLIPVisionModel
    from model_utils import ensure_wan21_model

    model_id = ensure_wan21_model("i2v-14B")
    print(f"Loading I2V model: {model_id}")
    image_encoder = CLIPVisionModel.from_pretrained(
        model_id, subfolder="image_encoder", torch_dtype=torch.float32)
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.bfloat16)
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id, vae=vae, image_encoder=image_encoder,
        torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    pipe.transformer.enable_gradient_checkpointing()
    print("I2V model loaded.")
    return pipe


# ---------------------------------------------------------------------------
# Demo runners
# ---------------------------------------------------------------------------

def run_style(pipe):
    """Exactly matches others_wan.ipynb cell 3 (Stylized video generation)."""
    guidance_lr = [3e0] * 50
    guidance_step = [0]*2 + [5]*8 + [3]*20 + [1]*10 + [0]*10

    prompt = ("A New York City street scene with a man and a woman walking "
              "down the street, a dog running after them, and a bicyclist "
              "passing by, water color painting style.")
    travel_time = (-1, -1)
    fixed_frames = [20, 40, 60, 80]

    style_image = Image.open(os.path.join(script_dir, "examples", "style", "11.jpg"))
    save_path = os.path.join(script_dir, "results", "wan_style_repro.gif")

    print(f"[STYLE] prompt: {prompt}")
    print(f"[STYLE] fixed_frames={fixed_frames} lr={guidance_lr[0]} travel_time={travel_time}")

    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=81,
        guidance_scale=5,
        generator=torch.Generator(device="cuda").manual_seed(42),
        fixed_frames=fixed_frames,
        guidance_step=guidance_step,
        guidance_lr=guidance_lr,
        loss_fn="style",
        additional_inputs={"style_image": style_image},
        travel_time=travel_time,
        latent_downscale_factor=4,
    ).frames[0]

    return video, save_path


def run_loop(pipe):
    """Exactly matches others_wan.ipynb cell 5 (Loop video generation)."""
    guidance_lr = [3e0] * 50
    guidance_step = [7]*3 + [3]*7 + [1]*10 + [0]*30

    prompt = ("A golden retriever digging a hole in the sand on a beach, "
              "head disappears into the hole, then pops up again from the "
              "same spot, ocean waves in the background, natural daylight.")
    travel_time = (15, 20)

    save_path = os.path.join(script_dir, "results", "wan_loop_repro.gif")

    print(f"[LOOP] prompt: {prompt}")
    print(f"[LOOP] travel_time={travel_time}")

    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=81,
        guidance_scale=5,
        generator=torch.Generator(device="cuda").manual_seed(42),
        guidance_step=guidance_step,
        guidance_lr=guidance_lr,
        loss_fn="loop",
        travel_time=travel_time,
        latent_downscale_factor=4,
    ).frames[0]

    return video, save_path


def run_keyframe(pipe):
    """Exactly matches keyframe_wan.ipynb cell 4 (Keyframe-guided generation)."""
    # Load reference frames
    init = Image.open(os.path.join(script_dir, "examples", "turtle_wan", "0.png"))
    last = Image.open(os.path.join(script_dir, "examples", "turtle_wan", "48.png"))

    video_init = [None] * 81
    video_init[:41] = [init] * 41
    video_init[41:] = [last] * 40

    seed = 0
    prompt = ("A mature sea turtle glides through clear blue waters above a "
              "coral reef, its flippers moving gracefully. Sunlight filters "
              "through, casting a tranquil glow on the turtle and its serene "
              "surroundings.")
    negative_prompt = "low quality"

    guidance_lr = [3e0] * 5 + [3e0] * 45
    guidance_step = [5]*5 + [3]*5 + [1]*10 + [0]*30

    travel_time = (15, 20)
    fixed_frames = [40, 80]

    save_path = os.path.join(script_dir, "results", "wan_keyframe_repro.gif")

    print(f"[KEYFRAME] prompt: {prompt}")
    print(f"[KEYFRAME] fixed_frames={fixed_frames} travel_time={travel_time}")

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
        num_frames=81,
        latent_downscale_factor=4,
    ).frames[0]

    return video, save_path


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_output(video, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    output_ = (video * 255).astype(np.uint8)
    frames = [Image.fromarray(f) for f in output_]
    export_to_gif(frames, save_path, fps=16)
    print(f"Saved gif: {save_path}")
    try:
        from diffusers.utils import export_to_video
        mp4_path = save_path.replace(".gif", ".mp4")
        export_to_video(frames, mp4_path, fps=16)
        print(f"Saved mp4: {mp4_path}")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", default="style",
                        choices=["style", "loop", "keyframe"],
                        help="Which demo to run (default: style)")
    args = parser.parse_args()

    if args.mode in ("style", "loop"):
        pipe = load_t2v_pipe()
    else:
        pipe = load_i2v_pipe()

    if args.mode == "style":
        video, path = run_style(pipe)
    elif args.mode == "loop":
        video, path = run_loop(pipe)
    else:
        video, path = run_keyframe(pipe)

    save_output(video, path)
    print("Done.")


if __name__ == "__main__":
    main()
