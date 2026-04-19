#!/usr/bin/env python3
"""
Gradio UI for verifying sketch-guided last-frame control.

Uses Wan 2.1 I2V 14B + official pipeline_wan_i2v.py with loss_fn="scribble".
Draw white strokes on the input image to define the last-frame target shape.

Usage:
    cd /root/Pathway/frame-guidance
    python run_sketch_verify.py
    # Then open http://localhost:7860 (or the Gradio public URL)
"""
import os
import sys
import subprocess

# Load .env from project root
from dotenv import load_dotenv
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_project_root, ".env"), override=True)


def select_free_gpu():
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
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import numpy as np
from PIL import Image
import gradio as gr

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, _project_root)

from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from transformers import CLIPVisionModel
from pipelines.pipeline_wan_i2v import WanImageToVideoPipeline

# Default input image
DEFAULT_IMAGE = os.path.join(_project_root, "example", "flame.jpg")

# ---------------------------------------------------------------------------
# Model loading (lazy, once)
# ---------------------------------------------------------------------------
_PIPE = None


def get_pipe():
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    from model_utils import ensure_wan21_model
    model_id = ensure_wan21_model("i2v-14B")
    print(f"Loading model: {model_id}")

    image_encoder = CLIPVisionModel.from_pretrained(
        model_id, subfolder="image_encoder", torch_dtype=torch.float32,
        ignore_mismatched_sizes=True,
    )
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.bfloat16)
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id, vae=vae, image_encoder=image_encoder,
        torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    pipe.transformer.enable_gradient_checkpointing()
    print("Model loaded.")
    _PIPE = pipe
    return pipe


# ---------------------------------------------------------------------------
# Sketch extraction from ImageEditor
# ---------------------------------------------------------------------------

def extract_sketch(editor_value, base_image):
    """Extract white brush strokes from ImageEditor as sketch on black bg.

    Returns an RGB PIL Image: white strokes on black background.
    """
    if editor_value is None or base_image is None:
        return None

    # gr.ImageEditor returns dict: {"composite": ..., "background": ..., "layers": ...}
    if isinstance(editor_value, dict):
        composite = editor_value.get("composite")
    else:
        composite = editor_value

    if composite is None:
        return None
    if isinstance(composite, np.ndarray):
        comp_pil = Image.fromarray(composite.astype(np.uint8)).convert("RGB")
    elif isinstance(composite, Image.Image):
        comp_pil = composite.convert("RGB")
    else:
        return None

    # Compare composite with base to find strokes
    base_rgb = base_image.convert("RGB").resize(comp_pil.size, Image.BILINEAR)
    comp_np = np.array(comp_pil, dtype=np.float32)
    base_np = np.array(base_rgb, dtype=np.float32)

    # Pixel diff → stroke mask
    diff = np.abs(comp_np - base_np).sum(axis=2)
    stroke_mask = diff > 30  # threshold

    if not np.any(stroke_mask):
        return None

    # White strokes on black background
    sketch = np.zeros_like(comp_np, dtype=np.uint8)
    sketch[stroke_mask] = 255
    return Image.fromarray(sketch)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(
    base_image,
    editor_value,
    prompt,
    guidance_lr,
    num_steps,
    seed,
    enable_guidance,
):
    if base_image is None:
        raise gr.Error("请上传输入图片")
    if not (prompt or "").strip():
        raise gr.Error("请输入 prompt")

    pipe = get_pipe()

    num_frames = 81
    target_w, target_h = 832, 480
    init_image = base_image.convert("RGB").resize((target_w, target_h), Image.BILINEAR)

    # Extract sketch from editor
    sketch_image = None
    if enable_guidance:
        sketch_image = extract_sketch(editor_value, base_image)
        if sketch_image is None:
            raise gr.Error("请在图片上绘制草图（白色笔刷）")
        sketch_image = sketch_image.resize((target_w, target_h), Image.BILINEAR)

    # Build video_init list (required by I2V pipeline)
    # All frames = init image (sketch is passed via additional_inputs, not video)
    video_init = [init_image] * num_frames

    seed_val = int(seed)
    if seed_val < 0:
        seed_val = torch.randint(0, 2**31, (1,)).item()

    additional_inputs = {}
    if enable_guidance:
        lr_val = float(guidance_lr)
        guidance_lr_list = [lr_val] * num_steps
        # Aggressive schedule: guide through step 40 (not just 20)
        guidance_step = [5]*5 + [3]*10 + [1]*20 + [0] * max(0, num_steps - 35)
        # Pad or trim to match num_steps
        guidance_step = (guidance_step + [0] * num_steps)[:num_steps]
        fixed_frames = [num_frames - 1]  # last frame
        travel_time = (-1, -1)
        loss_fn = "sketch"
        additional_inputs["sketch_image"] = sketch_image
    else:
        guidance_lr_list = [0.0] * num_steps
        guidance_step = [0] * num_steps
        fixed_frames = []
        travel_time = (-1, -1)
        loss_fn = "sketch"

    print(f"\n{'='*60}")
    print(f"  Sketch Verify — guidance={'ON' if enable_guidance else 'OFF'}")
    print(f"  prompt: {prompt}")
    print(f"  lr={guidance_lr}, steps={num_steps}, seed={seed_val}")
    print(f"  fixed_frames={fixed_frames}")
    print(f"{'='*60}\n")

    video = pipe(
        prompt=prompt,
        video=video_init,
        negative_prompt="low quality, blurry, camera motion, camera shake, pan, zoom",
        num_videos_per_prompt=1,
        num_inference_steps=num_steps,
        guidance_scale=5.0,
        generator=torch.Generator(device="cuda").manual_seed(seed_val),
        fixed_frames=fixed_frames,
        guidance_step=guidance_step,
        guidance_lr=guidance_lr_list,
        loss_fn=loss_fn,
        additional_inputs=additional_inputs if additional_inputs else None,
        height=480,
        width=832,
        num_frames=num_frames,
        latent_downscale_factor=4,
    ).frames[0]

    # Save video
    os.makedirs(os.path.join(script_dir, "results"), exist_ok=True)
    tag = "sketch_fg" if enable_guidance else "sketch_baseline"
    mp4_path = os.path.join(script_dir, "results", f"{tag}_{seed_val}.mp4")
    output_ = (video * 255).astype(np.uint8)
    frames = [Image.fromarray(f) for f in output_]
    export_to_video(frames, mp4_path, fps=16)
    print(f"Saved: {mp4_path}")

    # Extract last frame for comparison
    last_frame = frames[-1]

    # Also save sketch for reference
    if sketch_image is not None:
        sketch_path = os.path.join(script_dir, "results", f"{tag}_{seed_val}_sketch.png")
        sketch_image.save(sketch_path)

    return mp4_path, last_frame, sketch_image, f"seed={seed_val}, saved: {mp4_path}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    default_img = DEFAULT_IMAGE if os.path.isfile(DEFAULT_IMAGE) else None

    with gr.Blocks(title="Sketch Guidance Verify") as demo:
        gr.Markdown("# Sketch Guidance Verification (Wan 2.1 I2V + Official FG)")
        gr.Markdown(
            "在输入图上用白色笔刷绘制最后一帧的目标轮廓，点击 Generate 验证引导效果。\n\n"
            "**流程**: 输入图 → 在图上画草图 → 生成视频 → 最后一帧应匹配草图形状"
        )

        with gr.Row():
            with gr.Column():
                base_image = gr.Image(
                    type="pil", label="输入图片",
                    value=default_img,
                )
                sketch_editor = gr.ImageEditor(
                    type="pil",
                    label="在此绘制最后一帧草图（白色笔刷）",
                    value=default_img,
                    brush=gr.Brush(
                        colors=["#ffffff"],
                        color_mode="fixed",
                        default_size=8,
                    ),
                )
                prompt = gr.Textbox(
                    label="Prompt",
                    value="The flames flicker, static camera, fixed viewpoint",
                    lines=2,
                )

                with gr.Row():
                    guidance_lr = gr.Slider(
                        label="Guidance LR", minimum=0.1, maximum=100.0,
                        value=10.0, step=0.1,
                    )
                    num_steps = gr.Slider(
                        label="Steps", minimum=10, maximum=50,
                        value=50, step=1,
                    )
                    seed = gr.Number(label="Seed (-1=random)", value=42, precision=0)

                enable_guidance = gr.Checkbox(label="启用 Sketch Guidance", value=True)
                btn = gr.Button("Generate", variant="primary")

            with gr.Column():
                video_out = gr.Video(label="生成视频", autoplay=True)
                with gr.Row():
                    last_frame_out = gr.Image(type="pil", label="最后一帧")
                    sketch_out = gr.Image(type="pil", label="提取的草图")
                info_out = gr.Textbox(label="信息")

        # Sync base image to editor
        base_image.change(
            fn=lambda img: img,
            inputs=[base_image],
            outputs=[sketch_editor],
        )

        btn.click(
            fn=generate,
            inputs=[
                base_image, sketch_editor, prompt,
                guidance_lr, num_steps, seed, enable_guidance,
            ],
            outputs=[video_out, last_frame_out, sketch_out, info_out],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
