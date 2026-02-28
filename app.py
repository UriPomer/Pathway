"""IF-Edit Demo App — Zero-Shot Image Editing via I2V Models.

Based on "Are Image-to-Video Models Good Zero-Shot Image Editors?"
(arXiv: 2511.19435)

Implements IF-Edit framework:
- CoT Prompt Enhancement
- Temporal Latent Dropout (TLD)
- Self-Consistent Post-Refinement (SCPR)
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from pipeline import IFEditPipeline
from vlm import IFEditPromptEnhancer
from utils import run_pipeline_dispatch

# =============================================================
# 0. Model Configuration
# =============================================================

# Default model: Wan 2.1 I2V 14B (480P version for memory efficiency)
# Can be overridden via IFEDIT_MODEL_PATH env var
MODEL_PATH = os.environ.get(
    "IFEDIT_MODEL_PATH",
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    # Wan-AI/Wan2.2-I2V-A14B-Diffusers
)

# Model cache directory
MODEL_CACHE_DIR = os.environ.get(
    "IFEDIT_MODEL_CACHE",
    str(Path(__file__).resolve().parent / "models"),
)

# Force FP32 mode
if os.environ.get("IFEDIT_FORCE_FP32") == "1":
    import torch
    DTYPE = torch.float32
else:
    DTYPE = None  # Auto-detect (bfloat16 on CUDA)

# =============================================================
# 1. Initialize Pipeline
# =============================================================

prompt_enhancer = IFEditPromptEnhancer()

pipeline = IFEditPipeline(
    model_path=MODEL_PATH,
    model_cache_dir=MODEL_CACHE_DIR,
    torch_dtype=DTYPE,
    prompt_enhancer=prompt_enhancer,
    tld_enabled=True,
    tld_threshold_ratio=0.5,
    tld_step_K=2,
    scpr_enabled=True,
    scpr_refinement_ratio=0.6,
)


# =============================================================
# 2. Pipeline Runner
# =============================================================

def run_pipeline(
    image,
    prompt,
    num_frames,
    height,
    width,
    guidance_scale,
    num_inference_steps,
    seed,
    use_cot,
    use_tld,
    use_scpr,
    tld_step_K,
    tld_threshold,
):
    pipeline.tld_step_K = int(tld_step_K)
    pipeline.tld_threshold_ratio = float(tld_threshold)

    final_image, actual_seed, info, run_info = run_pipeline_dispatch(
        pipeline=pipeline,
        image=image,
        prompt=prompt,
        num_frames=int(num_frames),
        height=int(height),
        width=int(width),
        guidance_scale=float(guidance_scale),
        num_inference_steps=int(num_inference_steps),
        seed=seed,
        use_cot=use_cot,
        use_tld=use_tld,
        use_scpr=use_scpr,
    )

    debug_image = run_info.get("debug_input_image")
    debug_prompt = run_info.get("debug_prompt") or run_info.get("temporal_prompt") or ""
    h, w = run_info.get("debug_height"), run_info.get("debug_width")
    resolution = f"{h} × {w}" if (h is not None and w is not None) else ""
    neg_preview = run_info.get("debug_negative_prompt") or ""
    if len(neg_preview) > 100:
        neg_preview = neg_preview[:100] + "..."
    debug_params_lines = [
        f"分辨率: {resolution}" if resolution else "",
        f"帧数: {run_info.get('debug_num_frames', '')}",
        f"推理步数: {run_info.get('debug_num_inference_steps', '')}",
        f"guidance_scale: {run_info.get('debug_guidance_scale', '')}",
        f"TLD: {'ON' if run_info.get('tld_enabled') else 'OFF'} (K={run_info.get('tld_step_K')}, threshold={run_info.get('tld_threshold_ratio')})",
        f"CoT: {'ON' if run_info.get('cot_enabled') else 'OFF'}",
        f"负向 prompt 预览: {neg_preview}" if neg_preview else "",
    ]
    debug_params = "\n".join(s for s in debug_params_lines if s)

    return final_image, actual_seed, info, debug_image, debug_prompt, debug_params


# =============================================================
# 3. Build UI
# =============================================================

_DEFAULT_PROMPT = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
_DEFAULT_IMAGE_PATH = Path(__file__).resolve().parent / "Wan2.1" / "examples" / "i2v_input.JPG"
_DEFAULT_IMAGE = str(_DEFAULT_IMAGE_PATH) if _DEFAULT_IMAGE_PATH.exists() else None

with gr.Blocks(title="IF-Edit: Zero-Shot Image Editor") as demo:
    gr.Markdown("""# IF-Edit: Zero-Shot Image Editing via I2V Models

**"Are Image-to-Video Models Good Zero-Shot Image Editors?"** (arXiv: 2511.19435)

This demo implements the IF-Edit framework with three core components:
- **CoT Prompt Enhancement**: Converts edit instructions into temporal descriptions
- **Temporal Latent Dropout (TLD)**: Speeds up inference by dropping redundant frame latents
- **Self-Consistent Post-Refinement (SCPR)**: Improves output clarity via Laplacian-based selection
""")

    with gr.Row():
        # Left: Inputs
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            image = gr.Image(type="pil", label="Source Image", value=_DEFAULT_IMAGE)
            prompt = gr.Textbox(
                label="Edit Instruction",
                value=_DEFAULT_PROMPT,
                placeholder="Describe the edit you want to apply...",
            )

            gr.Markdown("### Generation Parameters")
            num_frames = gr.Slider(5, 257, value=81, step=4, label="Video Frames")
            height = gr.Slider(0, 720, value=0, step=16, label="Height (0=auto)")
            width = gr.Slider(0, 1280, value=0, step=16, label="Width (0=auto)")
            guidance_scale = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="Guidance Scale")
            num_inference_steps = gr.Slider(10, 100, value=40, step=5, label="Denoising Steps")
            seed = gr.Number(value=-1, label="Seed (-1 = random)")

            gr.Markdown("### IF-Edit Components")
            use_cot = gr.Checkbox(value=False, label="CoT Prompt Enhancement")
            use_tld = gr.Checkbox(value=False, label="Temporal Latent Dropout (TLD)")
            use_scpr = gr.Checkbox(value=False, label="Self-Consistent Post-Refinement (SCPR)")

            with gr.Accordion("TLD Parameters", open=False):
                tld_step_K = gr.Slider(2, 8, value=2, step=1,
                                       label="TLD Step K (sub-sampling stride)")
                tld_threshold = gr.Slider(0.1, 0.8, value=0.5, step=0.05,
                                          label="TLD Threshold Ratio (fraction of total steps)")

            run_btn = gr.Button("Run IF-Edit", variant="primary")

        # Right: Output
        with gr.Column(scale=1):
            gr.Markdown("### Output")
            result = gr.Image(type="pil", label="Edited Image")
            out_seed = gr.Number(label="Actual Seed Used")
            info = gr.Textbox(label="Run Information", lines=6)

            gr.Markdown("### Debug: 输入给视频模型的信息")
            with gr.Accordion("展开查看预处理与模型输入", open=True):
                debug_image = gr.Image(type="pil", label="送入视频模型的图片 (resize 后)")
                debug_prompt = gr.Textbox(label="送入视频模型的 Prompt", lines=5)
                debug_params = gr.Textbox(label="分辨率与参数", lines=8)

    run_btn.click(
        fn=run_pipeline,
        inputs=[
            image, prompt, num_frames, height, width,
            guidance_scale, num_inference_steps, seed,
            use_cot, use_tld, use_scpr,
            tld_step_K, tld_threshold,
        ],
        outputs=[result, out_seed, info, debug_image, debug_prompt, debug_params],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
