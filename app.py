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
    DTYPE = None  # Auto-detect

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
    tld_threshold_ratio=0.4,
    tld_step_K=3,
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
    # Update TLD parameters dynamically
    pipeline.tld_step_K = int(tld_step_K)
    pipeline.tld_threshold_ratio = float(tld_threshold)

    return run_pipeline_dispatch(
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


# =============================================================
# 3. Build UI
# =============================================================

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
            image = gr.Image(type="pil", label="Source Image")
            prompt = gr.Textbox(
                label="Edit Instruction",
                value="Add sunglasses to the person",
                placeholder="Describe the edit you want to apply...",
            )

            gr.Markdown("### Generation Parameters")
            num_frames = gr.Slider(17, 121, value=33, step=4, label="Video Frames")
            height = gr.Slider(240, 720, value=288, step=16, label="Height")
            width = gr.Slider(320, 1280, value=512, step=16, label="Width")
            guidance_scale = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="Guidance Scale")
            num_inference_steps = gr.Slider(10, 100, value=20, step=5, label="Denoising Steps")
            seed = gr.Number(value=-1, label="Seed (-1 = random)")

            gr.Markdown("### IF-Edit Components")
            use_cot = gr.Checkbox(value=True, label="CoT Prompt Enhancement")
            use_tld = gr.Checkbox(value=True, label="Temporal Latent Dropout (TLD)")
            use_scpr = gr.Checkbox(value=True, label="Self-Consistent Post-Refinement (SCPR)")

            with gr.Accordion("TLD Parameters", open=False):
                tld_step_K = gr.Slider(2, 8, value=3, step=1,
                                       label="TLD Step K (sub-sampling stride)")
                tld_threshold = gr.Slider(0.1, 0.8, value=0.4, step=0.05,
                                          label="TLD Threshold Ratio (fraction of total steps)")

            run_btn = gr.Button("Run IF-Edit", variant="primary")

        # Right: Output
        with gr.Column(scale=1):
            gr.Markdown("### Output")
            result = gr.Image(type="pil", label="Edited Image")
            out_seed = gr.Number(label="Actual Seed Used")
            info = gr.Textbox(label="Run Information", lines=8)

    # Bind
    run_btn.click(
        fn=run_pipeline,
        inputs=[
            image, prompt, num_frames, height, width,
            guidance_scale, num_inference_steps, seed,
            use_cot, use_tld, use_scpr,
            tld_step_K, tld_threshold,
        ],
        outputs=[result, out_seed, info],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
