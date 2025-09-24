import gradio as gr
from pipeline import Frame2FramePipeline
import random
import torch

"""Simple Gradio demo wiring the minimal Frame2FramePipeline with optional iterative path mode."""

pipeline = Frame2FramePipeline()

def run_pipeline(image, prompt, num_frames, guidance_scale, seed, use_iterative, iterative_steps, candidates_per_step, w_sem, w_step, w_id):
    if seed is None or seed < 0:
        seed = random.randint(0, 2**31 - 1)
    try:
        generator = torch.Generator(device=pipeline.device).manual_seed(int(seed))
    except Exception:
        generator = None
    if use_iterative:
        best_frame, path, metrics = pipeline.run(
            image=image,
            prompt_text=prompt,
            guidance_scale=float(guidance_scale),
            num_frames=int(num_frames),  # unused in iterative, kept for API
            generator=generator,
            use_iterative=True,
            iterative_steps=int(iterative_steps),
            candidates_per_step=int(candidates_per_step),
            w_sem=float(w_sem),
            w_step=float(w_step),
            w_id=float(w_id)
        )
        if metrics:
            info = (
                f"Iterative path length={len(path)} | path_length={metrics.get('path_length',0):.3f} "
                f"smoothness={metrics.get('smoothness',0):.4f}"
            )
        else:
            info = f"Iterative path length={len(path)}"
    else:
        best_frame, frames, _ = pipeline.run(
            image=image,
            prompt_text=prompt,
            num_frames=int(num_frames),
            guidance_scale=float(guidance_scale),
            generator=generator,
            use_iterative=False
        )
        info = f"Baseline frames={len(frames)}"
    return best_frame, seed, info

with gr.Blocks(title="Frame2Frame Baseline + Iterative") as demo:
    gr.Markdown("# Frame2Frame Baseline + Iterative Path Demo")
    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(type="pil", label="输入图像")
            prompt = gr.Textbox(label="编辑文本", value="make it a watercolor painting")
            num_frames = gr.Slider(4, 64, value=16, step=1, label="(Baseline) 生成帧数")
            guidance_scale = gr.Slider(1, 12, value=6, step=0.5, label="Guidance Scale")
            seed = gr.Number(value=-1, precision=0, label="随机种子(-1自动)")
            use_iterative = gr.Checkbox(value=False, label="启用迭代路径模式")
            iterative_steps = gr.Slider(2, 12, value=5, step=1, label="迭代步数")
            candidates_per_step = gr.Slider(2, 8, value=3, step=1, label="每步候选帧数")
            w_sem = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="w_sem 语义")
            w_step = gr.Slider(0.0, 1.0, value=0.2, step=0.01, label="w_step 步长")
            w_id = gr.Slider(0.0, 1.0, value=0.1, step=0.01, label="w_id 身份")
            run_btn = gr.Button("运行")
        with gr.Column(scale=1):
            result = gr.Image(type="pil", label="结果帧")
            out_seed = gr.Number(label="实际种子")
            info = gr.Textbox(label="信息", interactive=False)
    run_btn.click(
        fn=run_pipeline,
        inputs=[image, prompt, num_frames, guidance_scale, seed, use_iterative, iterative_steps, candidates_per_step, w_sem, w_step, w_id],
        outputs=[result, out_seed, info]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
