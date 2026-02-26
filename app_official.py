import os
import subprocess
import sys
import tempfile

import gradio as gr

BASE = os.path.dirname(os.path.abspath(__file__))
GENERATE_PY = os.path.join(BASE, "Wan2.1", "generate.py")
I2V_SIZES = ("720*1280", "1280*720", "480*832", "832*480")


def run_i2v(
    image,
    prompt,
    size_name,
    ckpt_dir,
    frame_num,
    sample_solver,
    sample_steps,
    sample_shift,
    sample_guide_scale,
    seed,
    offload_model,
    t5_cpu,
):
    if image is None:
        raise gr.Error("请上传输入图片")
    if not (prompt or "").strip():
        raise gr.Error("请输入提示词")
    if not size_name or size_name not in I2V_SIZES:
        raise gr.Error("请选择分辨率")
    if not (ckpt_dir or "").strip():
        raise gr.Error("请填写模型目录 (ckpt_dir)")

    ckpt_dir = ckpt_dir.strip()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        image.convert("RGB").save(f.name)
        img_path = f.name
    fd, out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        cmd = [
            sys.executable,
            GENERATE_PY,
            "--task", "i2v-14B",
            "--image", img_path,
            "--prompt", prompt.strip(),
            "--size", size_name,
            "--ckpt_dir", ckpt_dir,
            "--save_file", out_path,
            "--frame_num", str(int(frame_num)),
            "--sample_solver", sample_solver,
            "--sample_steps", str(int(sample_steps)),
            "--sample_shift", str(float(sample_shift)),
            "--sample_guide_scale", str(float(sample_guide_scale)),
            "--base_seed", str(int(seed)),
            "--offload_model", "True" if offload_model else "False",
        ]
        if t5_cpu:
            cmd.append("--t5_cpu")
        subprocess.run(cmd, cwd=BASE, check=True)
        return out_path
    finally:
        try:
            os.unlink(img_path)
        except OSError:
            pass


def build_ui():
    default_ckpt = os.environ.get("WAN2_CKPT_DIR", "")

    with gr.Blocks(title="Wan2.1 I2V") as app:
        gr.Markdown("# Wan2.1 Image-to-Video")

        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="输入图片")
                prompt = gr.Textbox(label="提示词", placeholder="描述希望生成的视频内容", lines=3)
                size_name = gr.Dropdown(label="分辨率", choices=list(I2V_SIZES), value="832*480")
                ckpt_dir = gr.Textbox(
                    label="模型目录 (ckpt_dir)",
                    value=default_ckpt,
                    placeholder="/path/to/Wan2.1-I2V-14B-480P",
                )

                with gr.Accordion("高级参数", open=False):
                    frame_num = gr.Number(label="帧数 (4n+1)", value=81, precision=0, minimum=5, maximum=257)
                    sample_solver = gr.Dropdown(label="采样器", choices=["unipc", "dpm++"], value="unipc")
                    sample_steps = gr.Slider(label="采样步数", minimum=1, maximum=100, value=40, step=1)
                    sample_shift = gr.Slider(label="Shift", minimum=0.0, maximum=10.0, value=3.0, step=0.1)
                    sample_guide_scale = gr.Slider(label="Guide scale", minimum=0.0, maximum=20.0, value=5.0, step=0.1)
                    seed = gr.Number(label="随机种子 (-1=随机)", value=-1, precision=0)
                    offload_model = gr.Checkbox(label="offload_model (省显存)", value=True)
                    t5_cpu = gr.Checkbox(label="t5_cpu (省显存)", value=True)

                btn = gr.Button("生成视频")

            with gr.Column():
                video_out = gr.Video(label="生成视频", autoplay=True)

        btn.click(
            fn=run_i2v,
            inputs=[
                image, prompt, size_name, ckpt_dir,
                frame_num, sample_solver, sample_steps, sample_shift,
                sample_guide_scale, seed, offload_model, t5_cpu,
            ],
            outputs=video_out,
        )

    return app


if __name__ == "__main__":
    build_ui().launch()
