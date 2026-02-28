import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime

import gradio as gr

BASE = os.path.dirname(os.path.abspath(__file__))
WAN2_DIR = os.path.join(BASE, "Wan2.1")
OUTPUT_DIR = os.path.join(BASE, "Wan2.1", "outputs")
if WAN2_DIR not in sys.path:
    sys.path.insert(0, WAN2_DIR)
from generate import generate, make_i2v_args  # type: ignore[import-untyped]
I2V_SIZES = ("720*1280", "1280*720", "480*832", "832*480")
IDLE_GPU_MB = 500
DEFAULT_CKPT = "/mnt/data3/zyx/models/Wan2.1-I2V-14B-480P"
DEFAULT_IMAGE = os.path.join(BASE, "Wan2.1", "examples", "i2v_input.JPG")
DEFAULT_PROMPT = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."


def get_least_used_gpu(threshold_mb=IDLE_GPU_MB):
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if out.returncode != 0:
            return ""
        cand = []
        for line in out.stdout.strip().splitlines():
            m = re.match(r"\s*(\d+)\s*,\s*(\d+)", line)
            if m:
                idx, used_mb = int(m.group(1)), int(m.group(2))
                if used_mb < threshold_mb:
                    cand.append((idx, used_mb))
        if not cand:
            return ""
        cand.sort(key=lambda x: x[1])
        return str(cand[0][0])
    except Exception:
        return ""


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
    dev = get_least_used_gpu()
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if dev:
        os.environ["CUDA_VISIBLE_DEVICES"] = dev

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        image.convert("RGB").save(f.name)
        img_path = f.name
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"i2v_{ts}.mp4")
    try:
        args = make_i2v_args(
            image=img_path,
            prompt=prompt.strip(),
            size=size_name,
            ckpt_dir=ckpt_dir,
            save_file=out_path,
            frame_num=int(frame_num),
            sample_solver=sample_solver,
            sample_steps=int(sample_steps),
            sample_shift=float(sample_shift),
            sample_guide_scale=float(sample_guide_scale),
            base_seed=int(seed),
            offload_model=offload_model,
            t5_cpu=t5_cpu,
        )
        generate(args)
        return out_path
    finally:
        try:
            os.unlink(img_path)
        except OSError:
            pass


def build_ui():
    default_ckpt = os.environ.get("WAN2_CKPT_DIR", DEFAULT_CKPT)

    with gr.Blocks(title="Wan2.1 I2V") as app:
        gr.Markdown("# Wan2.1 Image-to-Video")

        with gr.Row():
            with gr.Column():
                default_img = DEFAULT_IMAGE if os.path.isfile(DEFAULT_IMAGE) else None
                image = gr.Image(type="pil", label="输入图片", value=default_img)
                prompt = gr.Textbox(label="提示词", value=DEFAULT_PROMPT, lines=3)
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
                    t5_cpu = gr.Checkbox(label="t5_cpu (省显存)", value=False)

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
