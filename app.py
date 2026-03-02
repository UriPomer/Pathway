import os
import re
import subprocess
import sys
import tempfile
import math
from datetime import datetime

import cv2
import gradio as gr
import numpy as np
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
WAN2_DIR = os.path.join(BASE, "wan")
OUTPUT_DIR = os.path.join(BASE, "wan", "outputs")
if WAN2_DIR not in sys.path:
    sys.path.insert(0, WAN2_DIR)
from generate import generate, make_i2v_args  # type: ignore[import-untyped]
from vlm import IFEditPromptEnhancer
I2V_SIZES = ("720*1280", "1280*720", "480*832", "832*480")
IDLE_GPU_MB = 500
DEFAULT_CKPT = "/mnt/data3/zyx/models/Wan2.2-I2V-A14B"
DEFAULT_IMAGE = os.path.join(BASE, "wan", "examples", "i2v_input.JPG")
DEFAULT_PROMPT = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
SCPR_STILL_PROMPT = (
    "A perfectly still, high-resolution photograph with exceptional clarity "
    "and fine detail. The image remains completely static with no motion, "
    "camera shake, or blur. Ultra-sharp focus throughout the entire frame."
)
OFFICIAL_GUIDE_SCALE = 3.5

def auto_sample_shift_by_size(size_name: str) -> float:
    return 3.0 if size_name in ("480*832", "832*480") else 5.0

_PROMPT_ENHANCER = IFEditPromptEnhancer(strict=True)


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
    seed,
    offload_model,
    t5_cpu,
    use_cot,
    use_tld,
    tld_step_k,
    tld_threshold_ratio,
    use_scpr,
    scpr_refinement_ratio,
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
    input_prompt = prompt.strip()
    if use_cot:
        try:
            cot_prompt = _PROMPT_ENHANCER.enhance(image.convert("RGB"), input_prompt)
        except Exception as exc:
            raise gr.Error(f"CoT 生成失败: {exc}") from exc
        if not cot_prompt.strip():
            raise gr.Error("CoT 生成失败：返回空文本")
        input_prompt = cot_prompt.strip()

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
    auto_sample_shift = auto_sample_shift_by_size(size_name)
    sample_guide_scale = OFFICIAL_GUIDE_SCALE
    try:
        args = make_i2v_args(
            image=img_path,
            prompt=input_prompt,
            size=size_name,
            ckpt_dir=ckpt_dir,
            save_file=out_path,
            frame_num=int(frame_num),
            sample_solver=sample_solver,
            sample_steps=int(sample_steps),
            sample_shift=auto_sample_shift,
            sample_guide_scale=sample_guide_scale,
            base_seed=int(seed),
            offload_model=offload_model,
            t5_cpu=t5_cpu,
            ifedit_use_tld=use_tld,
            ifedit_tld_threshold_ratio=float(tld_threshold_ratio),
            ifedit_tld_step_k=int(tld_step_k),
        )
        generate(args)

        ref_input_image = None
        if use_scpr:
            ref_input_image, last_idx, total_frames = extract_last_frame(out_path)
            info_lines = [
                f"实际输入 prompt: {input_prompt}",
                f"CoT: {'ON' if use_cot else 'OFF'}",
                f"TLD: {'ON' if use_tld else 'OFF'} (K={int(tld_step_k)}, threshold={float(tld_threshold_ratio):.2f})",
                f"SCPR: ON",
                f"SCPR 输入: 主视频最后一帧 (第 {last_idx} 帧, 共 {total_frames} 帧)",
            ]
            ref_steps = max(1, math.ceil(int(sample_steps) * float(scpr_refinement_ratio)))
            ref_in_path = os.path.join(OUTPUT_DIR, f"scpr_input_{ts}.png")
            ref_out_path = os.path.join(OUTPUT_DIR, f"i2v_scpr_{ts}.mp4")
            ref_input_image.save(ref_in_path)
            ref_args = make_i2v_args(
                image=ref_in_path,
                prompt=SCPR_STILL_PROMPT,
                size=size_name,
                ckpt_dir=ckpt_dir,
                save_file=ref_out_path,
                frame_num=17,
                sample_solver=sample_solver,
                sample_steps=ref_steps,
                sample_shift=auto_sample_shift,
                sample_guide_scale=sample_guide_scale,
                base_seed=int(seed),
                offload_model=offload_model,
                t5_cpu=t5_cpu,
                ifedit_use_tld=False,
                ifedit_tld_threshold_ratio=float(tld_threshold_ratio),
                ifedit_tld_step_k=int(tld_step_k),
            )
            generate(ref_args)
            final_image, ref_best_idx, ref_best_score = extract_sharpest_frame(ref_out_path)
            info_lines.append(
                f"SCPR 精修步数: {ref_steps} | 输出最清晰帧: idx={ref_best_idx}, score={ref_best_score:.4f}"
            )
            try:
                os.unlink(ref_in_path)
            except OSError:
                pass
        else:
            final_image, main_best_idx, main_best_score = extract_sharpest_frame(out_path)
            info_lines = [
                f"实际输入 prompt: {input_prompt}",
                f"CoT: {'ON' if use_cot else 'OFF'}",
                f"TLD: {'ON' if use_tld else 'OFF'} (K={int(tld_step_k)}, threshold={float(tld_threshold_ratio):.2f})",
                f"SCPR: OFF",
                f"主视频最清晰帧: idx={main_best_idx}, score={main_best_score:.4f}",
            ]

        return out_path, final_image, input_prompt, "\n".join(info_lines), ref_input_image
    finally:
        try:
            os.unlink(img_path)
        except OSError:
            pass


def laplacian_sharpness(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(np.mean(np.abs(lap)))


def extract_sharpest_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error(f"无法读取视频: {video_path}")
    best_score = -1.0
    best_idx = 0
    best_frame = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        score = laplacian_sharpness(frame)
        if score > best_score:
            best_score = score
            best_idx = idx
            best_frame = frame
        idx += 1
    cap.release()
    if best_frame is None:
        raise gr.Error("视频中没有可用帧")
    final_image = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(final_image), best_idx, best_score


def extract_last_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error(f"无法读取视频: {video_path}")
    last_frame = None
    total = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        last_frame = frame
        total += 1
    cap.release()
    if last_frame is None:
        raise gr.Error("视频中没有可用帧")
    return Image.fromarray(cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)), total - 1, total


def build_ui():
    default_ckpt = os.environ.get("WAN2_CKPT_DIR", DEFAULT_CKPT)

    with gr.Blocks(title="Wan2.2 I2V") as app:
        gr.Markdown("# Wan2.2 Image-to-Video")

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
                    seed = gr.Number(label="随机种子 (-1=随机)", value=-1, precision=0)
                    offload_model = gr.Checkbox(label="offload_model (省显存)", value=True)
                    t5_cpu = gr.Checkbox(label="t5_cpu (省显存)", value=False)
                with gr.Accordion("IF-Edit", open=False):
                    use_cot = gr.Checkbox(label="CoT Prompt Enhancement", value=False)
                    use_tld = gr.Checkbox(label="Temporal Latent Dropout (TLD)", value=False)
                    tld_step_k = gr.Slider(label="TLD Step K", minimum=1, maximum=8, value=3, step=1)
                    tld_threshold_ratio = gr.Slider(label="TLD Threshold Ratio", minimum=0.0, maximum=1.0, value=0.9, step=0.05)
                    use_scpr = gr.Checkbox(label="Self-Consistent Post-Refinement (SCPR)", value=False)
                    scpr_refinement_ratio = gr.Slider(label="SCPR Refinement Ratio", minimum=0.1, maximum=1.0, value=0.6, step=0.05)

                btn = gr.Button("生成视频")

            with gr.Column():
                video_out = gr.Video(label="生成视频", autoplay=True)
                image_out = gr.Image(type="pil", label="最终编辑图")
                used_prompt = gr.Textbox(label="实际使用 Prompt", lines=4)
                run_info = gr.Textbox(label="运行信息", lines=8)
                scpr_input_out = gr.Image(type="pil", label="SCPR 输入帧（主视频最后一帧）", visible=True)

        btn.click(
            fn=run_i2v,
            inputs=[
                image, prompt, size_name, ckpt_dir,
                frame_num, sample_solver, sample_steps,
                seed, offload_model, t5_cpu,
                use_cot, use_tld, tld_step_k, tld_threshold_ratio,
                use_scpr, scpr_refinement_ratio,
            ],
            outputs=[video_out, image_out, used_prompt, run_info, scpr_input_out],
        )

    return app


if __name__ == "__main__":
    build_ui().launch()
