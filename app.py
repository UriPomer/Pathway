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
from ui_panels import IFEditPanel, MobiusPanel, get_output_panel_updates_by_mode
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
from wan.configs.wan_i2v_A14B import i2v_A14B
WAN_TLD_THRESHOLD_RATIO = float(getattr(i2v_A14B, "boundary", 0.9))

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
    loopless_enable,
    loop_shift_skip,
    loop_shift_stop_step,
):
    if image is None:
        raise gr.Error("请上传输入图片")
    if not (prompt or "").strip():
        raise gr.Error("请输入提示词")
    if not size_name or size_name not in I2V_SIZES:
        raise gr.Error("请选择分辨率")
    if not (ckpt_dir or "").strip():
        raise gr.Error("请填写模型目录 (ckpt_dir)")
    if bool(loopless_enable) and (bool(use_tld) or bool(use_cot) or bool(use_scpr)):
        raise gr.Error(
            "IF-Edit (CoT/TLD/SCPR) 与 Loopless Cinemagraph 互斥：请二选一。"
        )

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

    mode_subdir = "mobius" if loopless_enable else "ifedit"
    mode_output_dir = os.path.join(OUTPUT_DIR, mode_subdir)
    os.makedirs(mode_output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(mode_output_dir, f"i2v_{ts}.mp4")
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
            loopless_enable=bool(loopless_enable),
            loop_shift_skip=int(loop_shift_skip),
            loop_shift_stop_step=int(loop_shift_stop_step),
        )
        generate(args)

        ref_input_image = None
        if use_scpr:
            ref_input_image, main_pick_idx, total_frames, main_pick_score = select_frame(
                out_path, method="sharpest", start_ratio=1 / 3
            )
            info_lines = IFEditPanel.format_run_info_with_scpr(
                input_prompt,
                use_cot,
                use_tld,
                int(tld_step_k),
                float(tld_threshold_ratio),
                main_pick_idx,
                total_frames,
                main_pick_score,
            )
            ref_steps = max(1, math.ceil(int(sample_steps) * float(scpr_refinement_ratio)))
            ref_in_path = os.path.join(mode_output_dir, f"scpr_input_{ts}.png")
            ref_out_path = os.path.join(mode_output_dir, f"i2v_scpr_{ts}.mp4")
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
                loopless_enable=bool(loopless_enable),
                loop_shift_skip=int(loop_shift_skip),
                loop_shift_stop_step=int(loop_shift_stop_step),
            )
            generate(ref_args)
            final_image, ref_best_idx, _, ref_best_score = select_frame(
                ref_out_path, method="sharpest", start_ratio=1 / 3
            )
            info_lines.append(
                f"SCPR 精修步数: {ref_steps} | 输出后2/3最清晰帧: idx={ref_best_idx}, score={ref_best_score:.4f}"
            )
            try:
                os.unlink(ref_in_path)
            except OSError:
                pass
        else:
            final_image, main_best_idx, _, main_best_score = select_frame(
                out_path, method="sharpest", start_ratio=1 / 3
            )
            if loopless_enable:
                info_lines = MobiusPanel.format_run_info(
                    input_prompt,
                    int(loop_shift_skip),
                    int(loop_shift_stop_step),
                    main_best_idx,
                    main_best_score,
                )
            else:
                info_lines = IFEditPanel.format_run_info_no_scpr(
                    input_prompt,
                    use_cot,
                    use_tld,
                    int(tld_step_k),
                    float(tld_threshold_ratio),
                    main_best_idx,
                    main_best_score,
                )

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


def select_frame(video_path, method="sharpest", start_ratio=0.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error(f"无法读取视频: {video_path}")

    total_frames = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        total_frames += 1
    cap.release()

    if total_frames == 0:
        raise gr.Error("视频中没有可用帧")

    start_ratio = max(0.0, min(1.0, float(start_ratio)))
    start_idx = min(total_frames - 1, int(total_frames * start_ratio))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error(f"无法读取视频: {video_path}")

    selected_frame = None
    selected_idx = start_idx
    selected_score = None
    idx = 0

    if method == "last":
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx >= start_idx:
                selected_frame = frame
                selected_idx = idx
            idx += 1
    elif method == "sharpest":
        best_score = -1.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx < start_idx:
                idx += 1
                continue
            score = laplacian_sharpness(frame)
            if score > best_score:
                best_score = score
                selected_idx = idx
                selected_frame = frame
                selected_score = score
            idx += 1
    else:
        cap.release()
        raise gr.Error(f"未知选帧方式: {method}")

    cap.release()

    if selected_frame is None:
        raise gr.Error("选帧失败：指定范围内没有可用帧")

    final_image = cv2.cvtColor(selected_frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(final_image), selected_idx, total_frames, selected_score


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
                    frame_num = gr.Number(label="帧数 (4n+1)", value=61, precision=0, minimum=5, maximum=257)
                    sample_solver = gr.Dropdown(label="采样器", choices=["unipc", "dpm++"], value="unipc")
                    sample_steps = gr.Slider(label="采样步数", minimum=1, maximum=100, value=40, step=1)
                    seed = gr.Number(label="随机种子 (-1=随机)", value=-1, precision=0)
                    offload_model = gr.Checkbox(label="offload_model (省显存)", value=True)
                    t5_cpu = gr.Checkbox(label="t5_cpu (省显存)", value=False)
                ifedit_panel = IFEditPanel(tld_threshold_ratio=WAN_TLD_THRESHOLD_RATIO)
                (use_cot, use_tld, tld_step_k, tld_threshold_ratio, use_scpr, scpr_refinement_ratio) = (
                    ifedit_panel.create_accordion()
                )

                (loopless_enable, loop_shift_skip, loop_shift_stop_step) = MobiusPanel.create_accordion()

                loopless_enable.change(
                    fn=MobiusPanel.on_enable_disable_ifedit,
                    inputs=[loopless_enable],
                    outputs=[use_tld, tld_step_k, tld_threshold_ratio, use_cot, scpr_refinement_ratio, use_scpr],
                )
                use_tld.change(
                    fn=IFEditPanel.on_enable_disable_mobius,
                    inputs=[use_tld],
                    outputs=[loopless_enable, loop_shift_skip, loop_shift_stop_step],
                )

                btn = gr.Button("生成视频")

            with gr.Column():
                video_out = gr.Video(label="生成视频", autoplay=True)
                image_out = gr.Image(type="pil", label="主视频最清晰帧")
                used_prompt = gr.Textbox(label="实际使用 Prompt", lines=4)
                run_info = gr.Textbox(label="运行信息", lines=8)
                scpr_input_out = gr.Image(type="pil", label="SCPR 输入帧（主视频后2/3最清晰帧）", visible=False)

                loopless_enable.change(
                    fn=get_output_panel_updates_by_mode,
                    inputs=[loopless_enable, use_scpr],
                    outputs=[scpr_input_out, video_out, image_out],
                )
                use_scpr.change(
                    fn=get_output_panel_updates_by_mode,
                    inputs=[loopless_enable, use_scpr],
                    outputs=[scpr_input_out, video_out, image_out],
                )

        btn.click(
            fn=run_i2v,
            inputs=[
                image, prompt, size_name, ckpt_dir,
                frame_num, sample_solver, sample_steps,
                seed, offload_model, t5_cpu,
                use_cot, use_tld, tld_step_k, tld_threshold_ratio,
                use_scpr, scpr_refinement_ratio,
                loopless_enable, loop_shift_skip, loop_shift_stop_step,
            ],
            outputs=[video_out, image_out, used_prompt, run_info, scpr_input_out],
        )

    return app


if __name__ == "__main__":
    build_ui().launch()
