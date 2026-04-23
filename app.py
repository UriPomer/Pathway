import os
os.environ.pop("OMP_NUM_THREADS", None)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)

import re
import subprocess
import sys
import tempfile
import math
from datetime import datetime
from typing import Any, Optional

import cv2
import gradio as gr
import numpy as np
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
WAN2_DIR = os.path.join(BASE, "wan")
OUTPUT_DIR = os.path.join(BASE, "wan", "outputs")
LOG_DIR = os.path.join(BASE, "logs")
if WAN2_DIR not in sys.path:
    sys.path.insert(0, WAN2_DIR)
from generate import generate, make_i2v_args  # type: ignore[import-untyped]
from wan.configs.param_types import (
    FGParams, GenerateParams, IFEditParams, LooplessParams, ModelParams, SamplingParams,
)
from vlm import IFEditPromptEnhancer
from ui_panels import IFEditPanel, MobiusPanel, FrameGuidancePanel, get_output_panel_updates_by_mode
from wan.configs.wan_i2v_A14B import i2v_A14B
I2V_SIZES = ("720*1280", "1280*720", "480*832", "832*480")
IDLE_GPU_MB = 500
DEFAULT_CKPT = ""  # Auto-resolved from WAN2_CKPT_DIR env or auto-downloaded
DEFAULT_IMAGE = os.path.join(BASE, "example", "person.jpg")
DEFAULT_PROMPT = "A black fedora materializes on the character's head. Locked-off camera, no camera movement."
# DEFAULT_PROMPT = "Close The Door"
SCPR_STILL_PROMPT = (
    "A perfectly still, high-resolution photograph with exceptional clarity "
    "and fine detail. The image remains completely static with no motion, "
    "camera shake, or blur. Ultra-sharp focus throughout the entire frame."
)
OFFICIAL_GUIDE_SCALE = 3.5
# Official Wan negative prompt (wan/configs/shared_config.py). Chinese; matches
# what the model was trained with — safer than custom English prompts.
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)
WAN_TLD_THRESHOLD_RATIO = float(getattr(i2v_A14B, "boundary", 0.9))

# import subprocess
# import os

# result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
# output = result.stdout
# for line in output.splitlines():
#     if '=' in line:
#         var, value = line.split('=', 1)
#         os.environ[var] = value

def auto_sample_shift_by_size(size_name: str) -> float:
    return 3.0 if size_name in ("480*832", "832*480") else 5.0


def auto_loop_shift_skip_by_frame_num(frame_num: int) -> int:
    frame_num = int(frame_num)
    latent_len = max(2, (frame_num - 1) // 4 + 1)
    movable_len = max(1, latent_len - 1)
    if movable_len <= 2:
        return 1

    base = max(1, movable_len // 3)
    candidates = []
    for d in range(movable_len + 1):
        a = base + d
        b = base - d
        if 1 <= a < movable_len:
            candidates.append(a)
        if d > 0 and 1 <= b < movable_len:
            candidates.append(b)

    for c in candidates:
        if math.gcd(c, movable_len) == 1:
            return c
    return 1

_PROMPT_ENHANCER = IFEditPromptEnhancer(strict=True)


class _TeeWriter:
    """Tee stdout/stderr to both terminal and a log file."""

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return self.terminal.isatty()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


def _start_session_log():
    """Start capturing stdout/stderr to a timestamped log file under logs/."""
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"session_{ts}.log")
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _TeeWriter(sys.stdout, log_file)
    sys.stderr = _TeeWriter(sys.stderr, log_file)
    print(f"[Log] Session log started: {log_path}")
    return log_path


def get_least_used_gpu(threshold_mb=IDLE_GPU_MB):
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


def _parse_area(size_name: str) -> int:
    w_str, h_str = size_name.split("*")
    return int(w_str) * int(h_str)


def _compute_i2v_target_hw(image: Image.Image, size_name: str) -> tuple[int, int]:
    max_area = _parse_area(size_name)
    vae_stride_h, vae_stride_w = i2v_A14B.vae_stride[1], i2v_A14B.vae_stride[2]
    patch_h, patch_w = i2v_A14B.patch_size[1], i2v_A14B.patch_size[2]
    h, w = image.size[1], image.size[0]
    aspect_ratio = h / w
    lat_h = round(np.sqrt(max_area * aspect_ratio) // vae_stride_h // patch_h * patch_h)
    lat_w = round(np.sqrt(max_area / aspect_ratio) // vae_stride_w // patch_w * patch_w)
    return lat_h * vae_stride_h, lat_w * vae_stride_w




def _extract_sketch_from_canvas(canvas_value: Any) -> Optional[Image.Image]:
    """Extract brush strokes from gr.ImageEditor canvas.

    Returns an RGB PIL Image: white strokes on black background (binary).
    Uses the 'layers' field which contains ONLY brush strokes (no background).
    """
    if canvas_value is None:
        return None

    # Handle string path (Gradio cache on subsequent runs)
    if isinstance(canvas_value, str):
        if os.path.isfile(canvas_value):
            pil_img = Image.open(canvas_value).convert("L")
            mask = np.array(pil_img, dtype=np.float32) > 30
            if not np.any(mask):
                return None
            sketch = np.zeros((*mask.shape, 3), dtype=np.uint8)
            sketch[mask] = 255
            return Image.fromarray(sketch)
        return None

    # gr.ImageEditor returns dict with "composite", "background", "layers"
    if isinstance(canvas_value, dict):
        layers = canvas_value.get("layers")
        if layers and len(layers) > 0:
            # layers is a list of RGBA images (transparent bg, only brush strokes)
            # Merge all layers into one mask
            first = layers[0]
            if isinstance(first, np.ndarray):
                h, w = first.shape[:2]
            elif isinstance(first, Image.Image):
                w, h = first.size
            elif isinstance(first, str) and os.path.isfile(first):
                first = Image.open(first)
                w, h = first.size
                layers[0] = first
            else:
                h, w = 480, 832  # fallback

            merged = np.zeros((h, w), dtype=np.float32)
            for layer in layers:
                if isinstance(layer, str) and os.path.isfile(layer):
                    layer = Image.open(layer)
                if isinstance(layer, Image.Image):
                    layer_np = np.array(layer.convert("RGBA").resize((w, h), Image.BILINEAR))
                elif isinstance(layer, np.ndarray):
                    layer_np = layer
                else:
                    continue
                # Use alpha channel as stroke mask
                if layer_np.shape[-1] == 4:
                    alpha = layer_np[:, :, 3].astype(np.float32)
                    merged = np.maximum(merged, alpha)
                else:
                    gray = np.mean(layer_np[:, :, :3].astype(np.float32), axis=2)
                    merged = np.maximum(merged, gray)

            stroke_mask = merged > 30
            if not np.any(stroke_mask):
                return None
            sketch = np.zeros((h, w, 3), dtype=np.uint8)
            sketch[stroke_mask] = 255
            return Image.fromarray(sketch)

        # No layers — fall back to composite
        composite = canvas_value.get("composite")
        if composite is None:
            return None
        if isinstance(composite, str) and os.path.isfile(composite):
            composite = Image.open(composite)
        if isinstance(composite, np.ndarray):
            composite = Image.fromarray(composite.astype(np.uint8))
        if isinstance(composite, Image.Image):
            gray = np.array(composite.convert("L"), dtype=np.float32)
            if gray.max() < 10:
                return None
            mask = gray > 30
            sketch = np.zeros((*gray.shape, 3), dtype=np.uint8)
            sketch[mask] = 255
            return Image.fromarray(sketch)
        return None

    elif isinstance(canvas_value, Image.Image):
        pil_img = canvas_value
    elif isinstance(canvas_value, np.ndarray):
        pil_img = Image.fromarray(canvas_value.astype(np.uint8))
    else:
        return None

    # Fallback: threshold to binary
    gray = np.array(pil_img.convert("L"), dtype=np.float32)
    if gray.max() < 10:
        return None
    mask = gray > 30
    sketch = np.zeros((*gray.shape, 3), dtype=np.uint8)
    sketch[mask] = 255
    return Image.fromarray(sketch)


def run_i2v(
    image_path_str,
    prompt,
    size_name,
    ckpt_dir,
    backend,
    frame_num,
    sample_solver,
    sample_steps,
    guide_scale,
    negative_prompt,
    seed,
    offload_model,
    t5_cpu,
    use_cot,
    use_tld,
    tld_stop_fg,
    tld_step_k,
    tld_threshold_ratio,
    use_scpr,
    scpr_refinement_ratio,
    loopless_enable,
    loop_shift_skip,
    loop_shift_stop_step,
    fg_loop_enable=False,
    fg_loop_lr=3.0,
    fg_loop_downscale=4,
    fg_enable=False,
    fg_loss_type="草图引导 (Sketch)",
    fg_lr=10.0,
    fg_downscale=4,
    fg_style_image=None,
    fg_travel_start=-1,
    fg_travel_end=-1,
    fg_sketch_canvas=None,
):
    # Load image from server-side path (skips Gradio upload channel).
    image_path_str = (image_path_str or "").strip()
    if image_path_str and os.path.isfile(image_path_str):
        image = Image.open(image_path_str).convert("RGB")
        img_path = image_path_str
    else:
        # No input image: pass None to generate() to disable first-frame conditioning.
        # The I2V model will run in "T2V-like" mode with zeroed y condition,
        # relying purely on prompt + FG guidance.
        print(f"[T2V-MODE] No input image at path={image_path_str!r} — first-frame conditioning disabled", flush=True)
        image = None
        img_path = None
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
    fg_loss_fn = FrameGuidancePanel.get_loss_fn_name(fg_loss_type)
    fg_additional_inputs = {}

    total_frames = int(frame_num)

    # Determine effective FG mode: either FG panel (style/sketch) or Mobius FG Loop
    effective_fg_enable = bool(fg_enable)
    effective_fg_loss_fn = fg_loss_fn
    effective_fg_lr = float(fg_lr)
    effective_fg_downscale = int(fg_downscale)

    # FG Loop from Mobius panel overrides FG panel if both are on
    if bool(loopless_enable) and bool(fg_loop_enable):
        effective_fg_enable = True
        effective_fg_loss_fn = "loop"
        effective_fg_lr = float(fg_loop_lr)
        effective_fg_downscale = int(fg_loop_downscale)

    # Scribble: extract brush overlay from image editor, fixed to last frame
    fg_fixed_frames = None
    sketch_preview_img = None
    if effective_fg_enable:
        if effective_fg_loss_fn == "style":
            if not fg_style_image:
                raise gr.Error("风格化 (Style) 模式需要上传风格参考图。")
            fg_additional_inputs["style_image"] = Image.open(fg_style_image).convert("RGB")
            # Official style fixed_frames (others_wan.ipynb cell 3)
            fg_fixed_frames = [20, 40, 60, 80]
        elif effective_fg_loss_fn == "sketch":
            sketch_image = _extract_sketch_from_canvas(fg_sketch_canvas)
            if sketch_image is None:
                raise gr.Error("自由草图 (Sketch) 模式需要在画布上绘制。")
            # Resize sketch to pipeline target resolution to avoid non-uniform
            # F.interpolate scaling in the loss computation.
            if image is not None:
                target_h, target_w = _compute_i2v_target_hw(image, size_name)
                sketch_image = sketch_image.resize((target_w, target_h), Image.BILINEAR)
            fg_additional_inputs["sketch_image"] = sketch_image
            fg_fixed_frames = [total_frames - 1]  # always constrain the last frame
            sketch_preview_img = sketch_image

    # Yield sketch preview immediately so user can see it before generation starts
    yield (
        sketch_preview_img,  # sketch_preview
        None, None, None,    # video_out, video_out_original, image_out
        None, "正在生成...", None, None,  # used_prompt, run_info, scpr_input_out, seed_used_out
    )

    ckpt_dir = ckpt_dir.strip()
    input_prompt = prompt.strip()
    if use_cot:
        if image is None:
            raise gr.Error("CoT 模式需要输入图片")
        cot_prompt = _PROMPT_ENHANCER.enhance(image.convert("RGB"), input_prompt)
        if not cot_prompt.strip():
            raise gr.Error("CoT 生成失败：返回空文本")
        input_prompt = cot_prompt.strip()

    dev = get_least_used_gpu()
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if dev:
        os.environ["CUDA_VISIBLE_DEVICES"] = dev

    # img_path already set from image_path_str above — no need to write temp file.

    mode_subdir = "mobius" if loopless_enable else "ifedit"
    mode_output_dir = os.path.join(OUTPUT_DIR, mode_subdir)
    os.makedirs(mode_output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(mode_output_dir, f"i2v_{ts}.mp4")
    out_path_original = os.path.join(mode_output_dir, f"i2v_{ts}_original.mp4") if loopless_enable else None
    auto_sample_shift = auto_sample_shift_by_size(size_name)
    effective_loop_shift_skip = auto_loop_shift_skip_by_frame_num(int(frame_num)) if bool(loopless_enable) else int(loop_shift_skip)
    sample_guide_scale = float(guide_scale)
    effective_neg_prompt = (negative_prompt or "").strip() or DEFAULT_NEGATIVE_PROMPT
    # travel_time: use UI values directly
    # (-1, -1) = no re-noising; (0, steps) = full re-noising; (a, b) = re-noise steps a..b
    fg_travel_time = (int(fg_travel_start), int(fg_travel_end))

    params = make_i2v_args(
        image=img_path,
        prompt=input_prompt,
        n_prompt=effective_neg_prompt,
        size=size_name,
        ckpt_dir=ckpt_dir,
        save_file=out_path,
        frame_num=int(frame_num),
        seed=int(seed),
        sampling=SamplingParams(
            solver=sample_solver,
            steps=int(sample_steps),
            shift=auto_sample_shift,
            guide_scale=sample_guide_scale,
        ),
        model=ModelParams(
            offload_model=offload_model,
            t5_cpu=t5_cpu,
            backend=backend or "diffusers",
        ),
        fg=FGParams(
            enable=effective_fg_enable,
            loss_fn=effective_fg_loss_fn,
            fixed_frames=fg_fixed_frames if fg_fixed_frames else None,
            guidance_lr=effective_fg_lr,
            travel_time=fg_travel_time,
            downscale_factor=effective_fg_downscale,
            additional_inputs=fg_additional_inputs if fg_additional_inputs else None,
        ),
        ifedit=IFEditParams(
            use_tld=use_tld,
            tld_threshold_ratio=float(tld_threshold_ratio),
            tld_step_k=int(tld_step_k),
            tld_stop_fg=bool(tld_stop_fg),
        ),
        loopless=LooplessParams(
            enable=bool(loopless_enable),
            shift_skip=int(effective_loop_shift_skip),
            shift_stop_step=int(loop_shift_stop_step),
        ),
    )

    # --- Timing & TLD logging ---
    import time as _time
    tld_flag_str = f"TLD=ON(K={int(tld_step_k)}, threshold_ratio={float(tld_threshold_ratio):.2f})" if use_tld else "TLD=OFF"
    print(f"[TIMING] Generation start | backend={backend} steps={int(sample_steps)} frames={int(frame_num)} {tld_flag_str}", flush=True)
    _gen_t0 = _time.perf_counter()

    generate(params)

    _gen_elapsed = _time.perf_counter() - _gen_t0
    _mins, _secs = divmod(_gen_elapsed, 60)
    print(f"[TIMING] Generation done | elapsed={_gen_elapsed:.2f}s ({int(_mins)}m{_secs:.2f}s) | {tld_flag_str}", flush=True)

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
        scpr_shift = 1.0
        scpr_guide_scale = 1.0
        ref_args = make_i2v_args(
            image=ref_in_path,
            prompt=SCPR_STILL_PROMPT,
            size=size_name,
            ckpt_dir=ckpt_dir,
            save_file=ref_out_path,
            frame_num=17,
            seed=int(seed),
            sampling=SamplingParams(
                solver=sample_solver,
                steps=ref_steps,
                shift=scpr_shift,
                guide_scale=scpr_guide_scale,
            ),
            model=ModelParams(
                offload_model=offload_model,
                t5_cpu=t5_cpu,
                backend=backend or "diffusers",
            ),
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
                int(effective_loop_shift_skip),
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

    if effective_fg_enable:
        if effective_fg_loss_fn == "style":
            fg_frames_desc = "all frames"
        elif effective_fg_loss_fn == "loop":
            fg_frames_desc = "first/last"
        else:
            fg_frames_desc = str(fg_fixed_frames)
        info_lines.append(
            f"[Frame Guidance] ON | loss={effective_fg_loss_fn} | lr={effective_fg_lr:.2f} | "
            f"downscale={effective_fg_downscale} | frames={fg_frames_desc}"
        )
    else:
        info_lines.append("[Frame Guidance] OFF")

    info_lines.append(
        f"[Timing] total={_gen_elapsed:.2f}s ({int(_mins)}m{_secs:.2f}s) | {tld_flag_str}"
    )

    yield (
        sketch_preview_img,
        out_path,
        out_path_original,
        final_image,
        input_prompt,
        "\n".join(info_lines),
        ref_input_image,
        int(params.seed),
    )

    # (img_path is a user-provided server path — do NOT delete it.)


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
    from model_utils import ensure_wan_checkpoint
    default_ckpt = os.environ.get("WAN2_CKPT_DIR", DEFAULT_CKPT)
    default_ckpt = ensure_wan_checkpoint("i2v-A14B", default_ckpt or None)

    def _load_local_preview(path: str):
        """Load a server-side image path and return a downsampled preview.
        Skips Gradio's upload channel entirely — avoids the multi-minute stall
        on multi-MP images."""
        if not path or not os.path.isfile(path):
            return None
        try:
            im = Image.open(path).convert("RGB")
            im.thumbnail((1280, 1280), Image.BILINEAR)
            return im
        except Exception as e:
            print(f"[Image] failed to open {path}: {e}", flush=True)
            return None

    with gr.Blocks(title="Wan2.2 I2V") as app:
        gr.Markdown("# Wan2.2 Image-to-Video")

        with gr.Row():
            with gr.Column():
                image_path = gr.Textbox(
                    label="输入图片路径 (服务器文件路径)",
                    value=DEFAULT_IMAGE,
                    info="直接填服务器本地路径，不走 Gradio 上传通道。避免大图卡死。",
                )
                image_preview = gr.Image(
                    type="pil",
                    label="缩略图预览",
                    value=_load_local_preview(DEFAULT_IMAGE),
                    interactive=False,
                )
                sketch_canvas = gr.ImageEditor(
                    type="pil",
                    label="草图画布（白色笔刷绘制最后一帧目标轮廓）",
                    value=_load_local_preview(DEFAULT_IMAGE),
                    brush=gr.Brush(colors=["#ffffff"], color_mode="fixed", default_size=8),
                )
                prompt = gr.Textbox(label="提示词", value=DEFAULT_PROMPT, lines=3)
                size_name = gr.Dropdown(label="分辨率", choices=list(I2V_SIZES), value="832*480")
                backend_choice = gr.Dropdown(
                    label="Backend (后端引擎)",
                    choices=["diffusers", "wan22"],
                    value="diffusers",
                    info="diffusers: Wan 2.1 单模型 | wan22: Wan 2.2 双Expert",
                )

                with gr.Accordion("高级参数", open=False):
                    frame_num = gr.Number(label="帧数 (4n+1)", value=81, precision=0, minimum=5, maximum=257)
                    sample_solver = gr.Dropdown(label="采样器", choices=["unipc", "dpm++"], value="unipc")
                    sample_steps = gr.Slider(label="采样步数", minimum=1, maximum=100, value=20, step=1)
                    guide_scale = gr.Slider(
                        label="CFG Guidance Scale (文本权重)",
                        minimum=1.0, maximum=10.0, value=OFFICIAL_GUIDE_SCALE, step=0.5,
                        info="Wan 官方默认 3.5",
                    )
                    negative_prompt = gr.Textbox(
                        label="负面提示词 (Negative Prompt)",
                        value=DEFAULT_NEGATIVE_PROMPT,
                        lines=3,
                        info="Wan 官方默认负面提示词（中文）",
                    )
                    seed = gr.Number(label="随机种子 (-1=随机)", value=-1, precision=0)
                    offload_model = gr.Checkbox(label="offload_model (省显存)", value=True)
                    t5_cpu = gr.Checkbox(label="t5_cpu (省显存)", value=False)
                    ckpt_dir = gr.Textbox(
                        label="模型目录 (ckpt_dir, wan22模式)",
                        value=default_ckpt,
                        placeholder="自动从 WAN2_CKPT_DIR 环境变量读取",
                    )

                ifedit_panel = IFEditPanel(tld_threshold_ratio=WAN_TLD_THRESHOLD_RATIO)
                (use_cot, use_tld, tld_stop_fg, tld_step_k, tld_threshold_ratio, use_scpr, scpr_refinement_ratio) = (
                    ifedit_panel.create_accordion()
                )

                (loopless_enable, loop_shift_skip, loop_shift_stop_step,
                 fg_loop_enable, fg_loop_lr, fg_loop_downscale) = MobiusPanel.create_accordion()

                (fg_enable, fg_loss_type, fg_lr, fg_downscale, fg_style_image, fg_travel_start, fg_travel_end) = (
                    FrameGuidancePanel.create_accordion()
                )

                def _auto_sync_loop_skip(frame_num_val, loopless_flag):
                    if bool(loopless_flag):
                        return gr.update(value=int(auto_loop_shift_skip_by_frame_num(int(frame_num_val))))
                    return gr.update()

                loopless_enable.change(
                    fn=MobiusPanel.on_enable_disable_ifedit,
                    inputs=[loopless_enable],
                    outputs=[use_tld, tld_step_k, tld_threshold_ratio, use_cot, scpr_refinement_ratio, use_scpr],
                )
                loopless_enable.change(
                    fn=MobiusPanel.on_enable_change,
                    inputs=[loopless_enable],
                    outputs=[fg_loop_enable, fg_loop_lr, fg_loop_downscale],
                )
                loopless_enable.change(
                    fn=_auto_sync_loop_skip,
                    inputs=[frame_num, loopless_enable],
                    outputs=[loop_shift_skip],
                )
                frame_num.change(
                    fn=_auto_sync_loop_skip,
                    inputs=[frame_num, loopless_enable],
                    outputs=[loop_shift_skip],
                )
                use_tld.change(
                    fn=IFEditPanel.on_enable_disable_mobius,
                    inputs=[use_tld],
                    outputs=[loopless_enable, loop_shift_skip, loop_shift_stop_step,
                             fg_loop_enable, fg_loop_lr, fg_loop_downscale],
                )
                fg_loss_type.change(
                    fn=FrameGuidancePanel.on_loss_type_change,
                    inputs=[fg_loss_type],
                    outputs=[fg_style_image],
                )

                image_path.change(
                    fn=lambda p: (_load_local_preview(p), _load_local_preview(p)),
                    inputs=[image_path],
                    outputs=[image_preview, sketch_canvas],
                )

                btn = gr.Button("生成视频")

            # ── 输出区域（所有组件只定义一次）─────────────────────
            with gr.Column():
                sketch_preview = gr.Image(type="pil", label="提取的草图预览", visible=True)
                video_out = gr.Video(label="生成视频 (Mobius VAE Decode)", autoplay=True)
                video_out_original = gr.Video(label="对比视频 (原始 VAE Decode)", autoplay=True, visible=False)
                image_out = gr.Image(type="pil", label="主视频最清晰帧")
                used_prompt = gr.Textbox(label="实际使用 Prompt", lines=4)
                run_info = gr.Textbox(label="运行信息", lines=8)
                scpr_input_out = gr.Image(type="pil", label="SCPR 输入帧（主视频后2/3最清晰帧）", visible=False)
                seed_used_out = gr.Number(label="本次 Seed", precision=0)

                def _toggle_original_video(loopless):
                    return gr.update(visible=bool(loopless))
                loopless_enable.change(fn=get_output_panel_updates_by_mode, inputs=[loopless_enable, use_scpr], outputs=[scpr_input_out, video_out, image_out])
                loopless_enable.change(fn=_toggle_original_video, inputs=[loopless_enable], outputs=[video_out_original])
                use_scpr.change(fn=get_output_panel_updates_by_mode, inputs=[loopless_enable, use_scpr], outputs=[scpr_input_out, video_out, image_out])

        # ── 主按钮绑定（放在两个 Column 都定义完之后）─────────
        btn.click(
            fn=run_i2v,
            inputs=[
                image_path, prompt, size_name, ckpt_dir, backend_choice,
                frame_num, sample_solver, sample_steps,
                guide_scale, negative_prompt,
                seed, offload_model, t5_cpu,
                use_cot, use_tld, tld_stop_fg, tld_step_k, tld_threshold_ratio,
                use_scpr, scpr_refinement_ratio,
                loopless_enable, loop_shift_skip, loop_shift_stop_step,
                fg_loop_enable, fg_loop_lr, fg_loop_downscale,
                fg_enable, fg_loss_type, fg_lr, fg_downscale,
                fg_style_image, fg_travel_start, fg_travel_end, sketch_canvas,
            ],
            outputs=[
                sketch_preview,
                video_out, video_out_original, image_out,
                used_prompt, run_info, scpr_input_out, seed_used_out,
            ],
        )

    return app


if __name__ == "__main__":
    _start_session_log()
    build_ui().launch()
