"""工具函数集合 (utils)

包含两类：
1. 图像预处理辅助：尺寸调整、裁剪、填充
2. 推理调度辅助：种子生成、Baseline 与 Iterative 模式封装

目的：把 `app.py` 中的流程性逻辑抽离，保持主应用文件更易读。
"""

from __future__ import annotations

from PIL import Image, ImageOps
import random
import torch
from typing import Tuple, Any

# ========================
# 图像相关基础函数
# ========================
def resize_to_square(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), Image.Resampling.LANCZOS)


def center_crop(img: Image.Image, target_size: int) -> Image.Image:
    w, h = img.size
    min_side = min(w, h)
    left = (w - min_side) // 2
    top = (h - min_side) // 2
    img = img.crop((left, top, left + min_side, top + min_side))
    return img.resize((target_size, target_size), Image.Resampling.LANCZOS)


def pad_lr_to_720x480(img: Image.Image) -> Image.Image:
    """先生成 480x480 正方图，再左右填充至 720x480 (论文官方设定)。"""
    if img.mode != "RGB":
        img = img.convert("RGB")
    square = ImageOps.fit(img, (480, 480), Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    canvas = Image.new("RGB", (720, 480), (0, 0, 0))
    canvas.paste(square, ((720 - 480) // 2, 0))
    return canvas


def crop_center_square(img: Image.Image, size: int = 480) -> Image.Image:
    """对图像中心裁剪 size×size。若原尺寸不足则填充。"""
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if w < size or h < size:
        pad_w = max(size - w, 0)
        pad_h = max(size - h, 0)
        padded = Image.new("RGB", (max(w, size), max(h, size)), (0, 0, 0))
        padded.paste(img, ((padded.size[0] - w) // 2, (padded.size[1] - h) // 2))
        img = padded
        w, h = img.size
    left = max((w - size) // 2, 0)
    top = max((h - size) // 2, 0)
    right = left + size
    bottom = top + size
    return img.crop((left, top, right, bottom))


def postprocess_to_512(img: Image.Image) -> Image.Image:
    """中心裁剪 480×480 并缩放到 512×512。"""
    cropped = crop_center_square(img, 480)
    return cropped.resize((512, 512), Image.Resampling.LANCZOS)


def fit_to_720x480(img: Image.Image) -> Image.Image:
    """按比例裁剪 + 缩放到 720x480，避免填充黑边。"""
    if img.mode != "RGB":
        img = img.convert("RGB")
    return ImageOps.fit(img, (720, 480), Image.Resampling.LANCZOS, centering=(0.5, 0.5))


# ========================
# 推理流程辅助函数
# ========================
def build_generator(device: str, seed: int | float | None) -> Tuple[int, torch.Generator | None]:
    """根据 seed 生成 (实际种子, torch.Generator)。

    规则：seed < 0 或 None => 随机采样；生成器失败则返回 None。
    """
    if seed is None:
        seed = -1
    try:
        s = int(seed)
    except (ValueError, TypeError):
        s = -1
    if s < 0:
        s = random.randint(0, 2**31 - 1)
    
    gen = torch.Generator(device=device).manual_seed(s)
    return s, gen


def run_baseline(pipeline: Any, image: Image.Image, prompt: str,
                 num_frames: int, guidance_scale: float,
                 generator: torch.Generator | None, num_inference_steps: int):
    """一次性生成多帧并选出最佳帧。"""
    best_frame, frames, run_info = pipeline.run(
        image=image,
        prompt_text=prompt,
        num_frames=int(num_frames),
        guidance_scale=float(guidance_scale),
        num_inference_steps=int(num_inference_steps),
        generator=generator,
        use_iterative=False
    )
    info_lines = [f"Baseline: 总帧数={len(frames)}"]
    if isinstance(run_info, dict):
        selection = run_info.get("frame_selection")
        if selection:
            info_lines.append(
                f"选择策略={selection.get('strategy')} | 帧索引={selection.get('frame_index')}"
            )
        caption = run_info.get("temporal_caption")
        if caption:
            info_lines.append(f"Temporal Caption: {caption}")
        collage = run_info.get("frame_collage") or {}
        if collage.get("path"):
            info_lines.append(f"Collage: {collage['path']}")
        outputs = run_info.get("outputs") or {}
        if outputs.get("resized512_path"):
            info_lines.append(f"Output512: {outputs['resized512_path']}")
        if outputs.get("full_path"):
            info_lines.append(f"OutputFull: {outputs['full_path']}")
    info = "\n".join(info_lines)
    return best_frame, info, run_info


def run_iterative(pipeline: Any, image: Image.Image, prompt: str,
                  guidance_scale: float, generator: torch.Generator | None,
                  iterative_steps: int, candidates_per_step: int,
                  w_sem: float, w_step: float, w_id: float,
                  num_inference_steps: int):
    """多步迭代搜索方式生成路径并返回最终帧。"""
    best_frame, path, run_info = pipeline.run(
        image=image,
        prompt_text=prompt,
        guidance_scale=float(guidance_scale),
        num_inference_steps=int(num_inference_steps),
        generator=generator,
        use_iterative=True,
        iterative_steps=int(iterative_steps),
        candidates_per_step=int(candidates_per_step),
        w_sem=float(w_sem),
        w_step=float(w_step),
        w_id=float(w_id)
    )
    info_lines = [f"Iterative: 路径长度={len(path)}"]
    if isinstance(run_info, dict):
        metrics = run_info.get("clip_path_metrics") or {}
        if metrics:
            info_lines.append(
                f"path_length={metrics.get('path_length',0):.3f} | smoothness={metrics.get('smoothness',0):.4f}"
            )
        caption = run_info.get("temporal_caption")
        if caption:
            info_lines.append(f"Temporal Caption: {caption}")
        outputs = run_info.get("outputs") or {}
        if outputs.get("resized512_path"):
            info_lines.append(f"Output512: {outputs['resized512_path']}")
        if outputs.get("full_path"):
            info_lines.append(f"OutputFull: {outputs['full_path']}")
    info = "\n".join(info_lines)
    return best_frame, info, run_info


def run_pipeline_dispatch(pipeline: Any, image: Image.Image, prompt: str,
                          num_frames: int, guidance_scale: float, seed,
                          use_iterative: bool, iterative_steps: int,
                          candidates_per_step: int, w_sem: float, w_step: float, w_id: float,
                          num_inference_steps: int):
    """统一调度函数，供 UI 层调用。"""
    actual_seed, generator = build_generator(getattr(pipeline, 'device', 'cpu'), seed)

    # 统一获取 run_info
    run_info = {}

    if use_iterative:
        frame, info, run_info = run_iterative(
            pipeline=pipeline,
            image=image,
            prompt=prompt,
            guidance_scale=guidance_scale,
            generator=generator,
            iterative_steps=iterative_steps,
            candidates_per_step=candidates_per_step,
            w_sem=w_sem,
            w_step=w_step,
            w_id=w_id,
            num_inference_steps=num_inference_steps,
        )
    else:
        frame, info, run_info = run_baseline(
            pipeline=pipeline,
            image=image,
            prompt=prompt,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
        )

    full_image_path = run_info.get("outputs", {}).get("full_path")
    if full_image_path:
        try:
            frame = Image.open(full_image_path)
        except (FileNotFoundError, IOError) as e:
            info += f"\n[Warning] Failed to load full image: {e}"

    return frame, actual_seed, info


__all__ = [
    "resize_to_square", "center_crop", "pad_lr_to_720x480", "fit_to_720x480",
    "crop_center_square", "postprocess_to_512",
    "build_generator", "run_baseline", "run_iterative", "run_pipeline_dispatch"
]
