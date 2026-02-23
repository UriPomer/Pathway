"""工具函数集合 (utils)

包含：
1. 图像预处理辅助：尺寸调整、裁剪、填充
2. 拉普拉斯清晰度评估
3. 推理调度辅助：种子生成、IF-Edit pipeline 封装
"""

from __future__ import annotations

from PIL import Image, ImageOps
import random
import numpy as np
import torch
from typing import Tuple, Any, Dict, List, Optional

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
    """保持原图比例，通过填充黑色扩展至 720x480。"""
    if img.mode != "RGB":
        img = img.convert("RGB")
    target_w, target_h = 720, 480
    img_w, img_h = img.size
    scale = min(target_w / img_w, target_h / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized_img, (paste_x, paste_y))
    return canvas


def crop_center_square(img: Image.Image, size: int = 480) -> Image.Image:
    """对图像中心裁剪 size×size。若原尺寸不足则填充。"""
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if w < size or h < size:
        padded = Image.new("RGB", (max(w, size), max(h, size)), (0, 0, 0))
        padded.paste(img, ((padded.size[0] - w) // 2, (padded.size[1] - h) // 2))
        img = padded
        w, h = img.size
    left = max((w - size) // 2, 0)
    top = max((h - size) // 2, 0)
    return img.crop((left, top, left + size, top + size))


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

def build_generator(device, seed: int | float | None) -> Tuple[int, torch.Generator | None]:
    """根据 seed 生成 (实际种子, torch.Generator)。
    device 可以是 str 或 torch.device。
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


def run_ifedit(
    pipeline: Any,
    image: Image.Image,
    prompt: str,
    num_frames: int = 81,
    height: int = 480,
    width: int = 832,
    guidance_scale: float = 5.0,
    num_inference_steps: int = 40,
    generator: Optional[torch.Generator] = None,
    save_dir: str = "outputs",
    use_cot: bool = True,
    use_tld: bool = True,
    use_scpr: bool = True,
) -> Tuple[Image.Image, str, Dict[str, Any]]:
    """Run the IF-Edit pipeline and format output. height/width 0 → auto."""
    height = height or None
    width = width or None
    final_image, frames, run_info = pipeline.edit(
        image=image,
        edit_instruction=prompt,
        num_frames=num_frames,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        save_dir=save_dir,
        use_cot_prompt=use_cot,
        use_tld=use_tld,
        use_scpr=use_scpr,
    )

    info_lines = [
        f"IF-Edit: frames={run_info.get('num_frames_generated', 0)}",
        f"Selection: {run_info.get('selection_method', 'unknown')}",
    ]

    if run_info.get("temporal_prompt"):
        info_lines.append(f"CoT Prompt ({('ON' if run_info.get('cot_enabled') else 'OFF')}): "
                         f"{run_info['temporal_prompt'][:150]}")
    if run_info.get("tld_enabled"):
        info_lines.append(f"TLD: ON, K={run_info.get('tld_step_K')}, "
                         f"threshold={run_info.get('tld_threshold_ratio')}")
    else:
        info_lines.append("TLD: OFF")
    if run_info.get("scpr"):
        scpr = run_info["scpr"]
        info_lines.append(
            f"SCPR: initial_sharpness={scpr.get('initial_best_sharpness', 0):.2f}, "
            f"refined_sharpness={scpr.get('refinement_best_sharpness', 0):.2f}"
        )
    if run_info.get("video_path"):
        info_lines.append(f"Video: {run_info['video_path']}")
    if run_info.get("final_image_path"):
        info_lines.append(f"Output: {run_info['final_image_path']}")

    info = "\n".join(info_lines)
    return final_image, info, run_info


def run_pipeline_dispatch(
    pipeline: Any,
    image: Image.Image,
    prompt: str,
    num_frames: int,
    guidance_scale: float,
    seed,
    num_inference_steps: int,
    height: int = 480,
    width: int = 832,
    use_cot: bool = True,
    use_tld: bool = True,
    use_scpr: bool = True,
    save_dir: str = "outputs",
    **kwargs,
) -> Tuple[Image.Image, int, str, Dict[str, Any]]:
    """统一调度函数，供 UI 层调用。返回 (final_image, seed, info_text, run_info)。"""
    device = getattr(pipeline, "device", "cpu")
    actual_seed, generator = build_generator(device, seed)

    final_image, info, run_info = run_ifedit(
        pipeline=pipeline,
        image=image,
        prompt=prompt,
        num_frames=num_frames,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        save_dir=save_dir,
        use_cot=use_cot,
        use_tld=use_tld,
        use_scpr=use_scpr,
    )

    return final_image, actual_seed, info, run_info


__all__ = [
    "resize_to_square", "center_crop", "pad_lr_to_720x480", "fit_to_720x480",
    "crop_center_square", "postprocess_to_512",
    "build_generator", "run_ifedit", "run_pipeline_dispatch",
]
