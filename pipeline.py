"""IF-Edit Pipeline: Zero-Shot Image Editing via I2V Models.

Core components:
1. CoT Prompt Enhancement — temporal reasoning prompts via VLM
2. Temporal Latent Dropout (TLD) — drop intermediate frame latents after threshold
3. Self-Consistent Post-Refinement (SCPR) — Laplacian sharpness + still-video refinement

Base model: Wan 2.1 I2V-14B (diffusers WanPipeline).
"""

import math
import os
import time
from typing import List, Dict, Tuple, Optional, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
from transformers import CLIPVisionModel
from diffusers_patch import WanImageToVideoPipeline

from vlm import IFEditPromptEnhancer
from utils import postprocess_to_512

# ---------------------------------------------------------------------------
# Wan 2.1 I2V default negative prompt (from official example)
# ---------------------------------------------------------------------------
DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, "
    "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


# ---------------------------------------------------------------------------
# Laplacian sharpness
# ---------------------------------------------------------------------------

def laplacian_sharpness(image: Image.Image) -> float:
    arr = np.array(image.convert("L"), dtype=np.float32)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    kernel = torch.tensor([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    filtered = F.conv2d(tensor, kernel, padding=1)
    return filtered.abs().mean().item()


def select_sharpest_frame(frames: List[Image.Image]) -> Tuple[int, Image.Image, List[float]]:
    scores = [laplacian_sharpness(f) for f in frames]
    best_idx = int(np.argmax(scores))
    return best_idx, frames[best_idx], scores


# ---------------------------------------------------------------------------
# Temporal Latent Dropout (TLD)
# ---------------------------------------------------------------------------

def temporal_latent_dropout(latents: torch.Tensor, step_K: int = 3) -> torch.Tensor:
    """Sub-sample frame latents at stride K, always keeping first and last.
    Expects (B, C, F, H, W) format (Wan 2.1).
    """
    B, C, F, H, W = latents.shape
    indices = list(range(0, F, step_K))
    if indices[-1] != F - 1:
        indices.append(F - 1)
    print(f"[TLD] Keeping {len(indices)}/{F} frames, indices: {indices}")
    return latents[:, :, torch.tensor(indices, device=latents.device), :, :]


# ---------------------------------------------------------------------------
# IF-Edit Pipeline
# ---------------------------------------------------------------------------

class IFEditPipeline:

    def __init__(
        self,
        model_path: str = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        model_cache_dir: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        prompt_enhancer: Optional[IFEditPromptEnhancer] = None,
        tld_enabled: bool = True,
        tld_threshold_ratio: float = 0.5,
        tld_step_K: int = 2,
        scpr_enabled: bool = True,
        scpr_refinement_ratio: float = 0.6,
    ):
        self.model_path = model_path
        self.model_cache_dir = model_cache_dir
        self.tld_enabled = tld_enabled
        self.tld_threshold_ratio = tld_threshold_ratio
        self.tld_step_K = tld_step_K
        self.scpr_enabled = scpr_enabled
        self.scpr_refinement_ratio = scpr_refinement_ratio
        self.prompt_enhancer = prompt_enhancer or IFEditPromptEnhancer()
        self._dtype = torch_dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)

        print(f"[IF-Edit] Loading Wan I2V model from {model_path} (dtype={self._dtype})...")
        start = time.time()

        is_wan22 = "2.2" in model_path or "Wan2.2" in model_path
        if is_wan22:
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                model_path, torch_dtype=self._dtype, cache_dir=self.model_cache_dir,
            )
        else:
            image_encoder = CLIPVisionModel.from_pretrained(
                model_path, subfolder="image_encoder", torch_dtype=torch.float32,
                cache_dir=self.model_cache_dir,
            )
            vae = AutoencoderKLWan.from_pretrained(
                model_path, subfolder="vae", torch_dtype=torch.float32,
                cache_dir=self.model_cache_dir,
            )
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                model_path, vae=vae, image_encoder=image_encoder,
                torch_dtype=self._dtype, cache_dir=self.model_cache_dir,
            )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config, flow_shift=5.0
        )
        if torch.cuda.is_available():
            self.pipe.enable_sequential_cpu_offload()
            self.device = torch.device("cuda")
        else:
            self.pipe.to("cpu")
            self.device = torch.device("cpu")

        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        print(f"[IF-Edit] Model loaded in {time.time() - start:.1f}s")

    # ------------------------------------------------------------------
    # CoT Prompt Enhancement
    # ------------------------------------------------------------------
    def enhance_prompt(self, image: Image.Image, edit_instruction: str) -> str:
        return self.prompt_enhancer.enhance(image, edit_instruction)

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------
    def compute_resolution(self, image: Image.Image, max_area: int = 480 * 832) -> Tuple[int, int]:
        """Compute (height, width) aligned to model requirements, preserving aspect ratio.
        Uses the same formula as the official Wan2.1 I2V example.
        """
        vae_sf = self.pipe.vae_scale_factor_spatial  # 8
        patch_size = self.pipe.transformer.config.patch_size[1]  # typically 2
        mod_value = vae_sf * patch_size  # 16

        aspect_ratio = image.height / image.width
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        return int(height), int(width)

    # ------------------------------------------------------------------
    # Video Generation
    # ------------------------------------------------------------------
    def generate_video(
        self,
        image: Image.Image,
        prompt: str,
        num_frames: int = 81,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 40,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[List[Image.Image], Dict[str, Any]]:
        image = image.convert("RGB")

        if height is None or width is None:
            height, width = self.compute_resolution(image)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        flow_shift = 3.0 if (height * width) <= 480 * 832 else 5.0
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config, flow_shift=flow_shift
        )

        print(f"[IF-Edit] Generating video: {num_frames} frames, {width}x{height}")
        print(f"[IF-Edit] Prompt: {prompt[:100]}...")

        debug_info = {
            "input_image": image.copy(),
            "height": height,
            "width": width,
            "prompt": prompt,
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
            "num_frames": num_frames,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
        }

        pipe_kwargs = dict(
            image=image,
            prompt=prompt,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            num_frames=num_frames,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type="pil",
        )
        if self.tld_enabled:
            tld_callback = self._make_tld_callback(num_inference_steps)
            pipe_kwargs.update(
                callback_on_step_end=tld_callback,
                callback_on_step_end_tensor_inputs=["latents", "condition"],
            )

        result = self.pipe(**pipe_kwargs)
        frames = result.frames[0]
        print(f"[IF-Edit] Generated {len(frames)} frames")
        return frames, debug_info

    def _make_tld_callback(self, num_inference_steps: int):
        """Create a callback_on_step_end that applies TLD at the threshold step."""
        T_th = int(num_inference_steps * self.tld_threshold_ratio)
        step_K = self.tld_step_K
        applied = {"done": False}

        def tld_callback(pipe, i, t, callback_kwargs):
            if i >= T_th and not applied["done"]:
                latents = callback_kwargs["latents"]
                condition = callback_kwargs["condition"]
                orig = latents.shape
                callback_kwargs["latents"] = temporal_latent_dropout(latents, step_K=step_K)
                callback_kwargs["condition"] = temporal_latent_dropout(condition, step_K=step_K)
                applied["done"] = True
                print(f"[TLD] Applied at step {i}/{num_inference_steps}: "
                      f"latents {orig} -> {callback_kwargs['latents'].shape}")

                # Reset scheduler caches to avoid shape mismatch between
                # old (pre-TLD) model_outputs/last_sample and new (post-TLD) latents.
                scheduler = pipe.scheduler
                solver_order = getattr(scheduler.config, "solver_order", 1)
                scheduler.model_outputs = [None] * solver_order
                scheduler.timestep_list = [None] * solver_order
                scheduler.lower_order_nums = 0
                scheduler.last_sample = None
            return callback_kwargs

        return tld_callback

    # ------------------------------------------------------------------
    # Self-Consistent Post-Refinement (SCPR)
    # ------------------------------------------------------------------
    def self_consistent_refinement(
        self,
        frames: List[Image.Image],
        main_inference_steps: int,
        guidance_scale: float = 5.0,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        steps = math.ceil(main_inference_steps * self.scpr_refinement_ratio)
        print(f"[SCPR] Refinement steps: {steps} "
              f"({self.scpr_refinement_ratio*100:.0f}% of {main_inference_steps})")

        best_idx, best_frame, sharpness_scores = select_sharpest_frame(frames)
        print(f"[SCPR] Selected frame {best_idx} (sharpness={sharpness_scores[best_idx]:.2f})")

        still_prompt = (
            "A perfectly still, high-resolution photograph with exceptional clarity "
            "and fine detail. The image remains completely static with no motion, "
            "camera shake, or blur. Ultra-sharp focus throughout the entire frame."
        )
        h, w = best_frame.size[1], best_frame.size[0]

        prev_tld = self.tld_enabled
        self.tld_enabled = False
        refinement_frames, _ = self.generate_video(
            image=best_frame, prompt=still_prompt,
            num_frames=17, height=h, width=w,
            guidance_scale=guidance_scale, num_inference_steps=steps,
            generator=generator,
        )
        self.tld_enabled = prev_tld

        ref_idx, refined_image, ref_scores = select_sharpest_frame(refinement_frames)
        print(f"[SCPR] Refinement: selected frame {ref_idx} "
              f"(sharpness={ref_scores[ref_idx]:.2f})")

        info = {
            "initial_best_idx": best_idx,
            "initial_best_sharpness": sharpness_scores[best_idx],
            "refinement_best_idx": ref_idx,
            "refinement_best_sharpness": ref_scores[ref_idx],
        }
        return refined_image, info

    # ------------------------------------------------------------------
    # Full IF-Edit pipeline
    # ------------------------------------------------------------------
    def edit(
        self,
        image: Image.Image,
        edit_instruction: str,
        num_frames: int = 81,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 40,
        generator: Optional[torch.Generator] = None,
        save_dir: str = "outputs",
        use_cot_prompt: bool = True,
        use_tld: Optional[bool] = None,
        use_scpr: Optional[bool] = None,
    ) -> Tuple[Image.Image, List[Image.Image], Dict[str, Any]]:
        os.makedirs(save_dir, exist_ok=True)
        run_info: Dict[str, Any] = {"edit_instruction": edit_instruction}

        # Step 1: CoT Prompt Enhancement
        if use_cot_prompt:
            temporal_prompt = self.enhance_prompt(image, edit_instruction)
            print(f"[IF-Edit] CoT prompt: {temporal_prompt[:120]}...")
        else:
            temporal_prompt = edit_instruction
        run_info["temporal_prompt"] = temporal_prompt
        run_info["cot_enabled"] = use_cot_prompt

        # Step 2: Video Generation (with optional TLD)
        tld_active = use_tld if use_tld is not None else self.tld_enabled
        prev_tld = self.tld_enabled
        if use_tld is not None:
            self.tld_enabled = use_tld

        run_info["tld_enabled"] = tld_active
        run_info["tld_step_K"] = self.tld_step_K
        run_info["tld_threshold_ratio"] = self.tld_threshold_ratio

        frames, video_debug = self.generate_video(
            image=image, prompt=temporal_prompt,
            num_frames=num_frames, height=height, width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        self.tld_enabled = prev_tld
        run_info["debug_input_image"] = video_debug["input_image"]
        run_info["debug_height"] = video_debug["height"]
        run_info["debug_width"] = video_debug["width"]
        run_info["debug_prompt"] = video_debug["prompt"]
        run_info["debug_negative_prompt"] = video_debug["negative_prompt"]
        run_info["debug_num_frames"] = video_debug["num_frames"]
        run_info["debug_guidance_scale"] = video_debug["guidance_scale"]
        run_info["debug_num_inference_steps"] = video_debug["num_inference_steps"]
        run_info["num_frames_generated"] = len(frames)

        # Save video
        video_path = os.path.join(save_dir, f"generated_{int(time.time())}.mp4")
        self._save_video(frames, video_path)
        run_info["video_path"] = video_path

        # Step 3: Frame Selection + SCPR
        scpr_active = use_scpr if use_scpr is not None else self.scpr_enabled
        run_info["scpr_enabled"] = scpr_active
        if scpr_active:
            final_image, scpr_info = self.self_consistent_refinement(
                frames=frames, main_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale, generator=generator,
            )
            run_info["scpr"] = scpr_info
            run_info["selection_method"] = "SCPR"
        else:
            _, final_image, scores = select_sharpest_frame(frames)
            run_info["sharpness_scores"] = scores
            run_info["selection_method"] = "Laplacian (sharpest frame)"

        final_path = os.path.join(save_dir, "final_edit.png")
        final_image.save(final_path)
        run_info["final_image_path"] = final_path

        img_512 = postprocess_to_512(final_image)
        img_512.save(os.path.join(save_dir, "final_edit_512.png"))

        return final_image, frames, run_info

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _save_video(frames: List[Image.Image], save_path: str, fps: int = 16):
        w, h = frames[0].size
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"[IF-Edit] Saved video: {save_path} ({len(frames)} frames, {fps} fps)")
