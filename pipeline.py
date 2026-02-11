"""IF-Edit Pipeline: Zero-Shot Image Editing via Image-to-Video Models.

Implements the IF-Edit framework from "Are Image-to-Video Models Good Zero-Shot
Image Editors?" (arXiv: 2511.19435).

Core components:
1. CoT Prompt Enhancement — temporal reasoning prompts via VLM
2. Temporal Latent Dropout (TLD) — drop redundant intermediate frame latents
   after expert-switch threshold during denoising
3. Self-Consistent Post-Refinement (SCPR) — Laplacian sharpness scoring +
   still-video refinement pass

Base model: Wan 2.1 I2V-14B (compatible with diffusers WanPipeline).
"""

import os
import time
from typing import List, Dict, Tuple, Optional, Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DiffusionPipeline

from vlm import (
    IFEditPromptEnhancer,
    TemporalCaptionGenerator,
)
from utils import pad_lr_to_720x480, crop_center_square, postprocess_to_512


# ---------------------------------------------------------------------------
# Laplacian sharpness utilities
# ---------------------------------------------------------------------------

def laplacian_sharpness(image: Image.Image) -> float:
    """Compute Laplacian-based sharpness score for a PIL image.

    s = (1 / HW) * sum |∇²I(u,v)|
    """
    arr = np.array(image.convert("L"), dtype=np.float64)
    # Laplacian kernel
    laplacian = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]], dtype=np.float64)
    from scipy.signal import convolve2d
    filtered = convolve2d(arr, laplacian, mode="same", boundary="symm")
    score = np.abs(filtered).mean()
    return float(score)


def laplacian_sharpness_torch(image: Image.Image) -> float:
    """Torch-based Laplacian sharpness (no scipy dependency)."""
    arr = np.array(image.convert("L"), dtype=np.float32)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    kernel = torch.tensor([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    filtered = F.conv2d(tensor, kernel, padding=1)
    score = filtered.abs().mean().item()
    return score


def select_sharpest_frame(frames: List[Image.Image]) -> Tuple[int, Image.Image, List[float]]:
    """Select the sharpest frame using Laplacian scoring.

    Returns (best_index, best_frame, all_scores).
    """
    scores = []
    for frame in frames:
        try:
            s = laplacian_sharpness(frame)
        except ImportError:
            s = laplacian_sharpness_torch(frame)
        scores.append(s)
    best_idx = int(np.argmax(scores))
    return best_idx, frames[best_idx], scores


# ---------------------------------------------------------------------------
# Temporal Latent Dropout (TLD) helpers
# ---------------------------------------------------------------------------

def temporal_latent_dropout(
    latents: torch.Tensor,
    step_K: int = 3,
    total_frames: Optional[int] = None,
) -> torch.Tensor:
    """Sub-sample frame latents at stride K, always keeping first and last.

    Args:
        latents: Tensor of shape (B, F, C, H, W) or (B, C, F, H, W)
        step_K: Sub-sampling stride
        total_frames: Override for total frame count (auto-detected if None)

    Returns:
        Sub-sampled latents with the same batch/channel/spatial dims.
    """
    # Detect frame dimension — Wan uses (B, C, F, H, W)
    # We check if dim 1 or dim 2 is the frame dim based on typical sizes
    if latents.dim() == 5:
        # Wan 2.1 pipeline uses (B, C, F, H, W) format
        B, C, F, H, W = latents.shape
        if total_frames is None:
            total_frames = F

        # Build indices: {0, K, 2K, ..., F-1}
        indices = list(range(0, F, step_K))
        if indices[-1] != F - 1:
            indices.append(F - 1)
        indices_tensor = torch.tensor(indices, device=latents.device)
        return latents[:, :, indices_tensor, :, :]
    elif latents.dim() == 4:
        # (B, F, H, W) — unlikely but handle gracefully
        B, F, H, W = latents.shape
        indices = list(range(0, F, step_K))
        if indices[-1] != F - 1:
            indices.append(F - 1)
        indices_tensor = torch.tensor(indices, device=latents.device)
        return latents[:, indices_tensor, :, :]
    else:
        return latents


# ---------------------------------------------------------------------------
# IF-Edit Pipeline
# ---------------------------------------------------------------------------

class IFEditPipeline:
    """IF-Edit: Zero-shot image editing via I2V diffusion models.

    This pipeline wraps a Wan 2.1 I2V model and implements:
    - CoT temporal prompt enhancement
    - Temporal Latent Dropout (TLD) during denoising
    - Self-Consistent Post-Refinement (SCPR) for output clarity

    Usage:
        pipe = IFEditPipeline(model_path="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
        result = pipe.edit(image, "add sunglasses to the person")
    """

    def __init__(
        self,
        device: str = "cuda",
        model_path: str = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        model_cache_dir: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        enable_model_cpu_offload: bool = True,
        prompt_enhancer: Optional[IFEditPromptEnhancer] = None,
        # TLD parameters
        tld_enabled: bool = True,
        tld_threshold_ratio: float = 0.4,
        tld_step_K: int = 3,
        # SCPR parameters
        scpr_enabled: bool = True,
        scpr_refinement_steps: int = 30,
    ):
        self.device = device
        self.model_path = model_path
        self.model_cache_dir = model_cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface"
        )

        # TLD config
        self.tld_enabled = tld_enabled
        self.tld_threshold_ratio = tld_threshold_ratio  # fraction of total steps
        self.tld_step_K = tld_step_K

        # SCPR config
        self.scpr_enabled = scpr_enabled
        self.scpr_refinement_steps = scpr_refinement_steps

        # Prompt enhancer
        if prompt_enhancer is not None:
            self.prompt_enhancer = prompt_enhancer
        else:
            self.prompt_enhancer = IFEditPromptEnhancer()

        # Determine dtype
        if torch_dtype is not None:
            self._dtype = torch_dtype
        elif os.environ.get("IFEDIT_FORCE_FP32") == "1":
            self._dtype = torch.float32
        else:
            self._dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Load the Wan I2V pipeline
        print(f"[IF-Edit] Loading Wan I2V model from {model_path} (dtype={self._dtype})...")
        start = time.time()

        self.pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=self._dtype,
            cache_dir=self.model_cache_dir,
        )

        if enable_model_cpu_offload:
            self.pipe.enable_model_cpu_offload()
            if hasattr(self.pipe, "vae"):
                self.pipe.vae.enable_slicing()
                self.pipe.vae.enable_tiling()
            print("[IF-Edit] Model CPU offload + VAE optimizations enabled.")
        else:
            self.pipe = self.pipe.to(device)

        print(f"[IF-Edit] Model loaded in {time.time() - start:.1f}s")

    # ------------------------------------------------------------------
    # CoT Prompt Enhancement
    # ------------------------------------------------------------------
    def enhance_prompt(self, image: Image.Image, edit_instruction: str) -> str:
        """Generate a CoT temporal prompt describing how the scene evolves."""
        return self.prompt_enhancer.enhance(image, edit_instruction)

    # ------------------------------------------------------------------
    # Core: Generate video with optional TLD
    # ------------------------------------------------------------------
    def generate_video(
        self,
        image: Image.Image,
        prompt: str,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
    ) -> List[Image.Image]:
        """Generate video frames from image + prompt using Wan I2V.

        When TLD is enabled, this modifies the denoising loop to drop
        intermediate frame latents after the threshold step.
        """
        if image is None:
            raise ValueError("Input image cannot be None.")
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize image to match target resolution
        image = image.resize((width, height), Image.Resampling.LANCZOS)

        print(f"[IF-Edit] Generating video: {num_frames} frames, {width}x{height}")
        print(f"[IF-Edit] Prompt: {prompt[:100]}...")

        if self.tld_enabled:
            frames = self._generate_with_tld(
                image=image,
                prompt=prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            )
        else:
            frames = self._generate_standard(
                image=image,
                prompt=prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            )

        print(f"[IF-Edit] Generated {len(frames)} frames")
        return frames

    def _generate_standard(
        self,
        image: Image.Image,
        prompt: str,
        num_frames: int,
        height: int,
        width: int,
        guidance_scale: float,
        num_inference_steps: int,
        generator: Optional[torch.Generator],
    ) -> List[Image.Image]:
        """Standard generation without TLD — use pipeline as black box."""
        result = self.pipe(
            image=image,
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        return self._extract_frames(result)

    def _generate_with_tld(
        self,
        image: Image.Image,
        prompt: str,
        num_frames: int,
        height: int,
        width: int,
        guidance_scale: float,
        num_inference_steps: int,
        generator: Optional[torch.Generator],
    ) -> List[Image.Image]:
        """Generate with Temporal Latent Dropout.

        Strategy: We use a callback mechanism to intercept the denoising loop.
        After the threshold step T_th, we sub-sample frame latents at stride K.

        For Wan pipeline (diffusers), we use `callback_on_step_end` to modify
        latents in-place during denoising.
        """
        T_th = int(num_inference_steps * self.tld_threshold_ratio)
        step_K = self.tld_step_K
        tld_applied = [False]  # mutable flag
        original_num_frames = num_frames

        def tld_callback(pipe, step_index, timestep, callback_kwargs):
            """Callback to apply TLD at the threshold step."""
            if step_index >= T_th and not tld_applied[0]:
                latents = callback_kwargs.get("latents")
                if latents is not None:
                    original_shape = latents.shape
                    latents = temporal_latent_dropout(latents, step_K=step_K)
                    callback_kwargs["latents"] = latents
                    tld_applied[0] = True
                    print(f"[TLD] Applied at step {step_index}/{num_inference_steps}: "
                          f"{original_shape} -> {latents.shape}")
            return callback_kwargs

        # Try with callback first, fall back to standard if not supported
        try:
            result = self.pipe(
                image=image,
                prompt=prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                callback_on_step_end=tld_callback,
                callback_on_step_end_tensor_inputs=["latents"],
            )
        except TypeError as e:
            if "callback" in str(e).lower() or "unexpected keyword" in str(e).lower():
                print(f"[TLD] Callback not supported by this pipeline version, "
                      f"falling back to manual TLD.")
                result = self._generate_with_manual_tld(
                    image=image,
                    prompt=prompt,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                )
                return self._extract_frames(result)
            raise

        return self._extract_frames(result)

    def _generate_with_manual_tld(
        self,
        image: Image.Image,
        prompt: str,
        num_frames: int,
        height: int,
        width: int,
        guidance_scale: float,
        num_inference_steps: int,
        generator: Optional[torch.Generator],
    ) -> Any:
        """Manual TLD implementation for pipelines without callback support.

        We manually run the denoising loop:
        1. Encode prompt & image
        2. Prepare initial latents
        3. Run denoising steps, applying TLD at threshold
        4. Decode latents to frames
        """
        pipe = self.pipe
        T_th = int(num_inference_steps * self.tld_threshold_ratio)
        step_K = self.tld_step_K

        # --- Encode text ---
        if hasattr(pipe, "encode_prompt"):
            prompt_embeds = pipe.encode_prompt(
                prompt=prompt,
                num_videos_per_prompt=1,
                do_classifier_free_guidance=(guidance_scale > 1.0),
            )
            if isinstance(prompt_embeds, tuple):
                if len(prompt_embeds) >= 2:
                    prompt_embeds, negative_prompt_embeds = prompt_embeds[0], prompt_embeds[1]
                else:
                    prompt_embeds = prompt_embeds[0]
                    negative_prompt_embeds = None
            else:
                negative_prompt_embeds = None
        else:
            raise RuntimeError("Pipeline does not have encode_prompt method.")

        # --- Encode image ---
        if hasattr(pipe, "encode_image"):
            image_embeds = pipe.encode_image(image, device=pipe._execution_device)
        elif hasattr(pipe, "image_encoder"):
            from torchvision import transforms as T
            img_tensor = T.ToTensor()(image).unsqueeze(0).to(
                device=pipe._execution_device, dtype=self._dtype
            )
            image_embeds = pipe.image_encoder(img_tensor)
        else:
            image_embeds = None

        # --- Prepare latents ---
        pipe.scheduler.set_timesteps(num_inference_steps, device=pipe._execution_device)
        timesteps = pipe.scheduler.timesteps

        # Determine latent shape
        vae_scale_factor_spatial = getattr(pipe, "vae_scale_factor_spatial", 8)
        vae_scale_factor_temporal = getattr(pipe, "vae_scale_factor_temporal", 4)
        latent_num_frames = (num_frames - 1) // vae_scale_factor_temporal + 1
        latent_height = height // vae_scale_factor_spatial
        latent_width = width // vae_scale_factor_spatial
        num_channels = pipe.unet.config.in_channels if hasattr(pipe, "unet") else (
            pipe.transformer.config.in_channels if hasattr(pipe, "transformer") else 16
        )

        latent_shape = (1, num_channels, latent_num_frames, latent_height, latent_width)
        latents = torch.randn(
            latent_shape, generator=generator,
            device=pipe._execution_device, dtype=self._dtype,
        )

        # --- Denoising loop with TLD ---
        tld_applied = False
        for i, t in enumerate(timesteps):
            # Apply TLD at threshold
            if i >= T_th and not tld_applied:
                original_shape = latents.shape
                latents = temporal_latent_dropout(latents, step_K=step_K)
                tld_applied = True
                print(f"[TLD-Manual] Applied at step {i}/{num_inference_steps}: "
                      f"{original_shape} -> {latents.shape}")

            # Expand latents for classifier-free guidance
            latent_model_input = latents
            if guidance_scale > 1.0:
                latent_model_input = torch.cat([latents] * 2)

            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            # Predict noise
            model = getattr(pipe, "unet", None) or getattr(pipe, "transformer", None)
            model_kwargs = {
                "sample": latent_model_input,
                "timestep": t,
                "encoder_hidden_states": (
                    torch.cat([negative_prompt_embeds, prompt_embeds])
                    if guidance_scale > 1.0 and negative_prompt_embeds is not None
                    else prompt_embeds
                ),
            }
            if image_embeds is not None:
                model_kwargs["added_cond_kwargs"] = {"image_embeds": image_embeds}

            noise_pred = model(**model_kwargs)
            if hasattr(noise_pred, "sample"):
                noise_pred = noise_pred.sample

            # Classifier-free guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # Scheduler step
            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # --- Decode latents ---
        if hasattr(pipe, "decode_latents"):
            frames_tensor = pipe.decode_latents(latents)
        elif hasattr(pipe, "vae"):
            latents = latents / pipe.vae.config.scaling_factor
            frames_tensor = pipe.vae.decode(latents).sample
        else:
            raise RuntimeError("Cannot decode latents: no decode_latents or vae found.")

        return frames_tensor

    def _extract_frames(self, result: Any) -> List[Image.Image]:
        """Extract PIL frames from pipeline output."""
        frames: List[Image.Image] = []

        if isinstance(result, list):
            for item in result:
                if isinstance(item, Image.Image):
                    frames.append(item)
                elif isinstance(item, list):
                    frames.extend(f for f in item if isinstance(f, Image.Image))
            if frames:
                return frames

        # Try standard attributes
        for attr in ("frames", "videos", "images"):
            obj = getattr(result, attr, None)
            if obj is None:
                continue
            extracted = self._tensor_or_list_to_frames(obj)
            if extracted:
                return extracted

        # Try direct tensor
        if torch.is_tensor(result):
            return self._tensor_to_frames(result)

        raise RuntimeError(f"Could not extract frames from pipeline output: {type(result)}")

    def _tensor_or_list_to_frames(self, obj: Any) -> List[Image.Image]:
        """Convert a nested list or tensor to PIL frames."""
        if isinstance(obj, list):
            merged = []
            for item in obj:
                if isinstance(item, Image.Image):
                    merged.append(item)
                elif isinstance(item, list):
                    merged.extend(f for f in item if isinstance(f, Image.Image))
                elif torch.is_tensor(item):
                    merged.extend(self._tensor_to_frames(item))
            return merged
        if torch.is_tensor(obj):
            return self._tensor_to_frames(obj)
        return []

    @staticmethod
    def _tensor_to_frames(tensor: torch.Tensor) -> List[Image.Image]:
        """Convert a video tensor to list of PIL Images."""
        t = tensor.detach().float().cpu()
        if t.dim() == 5:
            t = t[0]  # remove batch dim -> (C, F, H, W) or (F, C, H, W)
        if t.dim() == 4:
            # Determine if (C, F, H, W) or (F, C, H, W)
            if t.shape[0] <= 4:
                # (C, F, H, W) -> (F, C, H, W)
                t = t.permute(1, 0, 2, 3)
            frames = []
            for frame in t:
                frame = frame.clamp(0, 1)
                arr = (frame.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                frames.append(Image.fromarray(arr))
            return frames
        return []

    # ------------------------------------------------------------------
    # Self-Consistent Post-Refinement (SCPR)
    # ------------------------------------------------------------------
    def self_consistent_refinement(
        self,
        frames: List[Image.Image],
        num_refinement_steps: Optional[int] = None,
        guidance_scale: float = 5.0,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """SCPR: Select sharpest frame, then refine via still-video generation.

        Steps:
        1. Compute Laplacian sharpness for all frames
        2. Select the sharpest frame x*
        3. Re-inject x* into the I2V model with a "still video" prompt
        4. Select the sharpest frame from the refinement output

        Returns:
            (refined_image, info_dict)
        """
        if not frames:
            raise ValueError("No frames to refine.")

        steps = num_refinement_steps or self.scpr_refinement_steps

        # Step 1 & 2: Select sharpest frame
        best_idx, best_frame, sharpness_scores = select_sharpest_frame(frames)
        print(f"[SCPR] Selected frame {best_idx} with sharpness {sharpness_scores[best_idx]:.2f}")
        print(f"[SCPR] Sharpness scores (sample): {[f'{s:.2f}' for s in sharpness_scores[:10]]}")

        # Step 3: Still-video refinement
        still_prompt = (
            "A perfectly still, high-resolution photograph with exceptional clarity "
            "and fine detail. The image remains completely static with no motion, "
            "camera shake, or blur. Ultra-sharp focus throughout the entire frame."
        )

        h, w = best_frame.size[1], best_frame.size[0]
        refinement_frames = self.generate_video(
            image=best_frame,
            prompt=still_prompt,
            num_frames=17,  # Fewer frames for refinement
            height=h,
            width=w,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator,
        )

        # Step 4: Select sharpest from refinement
        if refinement_frames:
            ref_idx, refined_image, ref_scores = select_sharpest_frame(refinement_frames)
            print(f"[SCPR] Refinement: selected frame {ref_idx} "
                  f"(sharpness {ref_scores[ref_idx]:.2f})")
        else:
            refined_image = best_frame
            ref_scores = sharpness_scores

        info = {
            "initial_sharpness_scores": sharpness_scores,
            "initial_best_idx": best_idx,
            "initial_best_sharpness": sharpness_scores[best_idx],
            "refinement_sharpness_scores": ref_scores if refinement_frames else [],
            "refinement_best_idx": ref_idx if refinement_frames else -1,
            "refinement_best_sharpness": (
                ref_scores[ref_idx] if refinement_frames else sharpness_scores[best_idx]
            ),
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
        height: int = 480,
        width: int = 832,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        save_dir: str = "outputs",
        use_cot_prompt: bool = True,
        use_tld: Optional[bool] = None,
        use_scpr: Optional[bool] = None,
    ) -> Tuple[Image.Image, List[Image.Image], Dict[str, Any]]:
        """Run the full IF-Edit pipeline.

        Args:
            image: Source image to edit
            edit_instruction: Text describing the desired edit
            num_frames: Number of video frames to generate
            height, width: Output resolution
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
            generator: Optional torch generator for reproducibility
            save_dir: Directory to save intermediate results
            use_cot_prompt: Whether to use CoT prompt enhancement
            use_tld: Override TLD setting (None = use self.tld_enabled)
            use_scpr: Override SCPR setting (None = use self.scpr_enabled)

        Returns:
            (final_image, all_frames, run_info)
        """
        os.makedirs(save_dir, exist_ok=True)
        run_info: Dict[str, Any] = {
            "edit_instruction": edit_instruction,
            "timestamp": time.time(),
        }

        # --- Step 1: CoT Prompt Enhancement ---
        if use_cot_prompt:
            temporal_prompt = self.enhance_prompt(image, edit_instruction)
            run_info["temporal_prompt"] = temporal_prompt
            print(f"[IF-Edit] CoT prompt: {temporal_prompt[:120]}...")
        else:
            temporal_prompt = edit_instruction
            run_info["temporal_prompt"] = temporal_prompt

        # --- Step 2: Video Generation (with optional TLD) ---
        tld_was_enabled = self.tld_enabled
        if use_tld is not None:
            self.tld_enabled = use_tld

        try:
            frames = self.generate_video(
                image=image,
                prompt=temporal_prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            )
        finally:
            self.tld_enabled = tld_was_enabled

        run_info["num_frames_generated"] = len(frames)
        run_info["tld_enabled"] = use_tld if use_tld is not None else tld_was_enabled
        run_info["tld_step_K"] = self.tld_step_K
        run_info["tld_threshold_ratio"] = self.tld_threshold_ratio

        # Save video
        if frames:
            video_path = os.path.join(save_dir, f"generated_{int(time.time())}.mp4")
            self._save_video(frames, video_path)
            run_info["video_path"] = video_path

        # --- Step 3: Frame Selection + SCPR ---
        scpr_active = use_scpr if use_scpr is not None else self.scpr_enabled

        if scpr_active and frames:
            final_image, scpr_info = self.self_consistent_refinement(
                frames=frames,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            run_info["scpr"] = scpr_info
            run_info["selection_method"] = "scpr"
        elif frames:
            # Fallback: just pick sharpest frame without refinement
            best_idx, final_image, scores = select_sharpest_frame(frames)
            run_info["sharpness_scores"] = scores
            run_info["selected_frame_idx"] = best_idx
            run_info["selection_method"] = "laplacian"
        else:
            raise RuntimeError("No frames generated.")

        # Save final result
        final_path = os.path.join(save_dir, "final_edit.png")
        final_image.save(final_path)
        run_info["final_image_path"] = final_path

        # Save 512x512 version
        try:
            img_512 = postprocess_to_512(final_image)
            path_512 = os.path.join(save_dir, "final_edit_512.png")
            img_512.save(path_512)
            run_info["final_image_512_path"] = path_512
        except Exception as e:
            print(f"[WARN] Could not save 512 version: {e}")

        return final_image, frames, run_info

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _save_video(frames: List[Image.Image], save_path: str, fps: int = 8):
        """Save frames as MP4 video."""
        if not frames:
            return
        try:
            import cv2
        except ImportError:
            print("[WARN] opencv-python not installed, cannot save video.")
            return

        w, h = frames[0].size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            print(f"[WARN] Cannot open video writer: {save_path}")
            return

        for frame in frames:
            arr = np.array(frame.convert("RGB"))
            writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"[IF-Edit] Saved video: {save_path} ({len(frames)} frames, {fps} fps)")


# For backward compatibility
Frame2FramePipeline = IFEditPipeline
