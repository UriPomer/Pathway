# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as _F
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image as PILImage
from tqdm import tqdm

_FRAME_GUIDANCE_DIR = Path(__file__).resolve().parents[2] / "frame-guidance"
if str(_FRAME_GUIDANCE_DIR) not in sys.path:
    sys.path.insert(0, str(_FRAME_GUIDANCE_DIR))

from pipelines.utils.models import setup_csd

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


def _make_tld_indices(frame_count, step_k, device):
    if frame_count <= 1:
        return torch.arange(frame_count, device=device)
    indices = list(range(0, frame_count, step_k))
    if indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    return torch.tensor(indices, device=device, dtype=torch.long)


def _apply_tld(tensor, indices):
    return tensor.index_select(1, indices)


def _reset_scheduler_state(scheduler):
    if hasattr(scheduler, "model_outputs"):
        scheduler.model_outputs = [None] * len(scheduler.model_outputs)
    if hasattr(scheduler, "timestep_list"):
        scheduler.timestep_list = [None] * len(scheduler.timestep_list)
    if hasattr(scheduler, "lower_order_nums"):
        scheduler.lower_order_nums = 0
    if hasattr(scheduler, "last_sample"):
        scheduler.last_sample = None


def _downscale_spatial(x, factor):
    """Downsample only the spatial (H, W) dims of a 4D [C,T,H,W] or 5D [B,C,T,H,W] tensor."""
    has_batch = (x.ndim == 5)
    if not has_batch:
        x = x.unsqueeze(0)                                          # [1, C, T, H, W]
    B, C, T, H, W = x.shape
    x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)           # [B*T, C, H, W]
    x = _F.interpolate(x, scale_factor=1 / factor, mode='bilinear', align_corners=False)
    x = x.reshape(B, T, C, x.shape[-2], x.shape[-1]).permute(0, 2, 1, 3, 4)
    if not has_batch:
        x = x.squeeze(0)                                            # back to [C, T, h, w]
    return x


def _temporal_roll(x, shift_idx, reverse=False):
    """Roll temporal axis (dim=1) across all frames, including the first frame."""
    t = int(x.shape[1])
    if shift_idx <= 0 or t <= 1:
        return x
    shift_idx = shift_idx % t
    if shift_idx == 0:
        return x
    if reverse:
        shift_idx = t - shift_idx
        if shift_idx == 0:
            return x
    return torch.cat([x[:, shift_idx:, ...], x[:, :shift_idx, ...]], dim=1)


def get_sigma_at_step(scheduler, step_idx):
    """Extract sigma value at a given step index from flow matching scheduler."""
    sigmas = getattr(scheduler, 'sigmas', None)
    if sigmas is not None and step_idx < len(sigmas):
        return sigmas[step_idx].to(scheduler.timesteps.device)
    if step_idx >= len(scheduler.timesteps):
        return torch.tensor(0.0)
    idx = scheduler.timesteps[step_idx].item()
    sigma = idx / 1000.0
    return torch.tensor(sigma)


class DifferentiableAugmenter(nn.Module):
    """Differentiable image-to-observation operator for scribble loss.

    Mirrors ``frame-guidance/pipelines/utils/models.py::DifferentiableAugmenter``.
    """

    def __init__(self):
        super().__init__()
        rgb2gray = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        self.register_buffer("rgb2gray", rgb2gray)
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]])
        sobel_y = sobel_x.t()
        sobel = torch.stack([sobel_x, sobel_y])  # 2x3x3
        self.register_buffer("sobel", sobel.unsqueeze(1))  # 2x1x3x3

    def forward(self, x: torch.Tensor, mode: str = "scribble") -> torch.Tensor:
        if mode == "scribble":
            g = (x * self.rgb2gray).sum(1, keepdim=True)
            gxgy = _F.conv2d(g, self.sobel, padding=1)
            grad_mag = torch.sqrt(gxgy.square().sum(1, keepdim=True) + 1e-6)
            scribble = torch.tanh(2.5 * grad_mag)
            return scribble.repeat(1, 3, 1, 1)
        else:
            raise ValueError(f"Unknown mode: {mode}")


def _weighted_mse_loss(pred, target, weight):
    return ((weight * (pred - target) ** 2).mean())


def _load_fg_image(path: str, device: torch.device, target_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """Load an image file to a [1, 3, 1, H, W] float tensor in [-1, 1]."""
    img = PILImage.open(path).convert("RGB")
    if target_hw is not None:
        img = img.resize((target_hw[1], target_hw[0]), PILImage.BILINEAR)
    arr = T.ToTensor()(img) * 2.0 - 1.0  # [3, H, W] in [-1, 1]
    return arr.unsqueeze(0).unsqueeze(2).to(device)  # [1, 3, 1, H, W]


def _load_fg_video(video_path: str, device: torch.device, target_hw: Optional[Tuple[int, int]] = None) -> Optional[torch.Tensor]:
    """Load a video file into a [1, 3, T, H, W] float tensor in [-1, 1]."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return None
    arr = np.stack(frames, axis=0)  # [T, H, W, 3]
    tensor = torch.from_numpy(arr).float() / 127.5 - 1.0
    tensor = tensor.permute(0, 3, 1, 2).unsqueeze(0)
    if target_hw is not None:
        # Resize: [1, 3, T, H, W]
        b, c, t, oh, ow = tensor.shape
        tensor = tensor.permute(0, 2, 1, 3, 4).reshape(b * t, c, oh, ow)
        tensor = TF.resize(tensor, target_hw, antialias=True)
        nh, nw = tensor.shape[-2], tensor.shape[-1]
        tensor = tensor.reshape(b, t, c, nh, nw).permute(0, 2, 1, 3, 4)
    return tensor.to(device)


FG_VALID_LOSS_FNS = {"style", "scribble", "loop"}


class WanI2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu
        # Track which noise model is currently "active" on GPU to avoid per-step device scans.
        # Possible values: "low", "high", or None (unknown/uninitialized).
        self._active_noise_model: Optional[str] = None

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.low_noise_model = WanModel.from_pretrained(
            checkpoint_dir, subfolder=config.low_noise_checkpoint)
        self.low_noise_model = self._configure_model(
            model=self.low_noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)

        self.high_noise_model = WanModel.from_pretrained(
            checkpoint_dir, subfolder=config.high_noise_checkpoint)
        self.high_noise_model = self._configure_model(
            model=self.high_noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)
        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def _prepare_model_for_timestep(self, t, boundary, offload_model):
        r"""
        Prepares and returns the required model for the current timestep.

        Args:
            t (torch.Tensor):
                current timestep.
            boundary (`int`):
                The timestep threshold. If `t` is at or above this value,
                the `high_noise_model` is considered as the required model.
            offload_model (`bool`):
                A flag intended to control the offloading behavior.

        Returns:
            torch.nn.Module:
                The active model on the target device for the current timestep.
        """
        if t.item() >= boundary:
            required_name = "high"
            required_model = self.high_noise_model
            inactive_name = "low"
            inactive_model = self.low_noise_model
        else:
            required_name = "low"
            required_model = self.low_noise_model
            inactive_name = "high"
            inactive_model = self.high_noise_model

        if not (offload_model or self.init_on_cpu):
            return required_model

        # Fundamental invariant:
        # - The active model must be fully on GPU before any forward/backward.
        # - The inactive model should be fully on CPU when offloading is enabled.
        #
        # Do the expensive "whole-module to(device)" only when the active model changes.
        if self._active_noise_model != required_name:
            required_model.to(device=self.device, dtype=self.param_dtype)
            if (offload_model or self.init_on_cpu) and next(inactive_model.parameters()).device.type == "cuda":
                inactive_model.to("cpu")
            torch.cuda.empty_cache()
            self._active_noise_model = required_name
        else:
            # If offload_model is enabled, keep the inactive model off GPU.
            if (offload_model or self.init_on_cpu) and next(inactive_model.parameters()).device.type == "cuda":
                inactive_model.to("cpu")
                torch.cuda.empty_cache()
            # Ensure active model isn't accidentally left on CPU (e.g. after external offload).
            if next(required_model.parameters()).device.type == "cpu":
                required_model.to(device=self.device, dtype=self.param_dtype)
                torch.cuda.empty_cache()

        return required_model

    def _prepare_fg_loss_context(self, fg_loss_fn: str, fg_additional_inputs: Optional[Dict],
                                  fg_video: Optional[torch.Tensor],
                                  target_h: int, target_w: int):
        """Prepare pre-computed context for the chosen FG loss function.

        Supports: style, scribble, loop.
        Mirrors ``frame-guidance/pipelines/pipeline_wan_i2v.py::_prepare_loss_context``.
        """
        if fg_additional_inputs is None:
            fg_additional_inputs = {}
        ctx: Dict[str, Any] = {}
        device = self.device
        dtype = self.param_dtype

        if fg_loss_fn == "style":
            style_image = fg_additional_inputs.get("style_image")
            assert style_image is not None, "style loss requires 'style_image' in fg_additional_inputs"
            if not hasattr(self, '_fg_csd') or self._fg_csd is None:
                self._fg_csd, self._fg_preprocess = setup_csd(device=device)
                # Cache CLIP-style preprocess for tensor inputs (no ToTensor)
                import torchvision
                self._fg_clip_preprocess_tensor = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(224,
                                                  interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                    torchvision.transforms.CenterCrop(224),
                    torchvision.transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711)),
                ])
                # Offload CSD to CPU after setup (will be moved to GPU on-demand during loss computation)
                if self.fg_offload_model:
                    self._fg_csd.cpu()
                    torch.cuda.empty_cache()
            ctx["cosine_loss"] = nn.CosineSimilarity(dim=1)
            ctx["style_image"] = style_image
            ctx["style_weight"] = fg_additional_inputs.get("style_weight", 1.0)
            ctx["content_image"] = fg_additional_inputs.get("content_image", None)
            ctx["content_weight"] = fg_additional_inputs.get("content_weight", 0.0)

        elif fg_loss_fn == "scribble":
            scribble_path = fg_additional_inputs.get("scribble_image_path")
            assert scribble_path is not None, "scribble loss requires 'scribble_image_path' in fg_additional_inputs"
            if not hasattr(self, '_fg_aug') or self._fg_aug is None:
                self._fg_aug = DifferentiableAugmenter().to(device).requires_grad_(False)
            # Load single scribble image → [1, 3, 1, H, W] in [-1,1]
            scribble_tensor = _load_fg_image(scribble_path, device, target_hw=(target_h, target_w))
            # Convert to [0,1] and extract scribble edge map
            frame_01 = (scribble_tensor[:, :, 0] + 1.0) / 2.0  # [1, 3, H, W]
            with torch.no_grad():
                scribble_cond = self._fg_aug(frame_01.to(dtype), mode="scribble")
                if self.fg_downscale_factor > 1:
                    scribble_cond = _F.interpolate(scribble_cond, scale_factor=1 / self.fg_downscale_factor,
                                                    mode='bilinear', align_corners=False)
            # Store as single tensor (will be indexed by fixed_frame index)
            ctx["scribble_cond"] = scribble_cond  # [1, 3, H, W] — keep batch dim for MSE with pred_f

        elif fg_loss_fn == "loop":
            pass  # No special context needed

        return ctx

    def _offload_fg_models_after_backward(self, fg_loss_fn: str):
        """Offload FG auxiliary models (VAE, CSD) to CPU after backward.

        Must be called AFTER loss.backward() completes, never before —
        the backward pass traverses the computation graph which includes
        these models, so their parameters must remain on GPU until then.
        """
        if not self.fg_offload_model:
            return
        # VAE is used in all loss types (loop, style, scribble)
        self.vae.model.cpu()
        # CSD is used only in style loss
        if fg_loss_fn == "style" and hasattr(self, '_fg_csd'):
            self._fg_csd.cpu()
        torch.cuda.empty_cache()

    def _compute_fg_loss(self, fg_loss_fn: str, pred_x0: torch.Tensor,
                          ff: int, fg_fixed_frames: list,
                          fg_loss_ctx: Dict, fg_additional_inputs: Optional[Dict],
                          frame_num: int, step_idx: int, rep_idx: int):
        """Compute FG loss for a single fixed frame or loop loss.

        Supports: style, scribble, loop.
        Mirrors ``frame-guidance/pipelines/pipeline_wan_i2v.py::_compute_single_frame_loss``
        and ``pipeline_wan.py::_compute_loop_loss``.

        NOTE: Model offloading (VAE, CSD to CPU) is NOT done here.
        The returned loss tensor's computation graph still references these
        models, so they must stay on GPU until after loss.backward().
        Call _offload_fg_models_after_backward() after backward completes.
        """
        offload_model = self.fg_offload_model

        if fg_loss_fn == "loop":
            # Decode first and last frame
            predicted_images = []
            for fixed_frame in [0, frame_num - 1]:
                if fixed_frame == 0:
                    pair = pred_x0[:, 0:1]          # [C, 1, H, W]
                    rel = 0
                else:
                    lf = (fixed_frame - 1) // 4 + 1
                    pair = pred_x0[:, lf - 1:lf + 1] # [C, 2, H, W]
                    rel = (fixed_frame - 1) % 4 + 1
                if self.fg_downscale_factor > 1:
                    pair = _downscale_spatial(pair, self.fg_downscale_factor)
                if offload_model:
                    self.vae.model.to(self.device)
                with torch.amp.autocast('cuda', enabled=False):
                    decoded = self.vae.model.decode(
                        pair.float().unsqueeze(0), self.vae.scale
                    ).float().clamp(-1, 1).squeeze(0)  # [3, T_out, H_out, W_out]
                pred_f = decoded[:, rel:rel + 1]
                predicted_images.append(pred_f)
                del pair, decoded
            loss = _F.mse_loss(predicted_images[0], predicted_images[1])
            del predicted_images
            return loss

        # Single frame losses (style / scribble)
        if ff <= 0:
            return None

        lf = (ff - 1) // 4 + 1
        pair = pred_x0[:, lf - 1:lf + 1]             # [C, 2, H, W]
        if self.fg_downscale_factor > 1:
            pair = _downscale_spatial(pair, self.fg_downscale_factor)
        if offload_model:
            self.vae.model.to(self.device)
        # Bypass Wan2_1_VAE.decode() wrapper to avoid:
        #   1. Internal amp.autocast that may interfere with gradient flow
        #   2. In-place .clamp_(-1, 1) that corrupts the autograd graph
        # Instead, call the raw WanVAE_.decode() directly with float32.
        # This matches the official pipeline's decode_latents() which has
        # no autocast wrapper and no clamping.
        with torch.amp.autocast('cuda', enabled=False):
            pair_f32 = pair.float()
            decoded = self.vae.model.decode(
                pair_f32.unsqueeze(0), self.vae.scale
            ).squeeze(0)  # [3, T_out, H_out, W_out]
            # No hard clamp — official decode_latents() has none.
            # Hard clamp zeros gradients at boundaries. The downstream
            # clip_preprocess Normalize handles arbitrary ranges fine.
            decoded = decoded.float()
        rel = (ff - 1) % 4 + 1
        pred_f = decoded[:, rel:rel + 1]
        del pair

        if fg_loss_fn == "style":
            pred_f_sq = pred_f[:, 0].unsqueeze(0)  # [3,1,H,W] → [3,H,W] → [1,3,H,W]

            # Ensure CSD is on device
            if offload_model and hasattr(self, '_fg_csd'):
                self._fg_csd.to(self.device)
            # Disable autocast: CSD (CLIP ViT-L) needs float32 for stable
            # gradients. The outer generate() runs under bfloat16 autocast,
            # which kills small gradient signals in the style embedding space.
            with torch.amp.autocast('cuda', enabled=False):
                pred_f_input = self._fg_clip_preprocess_tensor(pred_f_sq.float())
                _, content_emb, style_emb = self._fg_csd(pred_f_input)

            # Reference embeddings only — no grad needed for constants
            with torch.no_grad():
                if "style_embed" not in fg_additional_inputs:
                    with torch.amp.autocast('cuda', enabled=False):
                        _, _, style_emb_ref = self._fg_csd(
                            self._fg_preprocess(fg_loss_ctx["style_image"]).unsqueeze(0).to(self.device))
                    fg_additional_inputs["style_embed"] = style_emb_ref
                style_emb_ref = fg_additional_inputs["style_embed"].to(self.device)
                if fg_loss_ctx.get("content_image") is not None and fg_loss_ctx.get("content_weight", 0) > 0:
                    if "content_embed" not in fg_additional_inputs:
                        with torch.amp.autocast('cuda', enabled=False):
                            _, content_emb_ref, _ = self._fg_csd(
                                self._fg_preprocess(fg_loss_ctx["content_image"]).unsqueeze(0).to(self.device))
                        fg_additional_inputs["content_embed"] = content_emb_ref

            # Loss computation MUST be OUTSIDE torch.no_grad() so that
            # cos() and (1 - style_sim) create grad_fn nodes, allowing
            # loss.backward() to propagate gradients back through CSD →
            # clip_preprocess → VAE decode → pred_x0 → current_latent.
            # (Previously these were inside no_grad, killing the gradient chain.)
            cos = fg_loss_ctx["cosine_loss"]
            style_sim = cos(style_emb, style_emb_ref)
            print(f"    style_similarity: {style_sim.item():.3f}", flush=True)
            loss = fg_loss_ctx["style_weight"] * (1 - style_sim)
            if fg_loss_ctx.get("content_image") is not None and fg_loss_ctx.get("content_weight", 0) > 0:
                content_emb_ref = fg_additional_inputs["content_embed"].to(self.device)
                content_sim = cos(content_emb, content_emb_ref)
                loss = loss + fg_loss_ctx["content_weight"] * (1 - content_sim)

        elif fg_loss_fn == "scribble":
            sketch_obs = self._fg_aug(pred_f[:, 0].unsqueeze(0).float(), mode="scribble")  # [1,3,H,W]
            loss = _F.mse_loss(sketch_obs, fg_loss_ctx["scribble_cond"].float())

        del decoded, pred_f
        return loss

    def generate(self,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 ifedit_use_tld=False,
                 ifedit_tld_threshold_ratio=0.5,
                 ifedit_tld_step_k=2,
                 loopless_enable=False,
                 loop_shift_skip=6,
                 loop_shift_stop_step=4,
                 fg_enable=False,
                 fg_fixed_frames=None,
                 fg_guidance_lr=10.0,
                 fg_travel_time=(3, 10),
                 fg_downscale_factor=4,
                 fg_loss_fn="style",
                 fg_additional_inputs=None):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float` or tuple[`float`], *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
                If tuple, the first guide_scale will be used for low noise model and
                the second guide_scale will be used for high noise model.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            ifedit_use_tld (`bool`, *optional*, defaults to False):
                Enable Temporal Latent Dropout (IF-Edit).
            ifedit_tld_threshold_ratio (`float`, *optional*, defaults to 0.5):
                Apply TLD once when step index reaches `sampling_steps * ratio`.
            ifedit_tld_step_k (`int`, *optional*, defaults to 2):
                Temporal sub-sampling stride for TLD.

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        # preprocess
        guide_scale = (guide_scale, guide_scale) if isinstance(
            guide_scale, float) else guide_scale

        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16,
            (F - 1) // self.vae_stride[0] + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, F - 1, h, w)
            ],
                         dim=1).to(self.device)
        ])[0]
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low_noise = getattr(self.low_noise_model, 'no_sync',
                                    noop_no_sync)
        no_sync_high_noise = getattr(self.high_noise_model, 'no_sync',
                                     noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_low_noise(),
                no_sync_high_noise(),
        ):
            boundary = self.boundary * self.num_train_timesteps

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise

            cond_context = [context[0]]
            uncond_context = context_null
            cond_y = [y]
            uncond_y = [y]

            if offload_model:
                torch.cuda.empty_cache()

            total_steps = len(timesteps)
            tld_applied = False
            threshold_timestep = ifedit_tld_threshold_ratio * self.num_train_timesteps

            shift_idx = 0
            latent_length = latent.shape[1]
            shift_skip = max(0, int(loop_shift_skip))
            use_shift_threshold = total_steps - int(loop_shift_stop_step) if int(loop_shift_stop_step) > 0 else total_steps

            # --- Frame Guidance setup ---
            self.fg_downscale_factor = fg_downscale_factor
            self.fg_offload_model = offload_model

            # Enable gradient checkpointing for FG to avoid OOM
            # (paper's notebook: pipe.transformer.enable_gradient_checkpointing())
            # This trades compute for memory by recomputing activations during backward.
            _fg_grad_ckpt_enabled = False
            if fg_enable:
                for m in [self.high_noise_model, self.low_noise_model]:
                    if m is not None:
                        m.gradient_checkpointing = True
                        _fg_grad_ckpt_enabled = True
                if _fg_grad_ckpt_enabled and self.rank == 0:
                    print("[FG] Gradient checkpointing enabled for transformer (saves ~60% VRAM during backward)", flush=True)

            if fg_enable and fg_loss_fn not in FG_VALID_LOSS_FNS:
                raise ValueError(f"fg_loss_fn must be one of {sorted(FG_VALID_LOSS_FNS)}, got '{fg_loss_fn}'")

            # Determine effective_fixed_frames based on loss type
            effective_fixed_frames = fg_fixed_frames
            fg_frame_weights = {}  # per-frame loss weights (style only)
            if fg_enable and fg_loss_fn == "loop":
                effective_fixed_frames = [0, frame_num - 1]
            elif fg_enable and fg_loss_fn == "style":
                if fg_fixed_frames is not None and len(fg_fixed_frames) > 0:
                    effective_fixed_frames = fg_fixed_frames
                else:
                    # I2V mode: concentrate on back half to avoid fighting
                    # the first-frame condition.
                    last = frame_num - 1
                    mid = last // 2
                    span = last - mid
                    n_fixed = 4
                    effective_fixed_frames = []
                    for i in range(n_fixed):
                        t_ratio = (i + 1) / n_fixed
                        ff = mid + int(span * t_ratio * t_ratio)
                        ff = min(ff, last)
                        effective_fixed_frames.append(ff)
                    effective_fixed_frames[-1] = last
                    seen = set()
                    deduped = []
                    for ff in effective_fixed_frames:
                        if ff not in seen:
                            seen.add(ff)
                            deduped.append(ff)
                    effective_fixed_frames = deduped

                # Per-frame weights: later frames get higher weight
                fg_frame_weights = {}
                if effective_fixed_frames:
                    last = frame_num - 1
                    for ff in effective_fixed_frames:
                        fg_frame_weights[ff] = ff / last if last > 0 else 1.0

            # Load FG video (no longer needed — scribble uses single image via _prepare_fg_loss_context)
            fg_video = None

            # Prepare FG loss context (models, embeddings, condition tensors)
            fg_loss_ctx = {}
            if fg_enable and effective_fixed_frames:
                fg_loss_ctx = self._prepare_fg_loss_context(
                    fg_loss_fn, fg_additional_inputs, fg_video, h, w)

            # Generate FG guidance schedules
            # Aligned with paper's Algorithm 1 (others_wan.ipynb):
            #   official: [0]*2 + [5]*8 + [3]*20 + [1]*10 + [0]*10  (50 steps, lr=3)
            #   ours:     [0]*2 + [5]*6 + [3]*16 + [1]*8  + [0]*8   (40 steps, lr=5)
            # Paper shows early steps have global gradient propagation (Fig 4d),
            # so only skip 2 steps. Previous skip=20% was too conservative.
            if fg_enable and effective_fixed_frames:
                _skip = 2                                    # match paper
                _s = max(1, round(total_steps * 0.15))       # 5 reps
                _m = max(1, round(total_steps * 0.40))       # 3 reps
                _l = max(1, round(total_steps * 0.20))       # 1 rep
                _r = max(0, total_steps - _skip - _s - _m - _l)  # 0 reps
                fg_step_schedule = ([0] * _skip + [5] * _s + [3] * _m +
                                    [1] * _l + [0] * _r)
                fg_step_schedule = fg_step_schedule[:total_steps]
                fg_lr_schedule = [fg_guidance_lr] * total_steps
            else:
                fg_step_schedule = [0] * total_steps
                fg_lr_schedule = [0.0] * total_steps

            if fg_enable and self.rank == 0:
                guided_steps = sum(1 for s in fg_step_schedule if s > 0)
                total_reps = sum(fg_step_schedule)
                print(f"[FG] Schedule: {guided_steps}/{total_steps} steps guided, "
                      f"{total_reps} total reps, schedule={fg_step_schedule}", flush=True)
                print(f"[FG] fixed_frames={effective_fixed_frames} "
                      f"frame_weights={fg_frame_weights} lr={fg_guidance_lr}", flush=True)

            for step_idx, t in enumerate(tqdm(timesteps)):
                current_latent = latent.to(self.device)
                movable_length = max(0, int(latent_length))
                use_shift = loopless_enable and shift_skip > 0 and step_idx < use_shift_threshold and movable_length > 1

                # Prepare model once per step
                model = self._prepare_model_for_timestep(
                    t, boundary, offload_model)
                sample_guide_scale = guide_scale[1] if t.item(
                ) >= boundary else guide_scale[0]

                sigma = get_sigma_at_step(sample_scheduler, step_idx)
                n_repeats = fg_step_schedule[step_idx]
                timestep_tensor = torch.stack([t]).to(self.device)

                # --- Inner repeat loop for FG guidance (paper's multi-step refinement) ---
                # Same structure as official frame-guidance: each rep re-runs
                # transformer with enable_grad, computes loss, backpropagates
                # through transformer.  Two-phase backward splits at pred_x0
                # so VAE/CSD can be offloaded before the heavy transformer backward.

                # Shift y condition (once per step)
                if use_shift and shift_idx > 0:
                    shifted_y = _temporal_roll(y, shift_idx)
                    cur_cond_y = [shifted_y]
                    cur_uncond_y = [shifted_y]
                else:
                    cur_cond_y = cond_y
                    cur_uncond_y = uncond_y

                for rep in range(n_repeats + 1):
                    need_grad = n_repeats > 0 and rep < n_repeats

                    # Detach latent for each rep (same as official: latents.detach().clone().requires_grad_())
                    current_latent = current_latent.detach().clone().requires_grad_(need_grad)
                    shifted_latent = _temporal_roll(current_latent, shift_idx) if use_shift else current_latent

                    # Transformer forward (with or without grad)
                    ctx_mgr = torch.enable_grad() if need_grad else torch.no_grad()
                    with ctx_mgr:
                        noise_pred_cond = model(
                            [shifted_latent],
                            t=timestep_tensor,
                            context=cond_context,
                            seq_len=max_seq_len,
                            y=cur_cond_y,
                        )[0]
                        if offload_model:
                            torch.cuda.empty_cache()
                        noise_pred_uncond = model(
                            [shifted_latent],
                            t=timestep_tensor,
                            context=uncond_context,
                            seq_len=max_seq_len,
                            y=cur_uncond_y,
                        )[0]
                        if offload_model:
                            torch.cuda.empty_cache()
                        noise_pred = noise_pred_uncond + sample_guide_scale * (
                            noise_pred_cond - noise_pred_uncond)
                        if use_shift:
                            noise_pred = _temporal_roll(noise_pred, shift_idx, reverse=True)

                        pred_x0 = current_latent - sigma * noise_pred

                    if need_grad:
                        # --- Single backward pass (direct gradient flow) ---
                        # Unlike the two-phase split, this allows full gradient flow
                        # from the loss through pred_x0 all the way to current_latent.
                        # This prevents the "gradient wall" that attenuates style signal.

                        with torch.enable_grad():
                            if fg_loss_fn == "loop":
                                loss = self._compute_fg_loss(
                                    fg_loss_fn, pred_x0, None, effective_fixed_frames,
                                    fg_loss_ctx, fg_additional_inputs, frame_num, step_idx, rep)
                            else:
                                if fg_loss_fn == "style":
                                    sample_frames = random.sample(
                                        effective_fixed_frames,
                                        min(4, len(effective_fixed_frames)))
                                else:
                                    sample_frames = effective_fixed_frames
                                loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                                for ff in sample_frames:
                                    ff_loss = self._compute_fg_loss(
                                        fg_loss_fn, pred_x0, ff, effective_fixed_frames,
                                        fg_loss_ctx, fg_additional_inputs, frame_num, step_idx, rep)
                                    if ff_loss is not None:
                                        # Apply per-frame weight for style loss:
                                        # later frames get higher weight to concentrate
                                        # gradient on the tail (last frame = edit output).
                                        w = fg_frame_weights.get(ff, 1.0) if fg_loss_fn == "style" else 1.0
                                        loss = loss + w * ff_loss

                            # Single backward: loss → current_latent (full graph)
                            loss.backward()

                        print(f"  [FG] step={step_idx} rep={rep} loss={loss.item():.6f} "
                              f"|current_latent.grad|={current_latent.grad.norm().item():.6f}", flush=True)

                        # Diagnostic: check gradient structure per temporal frame
                        if self.rank == 0 and rep == 0:
                            g = current_latent.grad
                            # Per-frame gradient norm: [C, T, H, W] → norm over C,H,W for each T
                            per_frame_norms = [g[:, ti, :, :].norm().item() for ti in range(g.shape[1])]
                            print(f"  [FG-DIAG] grad per-frame norms: {[f'{n:.4f}' for n in per_frame_norms]}", flush=True)
                            # Gradient sparsity: what fraction of elements are near-zero?
                            total_els = g.numel()
                            near_zero = (g.abs() < 1e-8).sum().item()
                            print(f"  [FG-DIAG] grad sparsity: {near_zero}/{total_els} ({100*near_zero/total_els:.1f}%) near-zero", flush=True)
                            print(f"  [FG-DIAG] grad min={g.min().item():.6f} max={g.max().item():.6f} "
                                  f"mean={g.mean().item():.6f} std={g.std().item():.6f}", flush=True)
                            del g

                        del loss, pred_x0

                        # Free VAE/CSD from GPU after backward
                        self._offload_fg_models_after_backward(fg_loss_fn)
                        torch.cuda.empty_cache()

                        grad = current_latent.grad.detach().clone()
                        current_latent.grad = None

                        # Temporal gradient mask: protect early frames, concentrate
                        # style on tail frames. Quadratic ramp along T dimension.
                        if fg_loss_fn == "style" and grad.dim() >= 2:
                            T = grad.shape[1]
                            tw = torch.linspace(0, 1, T, device=grad.device, dtype=grad.dtype) ** 2
                            grad = grad * tw.reshape(1, T, 1, 1)

                        # Apply gradient update (paper's _apply_guidance_update)
                        grad_norm = grad.norm(2)
                        rho = 1.0 / grad_norm if grad_norm > 0 else 0.0
                        latent_before_fg = current_latent.detach().clone()
                        with torch.no_grad():
                            in_travel = fg_travel_time[0] <= step_idx <= fg_travel_time[1]
                            lr = fg_lr_schedule[step_idx]
                            if in_travel:
                                pred_x0_no_graph = current_latent.detach() - sigma * noise_pred.detach()
                                noise_g = torch.randn_like(current_latent)
                                current_latent = sigma * noise_g + (1 - sigma) * pred_x0_no_graph
                            current_latent = current_latent - lr * rho * grad
                        fg_delta = (current_latent - latent_before_fg).norm().item()
                        print(f"  [FG] step={step_idx} rep={rep} |grad|={grad.norm().item():.4f} "
                              f"lr={lr:.4f} rho={rho:.4f} |fg_delta|={fg_delta:.4f} "
                              f"|latent|={current_latent.norm().item():.4f} "
                              f"travel={in_travel}", flush=True)
                        del grad
                        torch.cuda.empty_cache()

                # Update Loopless shift_idx once per denoising step
                if use_shift:
                    shift_idx = (shift_idx + shift_skip) % movable_length

                # Scheduler step — always use Euler when FG is enabled.
                if fg_enable:
                    sigma_next = get_sigma_at_step(sample_scheduler, step_idx + 1) \
                        if step_idx + 1 < total_steps else torch.tensor(0.0, device=self.device, dtype=current_latent.dtype)
                    with torch.no_grad():
                        euler_delta = (sigma_next - sigma) * noise_pred
                        latent = current_latent + euler_delta
                    if n_repeats > 0 and self.rank == 0:
                        print(f"  [FG-DEBUG] step={step_idx} sigma={sigma:.4f} sigma_next={sigma_next:.4f} "
                              f"|euler_delta|={euler_delta.norm().item():.4f} "
                              f"|current_latent|={current_latent.norm().item():.4f} "
                              f"|latent_after|={latent.norm().item():.4f}", flush=True)
                    # Advance scheduler index so timesteps stay in sync
                    if hasattr(sample_scheduler, '_step_index') and sample_scheduler._step_index is not None:
                        sample_scheduler._step_index += 1
                    elif hasattr(sample_scheduler, 'step_index'):
                        pass  # some schedulers auto-advance
                else:
                    scheduler_t = t.to(current_latent.device) if isinstance(t, torch.Tensor) else t
                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0),
                        scheduler_t,
                        current_latent.unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)[0]
                    latent = temp_x0.squeeze(0)

                if ifedit_use_tld and t.item() <= threshold_timestep and not tld_applied:
                    tld_indices = _make_tld_indices(
                        frame_count=latent.shape[1],
                        step_k=max(1, int(ifedit_tld_step_k)),
                        device=latent.device,
                    )
                    latent = _apply_tld(latent, tld_indices)
                    y = _apply_tld(y, tld_indices)
                    cond_y = [y]
                    uncond_y = [y]
                    _reset_scheduler_state(sample_scheduler)
                    tld_applied = True
                    latent_length = latent.shape[1]
                    movable_length = max(0, int(latent_length))
                    shift_idx = shift_idx % max(1, movable_length)
                    if self.rank == 0:
                        print(
                            f"I2V: TLD applied at step {step_idx + 1}/{total_steps}, "
                            f"frames {latent.shape[1]}",
                            flush=True,
                        )

                x0 = [latent]
                if self.rank == 0:
                    print(f"Step {step_idx + 1}/{total_steps}", flush=True)

            # Disable gradient checkpointing after FG loop to restore normal inference
            if _fg_grad_ckpt_enabled:
                for m in [self.high_noise_model, self.low_noise_model]:
                    if m is not None:
                        m.gradient_checkpointing = False

            # Offload FG auxiliary models (CSD etc.) to free VRAM for VAE decode
            if fg_enable and hasattr(self, '_fg_csd') and self._fg_csd is not None:
                self._fg_csd.cpu()
                torch.cuda.empty_cache()

            if offload_model:
                self.low_noise_model.cpu()
                self.high_noise_model.cpu()
                self._active_noise_model = None  # reset so next generate() re-loads correctly
                torch.cuda.empty_cache()

            if self.rank == 0:
                self.vae.model.to(self.device)
                videos = self.vae.decode(x0)
                video_original = videos[0]
                if offload_model:
                    self.vae.model.cpu()
                    torch.cuda.empty_cache()

        del noise, latent, x0
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        if self.rank == 0:
            return video_original
        return None
