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


# ---------------------------------------------------------------------------
# Module-level FG helpers (shared with image2video.py)
# ---------------------------------------------------------------------------

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
    """Differentiable image-to-observation operator for scribble loss."""

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


FG_VALID_LOSS_FNS = {"style", "scribble", "loop"}


class WanT2V:

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
        Initializes the Wan text-to-video generation model components.

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
            shard_fn=shard_fn if t5_fsdp else None)

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
        """
        if t.item() >= boundary:
            required_model_name = 'high_noise_model'
            offload_model_name = 'low_noise_model'
        else:
            required_model_name = 'low_noise_model'
            offload_model_name = 'high_noise_model'
        if offload_model or self.init_on_cpu:
            if next(getattr(
                    self,
                    offload_model_name).parameters()).device.type == 'cuda':
                getattr(self, offload_model_name).to('cpu')
            if next(getattr(
                    self,
                    required_model_name).parameters()).device.type == 'cpu':
                getattr(self, required_model_name).to(self.device)
        return getattr(self, required_model_name)

    # ------------------------------------------------------------------
    # Frame Guidance helper methods (adapted from WanI2V)
    # ------------------------------------------------------------------

    def _prepare_fg_loss_context(self, fg_loss_fn: str, fg_additional_inputs: Optional[Dict],
                                  fg_video: Optional[torch.Tensor],
                                  target_h: int, target_w: int):
        """Prepare pre-computed context for the chosen FG loss function.

        Supports: style, scribble, loop.
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
                import torchvision
                self._fg_clip_preprocess_tensor = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(224,
                                                  interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                    torchvision.transforms.CenterCrop(224),
                    torchvision.transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711)),
                ])
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
            scribble_tensor = _load_fg_image(scribble_path, device, target_hw=(target_h, target_w))
            frame_01 = (scribble_tensor[:, :, 0] + 1.0) / 2.0  # [1, 3, H, W]
            with torch.no_grad():
                scribble_cond = self._fg_aug(frame_01.to(dtype), mode="scribble")
                if self.fg_downscale_factor > 1:
                    scribble_cond = _F.interpolate(scribble_cond, scale_factor=1 / self.fg_downscale_factor,
                                                    mode='bilinear', align_corners=False)
            ctx["scribble_cond"] = scribble_cond

        elif fg_loss_fn == "loop":
            pass

        return ctx

    def _offload_fg_models_after_backward(self, fg_loss_fn: str):
        """Offload FG auxiliary models (VAE, CSD) to CPU after backward."""
        if not self.fg_offload_model:
            return
        self.vae.model.cpu()
        if fg_loss_fn == "style" and hasattr(self, '_fg_csd'):
            self._fg_csd.cpu()
        torch.cuda.empty_cache()

    def _compute_fg_loss(self, fg_loss_fn: str, pred_x0: torch.Tensor,
                          ff: int, fg_fixed_frames: list,
                          fg_loss_ctx: Dict, fg_additional_inputs: Optional[Dict],
                          frame_num: int, step_idx: int, rep_idx: int):
        """Compute FG loss for a single fixed frame or loop loss.

        Supports: style, scribble, loop.
        """
        offload_model = self.fg_offload_model

        if fg_loss_fn == "loop":
            predicted_images = []
            for fixed_frame in [0, frame_num - 1]:
                if fixed_frame == 0:
                    pair = pred_x0[:, 0:1]
                    rel = 0
                else:
                    lf = (fixed_frame - 1) // 4 + 1
                    pair = pred_x0[:, lf - 1:lf + 1]
                    rel = (fixed_frame - 1) % 4 + 1
                if self.fg_downscale_factor > 1:
                    pair = _downscale_spatial(pair, self.fg_downscale_factor)
                if offload_model:
                    self.vae.model.to(self.device)
                with torch.amp.autocast('cuda', enabled=False):
                    decoded = self.vae.model.decode(
                        pair.float().unsqueeze(0), self.vae.scale
                    ).float().clamp(-1, 1).squeeze(0)
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
        pair = pred_x0[:, lf - 1:lf + 1]
        if self.fg_downscale_factor > 1:
            pair = _downscale_spatial(pair, self.fg_downscale_factor)
        if offload_model:
            self.vae.model.to(self.device)
        with torch.amp.autocast('cuda', enabled=False):
            pair_f32 = pair.float()
            decoded = self.vae.model.decode(
                pair_f32.unsqueeze(0), self.vae.scale
            ).squeeze(0)
            decoded = decoded.float()
        rel = (ff - 1) % 4 + 1
        pred_f = decoded[:, rel:rel + 1]
        del pair

        if fg_loss_fn == "style":
            pred_f_sq = pred_f[:, 0].unsqueeze(0)  # [1, 3, H, W]

            if offload_model and hasattr(self, '_fg_csd'):
                self._fg_csd.to(self.device)
            with torch.amp.autocast('cuda', enabled=False):
                pred_f_input = self._fg_clip_preprocess_tensor(pred_f_sq.float())
                _, content_emb, style_emb = self._fg_csd(pred_f_input)

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

            cos = fg_loss_ctx["cosine_loss"]
            style_sim = cos(style_emb, style_emb_ref)
            print(f"    style_similarity: {style_sim.item():.3f}", flush=True)
            loss = fg_loss_ctx["style_weight"] * (1 - style_sim)
            if fg_loss_ctx.get("content_image") is not None and fg_loss_ctx.get("content_weight", 0) > 0:
                content_emb_ref = fg_additional_inputs["content_embed"].to(self.device)
                content_sim = cos(content_emb, content_emb_ref)
                loss = loss + fg_loss_ctx["content_weight"] * (1 - content_sim)

        elif fg_loss_fn == "scribble":
            sketch_obs = self._fg_aug(pred_f[:, 0].unsqueeze(0).float(), mode="scribble")
            loss = _F.mse_loss(sketch_obs, fg_loss_ctx["scribble_cond"].float())

        del decoded, pred_f
        return loss

    # ------------------------------------------------------------------
    # Main generation method
    # ------------------------------------------------------------------

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 # Frame Guidance parameters
                 fg_enable=False,
                 fg_fixed_frames=None,
                 fg_guidance_lr=5.0,
                 fg_travel_time=(-1, -1),
                 fg_downscale_factor=4,
                 fg_loss_fn="style",
                 fg_additional_inputs=None):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (`tuple[int]`, *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps.
            guide_scale (`float` or tuple[`float`], *optional*, defaults 5.0):
                Classifier-free guidance scale.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion.
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            fg_enable (`bool`, *optional*, defaults to False):
                Enable Frame Guidance.
            fg_fixed_frames (`list[int]`, *optional*):
                Which pixel-space frames to apply FG loss on.
                If None, uses uniform distribution matching the paper.
            fg_guidance_lr (`float`, *optional*, defaults to 5.0):
                Learning rate for FG gradient updates.
            fg_travel_time (`tuple[int,int]`, *optional*, defaults to (-1,-1)):
                Step range for travel-time noise injection.
            fg_downscale_factor (`int`, *optional*, defaults to 4):
                Spatial downscale factor for VAE decode during FG.
            fg_loss_fn (`str`, *optional*, defaults to "style"):
                FG loss function: "style", "scribble", or "loop".
            fg_additional_inputs (`dict`, *optional*):
                Additional inputs for FG loss (e.g. style_image, scribble_image_path).

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N, H, W)
        """
        # preprocess
        guide_scale = (guide_scale, guide_scale) if isinstance(
            guide_scale, float) else guide_scale
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

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

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low_noise = getattr(self.low_noise_model, 'no_sync',
                                    noop_no_sync)
        no_sync_high_noise = getattr(self.high_noise_model, 'no_sync',
                                     noop_no_sync)

        # Store FG config on self so helper methods can access them
        self.fg_offload_model = offload_model
        self.fg_downscale_factor = fg_downscale_factor

        # Pixel-space target dimensions (for scribble loss)
        h = size[1]
        w = size[0]

        # evaluation mode — torch.no_grad() wraps the whole block.
        # FG steps temporarily exit no_grad via torch.enable_grad().
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_low_noise(),
                no_sync_high_noise(),
        ):
            boundary = self.boundary * self.num_train_timesteps

            # Force Euler scheduler when FG is enabled to avoid multi-step
            # solver interference with per-step gradient updates.
            effective_solver = sample_solver
            if fg_enable:
                effective_solver = 'unipc'  # We'll use Euler stepping manually

            if effective_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif effective_solver == 'dpm++':
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

            total_steps = len(timesteps)

            # sample videos
            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            # -------------------------------------------------------
            # FG setup (only when fg_enable=True)
            # -------------------------------------------------------
            _fg_grad_ckpt_enabled = False
            if fg_enable:
                # Enable gradient checkpointing to save VRAM during backward
                for m in [self.high_noise_model, self.low_noise_model]:
                    if m is not None:
                        m.gradient_checkpointing = True
                        _fg_grad_ckpt_enabled = True
                if _fg_grad_ckpt_enabled and self.rank == 0:
                    print("[FG] Gradient checkpointing enabled for transformer", flush=True)

            if fg_enable and fg_loss_fn not in FG_VALID_LOSS_FNS:
                raise ValueError(f"fg_loss_fn must be one of {sorted(FG_VALID_LOSS_FNS)}, got '{fg_loss_fn}'")

            # Determine effective fixed frames
            effective_fixed_frames = fg_fixed_frames
            fg_frame_weights = {}
            if fg_enable and fg_loss_fn == "loop":
                effective_fixed_frames = [0, frame_num - 1]
            elif fg_enable and fg_loss_fn in ("style", "scribble"):
                if fg_fixed_frames is not None and len(fg_fixed_frames) > 0:
                    effective_fixed_frames = fg_fixed_frames
                else:
                    # T2V mode: uniform distribution matching the paper
                    # For 81 frames: [20, 40, 60, 80]
                    last = frame_num - 1
                    n_fixed = 4
                    effective_fixed_frames = []
                    for i in range(1, n_fixed + 1):
                        ff = int(last * i / n_fixed)
                        effective_fixed_frames.append(ff)
                    # Deduplicate
                    seen = set()
                    deduped = []
                    for ff in effective_fixed_frames:
                        if ff not in seen:
                            seen.add(ff)
                            deduped.append(ff)
                    effective_fixed_frames = deduped

                # Per-frame weights for style loss (T2V: equal weight, no first-frame bias)
                if fg_loss_fn == "style":
                    fg_frame_weights = {}
                    if effective_fixed_frames:
                        for ff in effective_fixed_frames:
                            fg_frame_weights[ff] = 1.0

            # Prepare FG loss context
            fg_video = None
            fg_loss_ctx = {}
            if fg_enable and effective_fixed_frames:
                fg_loss_ctx = self._prepare_fg_loss_context(
                    fg_loss_fn, fg_additional_inputs, fg_video, h, w)

            # Generate FG guidance schedules
            if fg_enable and effective_fixed_frames:
                _skip = 2
                _s = max(1, round(total_steps * 0.15))
                _m = max(1, round(total_steps * 0.40))
                _l = max(1, round(total_steps * 0.20))
                _r = max(0, total_steps - _skip - _s - _m - _l)
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

            # -------------------------------------------------------
            # Denoising loop
            # -------------------------------------------------------
            for step_idx, t in enumerate(tqdm(timesteps)):
                current_latent = latents[0]

                model = self._prepare_model_for_timestep(
                    t, boundary, offload_model)
                sample_guide_scale = guide_scale[1] if t.item(
                ) >= boundary else guide_scale[0]

                timestep_tensor = torch.stack([t]).to(self.device)

                if fg_enable:
                    sigma = get_sigma_at_step(sample_scheduler, step_idx)
                    n_repeats = fg_step_schedule[step_idx]
                else:
                    n_repeats = 0

                for rep in range(n_repeats + 1):
                    need_grad = fg_enable and n_repeats > 0 and rep < n_repeats

                    if fg_enable:
                        current_latent = current_latent.detach().clone().requires_grad_(need_grad)

                    ctx_mgr = torch.enable_grad() if need_grad else torch.no_grad()
                    with ctx_mgr:
                        noise_pred_cond = model(
                            [current_latent], t=timestep_tensor, **arg_c)[0]
                        noise_pred_uncond = model(
                            [current_latent], t=timestep_tensor, **arg_null)[0]

                        noise_pred = noise_pred_uncond + sample_guide_scale * (
                            noise_pred_cond - noise_pred_uncond)

                        if fg_enable:
                            pred_x0 = current_latent - sigma * noise_pred

                    if need_grad:
                        # FG loss computation + backward + gradient update
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
                                        w = fg_frame_weights.get(ff, 1.0) if fg_loss_fn == "style" else 1.0
                                        loss = loss + w * ff_loss

                            loss.backward()

                        print(f"  [FG] step={step_idx} rep={rep} loss={loss.item():.6f} "
                              f"|current_latent.grad|={current_latent.grad.norm().item():.6f}", flush=True)

                        # Diagnostic
                        if self.rank == 0 and rep == 0:
                            g = current_latent.grad
                            per_frame_norms = [g[:, ti, :, :].norm().item() for ti in range(g.shape[1])]
                            print(f"  [FG-DIAG] grad per-frame norms: {[f'{n:.4f}' for n in per_frame_norms]}", flush=True)
                            total_els = g.numel()
                            near_zero = (g.abs() < 1e-8).sum().item()
                            print(f"  [FG-DIAG] grad sparsity: {near_zero}/{total_els} ({100*near_zero/total_els:.1f}%) near-zero", flush=True)
                            print(f"  [FG-DIAG] grad min={g.min().item():.6f} max={g.max().item():.6f} "
                                  f"mean={g.mean().item():.6f} std={g.std().item():.6f}", flush=True)
                            del g

                        del loss, pred_x0

                        self._offload_fg_models_after_backward(fg_loss_fn)
                        torch.cuda.empty_cache()

                        grad = current_latent.grad.detach().clone()
                        current_latent.grad = None

                        # T2V: no temporal gradient mask (no first-frame constraint,
                        # all frames should be styled uniformly)

                        # Apply gradient update
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

                # Scheduler step
                if fg_enable:
                    sigma_next = get_sigma_at_step(sample_scheduler, step_idx + 1) \
                        if step_idx + 1 < total_steps else torch.tensor(0.0, device=self.device, dtype=current_latent.dtype)
                    with torch.no_grad():
                        euler_delta = (sigma_next - sigma) * noise_pred
                        latents = [current_latent + euler_delta]
                    if n_repeats > 0 and self.rank == 0:
                        print(f"  [FG-DEBUG] step={step_idx} sigma={sigma:.4f} sigma_next={sigma_next:.4f} "
                              f"|euler_delta|={euler_delta.norm().item():.4f} "
                              f"|current_latent|={current_latent.norm().item():.4f} "
                              f"|latent_after|={latents[0].norm().item():.4f}", flush=True)
                    if hasattr(sample_scheduler, '_step_index') and sample_scheduler._step_index is not None:
                        sample_scheduler._step_index += 1
                    elif hasattr(sample_scheduler, 'step_index'):
                        pass
                else:
                    with torch.no_grad():
                        temp_x0 = sample_scheduler.step(
                            noise_pred.unsqueeze(0),
                            t,
                            latents[0].unsqueeze(0),
                            return_dict=False,
                            generator=seed_g)[0]
                        latents = [temp_x0.squeeze(0)]

            # Disable gradient checkpointing after FG loop
            if _fg_grad_ckpt_enabled:
                for m in [self.high_noise_model, self.low_noise_model]:
                    if m is not None:
                        m.gradient_checkpointing = False

            print("[T2V] Denoising loop done, preparing for VAE decode...", flush=True)

            # Detach final latent from any computation graph
            x0 = [latents[0].detach().clone()]
            del latents

            # Offload FG auxiliary models (CSD etc.) to free VRAM for VAE decode
            if fg_enable and hasattr(self, '_fg_csd') and self._fg_csd is not None:
                self._fg_csd.cpu()
                torch.cuda.empty_cache()

            # Offload transformers before VAE decode (same order as I2V)
            if offload_model:
                self.low_noise_model.cpu()
                self.high_noise_model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                lat = x0[0]
                print(f"[T2V] Latent stats: shape={lat.shape} "
                      f"min={lat.min().item():.2f} max={lat.max().item():.2f} "
                      f"mean={lat.mean().item():.2f} std={lat.std().item():.2f}", flush=True)
                if lat.abs().max().item() > 100:
                    print("[T2V] WARNING: clamping extreme latent values", flush=True)
                    x0 = [lat.clamp(-50, 50)]

                self.vae.model.to(self.device)
                print("[T2V] Starting VAE decode...", flush=True)
                videos = self.vae.decode(x0)
                print("[T2V] VAE decode complete.", flush=True)
                if offload_model:
                    self.vae.model.cpu()
                    torch.cuda.empty_cache()

        del noise
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
