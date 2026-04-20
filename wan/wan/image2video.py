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
from typing import Optional

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .configs.param_types import GenerateParams
from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .fg_utils import FGConfig, FGMixin, get_sigma_at_step
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
# I2V-specific helpers (TLD, temporal roll)
# ---------------------------------------------------------------------------

def _make_tld_indices(frame_count, step_k, device):
    """Create progressive decimation indices: sparse at front, dense at back.

    For sketch FG the last frame matters most, so we preserve more temporal
    resolution near the end.  Always includes first and last frame.
    """
    if frame_count <= 1 or step_k <= 1:
        return torch.arange(frame_count, device=device)

    target_count = max(3, (frame_count + step_k - 1) // step_k + 1)

    # Quadratic spacing: t^2 maps [0,1] → [0,1], dense at end
    t = np.linspace(0, 1, target_count)
    positions = t ** 2
    raw = (positions * (frame_count - 1)).astype(int)

    indices = sorted(set([0] + raw.tolist() + [frame_count - 1]))
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


class WanI2V(FGMixin):

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

        if self._active_noise_model != required_name:
            required_model.to(device=self.device, dtype=self.param_dtype)
            if (offload_model or self.init_on_cpu) and next(inactive_model.parameters()).device.type == "cuda":
                inactive_model.to("cpu")
            torch.cuda.empty_cache()
            self._active_noise_model = required_name
        else:
            if (offload_model or self.init_on_cpu) and next(inactive_model.parameters()).device.type == "cuda":
                inactive_model.to("cpu")
                torch.cuda.empty_cache()
            if next(required_model.parameters()).device.type == "cpu":
                required_model.to(device=self.device, dtype=self.param_dtype)
                torch.cuda.empty_cache()

        return required_model

    # ------------------------------------------------------------------
    # Main generation method
    # ------------------------------------------------------------------

    def generate(self, p: GenerateParams, img):
        r"""
        Generates video frames from input image and text prompt.

        Args:
            p: Structured generation parameters.
            img: PIL Image (already opened).

        Returns:
            torch.Tensor: Generated video frames tensor. Dimensions: (C, N, H, W)
        """
        from .configs import MAX_AREA_CONFIGS

        # Unpack params
        max_area = MAX_AREA_CONFIGS[p.size]
        frame_num = p.frame_num
        seed = p.seed
        offload_model = p.model.offload_model
        s = p.sampling
        ie = p.ifedit
        lp = p.loopless

        guide_scale = s.guide_scale
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

        n_prompt = p.n_prompt or self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([p.prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([p.prompt], torch.device('cpu'))
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

            if s.solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    s.steps, device=self.device, shift=s.shift)
                timesteps = sample_scheduler.timesteps
            elif s.solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(s.steps, s.shift)
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
            threshold_timestep = ie.tld_threshold_ratio * self.num_train_timesteps

            shift_idx = 0
            latent_length = latent.shape[1]
            shift_skip = max(0, int(lp.shift_skip))
            use_shift_threshold = total_steps - int(lp.shift_stop_step) if int(lp.shift_stop_step) > 0 else total_steps

            # --- FG setup ------------------------------------------------
            fg = self.setup_fg(
                fg_params=p.fg,
                offload_model=offload_model,
                frame_num=frame_num,
                total_steps=total_steps,
                target_h=h,
                target_w=w,
                mode="i2v",
            )

            for step_idx, t in enumerate(tqdm(timesteps)):
                current_latent = latent.to(self.device)
                movable_length = max(0, int(latent_length))
                use_shift = lp.enable and shift_skip > 0 and step_idx < use_shift_threshold and movable_length > 1

                # Prepare model once per step
                model = self._prepare_model_for_timestep(
                    t, boundary, offload_model)
                sample_guide_scale = guide_scale[1] if t.item(
                ) >= boundary else guide_scale[0]

                sigma = get_sigma_at_step(sample_scheduler, step_idx)
                sigma_next = get_sigma_at_step(sample_scheduler, step_idx + 1) \
                    if step_idx + 1 < total_steps else torch.tensor(0.0, device=self.device, dtype=noise.dtype)
                n_repeats = fg.step_schedule[step_idx]
                timestep_tensor = torch.stack([t]).to(self.device)

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

                    # Detach latent for each rep
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

                        # DEBUG: Verify noise_pred changes across reps
                        if self.rank == 0 and n_repeats > 0:
                            print(f"    [FG-REP-DEBUG] rep={rep} |noise_pred|={noise_pred.norm().item():.4f} |latent_in|={shifted_latent.norm().item():.4f}", flush=True)

                    if need_grad:
                        grad = self.compute_fg_step_loss(
                            fg, pred_x0, current_latent,
                            frame_num, step_idx, rep)
                        current_latent = self.apply_fg_gradient(
                            fg, current_latent, grad, noise_pred,
                            sigma, sigma_next, step_idx, rep)

                # Update Loopless shift_idx once per denoising step
                if use_shift:
                    shift_idx = (shift_idx + shift_skip) % movable_length

                # Scheduler step — always use Euler when FG is enabled.
                if fg.enabled:
                    # sigma_next already computed above
                    with torch.no_grad():
                        euler_delta = (sigma_next - sigma) * noise_pred
                        latent_before_euler = current_latent.norm().item()
                        latent = current_latent + euler_delta
                    if n_repeats > 0 and self.rank == 0:
                        latent_change_by_euler = (latent - current_latent).norm().item()
                        print(f"  [FG-DEBUG] step={step_idx} sigma={sigma:.4f} sigma_next={sigma_next:.4f} "
                              f"|euler_delta|={euler_delta.norm().item():.4f} "
                              f"|fg_modified_latent|={latent_before_euler:.4f} "
                              f"|latent_after_euler|={latent.norm().item():.4f} "
                              f"|change_by_euler|={latent_change_by_euler:.4f}", flush=True)
                    # Advance scheduler index so timesteps stay in sync
                    if hasattr(sample_scheduler, '_step_index') and sample_scheduler._step_index is not None:
                        sample_scheduler._step_index += 1
                    elif hasattr(sample_scheduler, 'step_index'):
                        pass
                else:
                    scheduler_t = t.to(current_latent.device) if isinstance(t, torch.Tensor) else t
                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0),
                        scheduler_t,
                        current_latent.unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)[0]
                    latent = temp_x0.squeeze(0)

                if ie.use_tld and t.item() <= threshold_timestep and not tld_applied:
                    tld_indices = _make_tld_indices(
                        frame_count=latent.shape[1],
                        step_k=max(1, int(ie.tld_step_k)),
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
                    # Update frame_num and handle FG after TLD
                    frame_num = (latent_length - 1) * 4 + 1
                    if fg.enabled:
                        if ie.tld_stop_fg:
                            # Stop FG completely after TLD
                            for si in range(step_idx + 1, len(fg.step_schedule)):
                                fg.step_schedule[si] = 0
                        elif fg.fixed_frames:
                            # Continue FG with remapped frames
                            fg.fixed_frames = [frame_num - 1]
                    if self.rank == 0:
                        print(
                            f"I2V: TLD applied at step {step_idx + 1}/{total_steps}, "
                            f"latent_frames={latent_length} pixel_frames={frame_num} "
                            f"tld_stop_fg={ie.tld_stop_fg} fg.fixed_frames={fg.fixed_frames}",
                            flush=True,
                        )

                x0 = [latent]
                if self.rank == 0:
                    print(f"Step {step_idx + 1}/{total_steps}", flush=True)

            # --- FG teardown ------------------------------------------
            self.teardown_fg(fg)

            if offload_model:
                self.low_noise_model.cpu()
                self.high_noise_model.cpu()
                self._active_noise_model = None
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
