# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial
from typing import Optional

import torch
import torch.distributed as dist
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


class WanT2V(FGMixin):

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
    # Main generation method
    # ------------------------------------------------------------------

    def generate(self, p: GenerateParams):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            p: Structured generation parameters.

        Returns:
            torch.Tensor: Generated video frames tensor. Dimensions: (C, N, H, W)
        """
        from .configs import SIZE_CONFIGS

        def _dump_cuda_top(tag="", top_n=15):
            """Print top CUDA tensors by size for memory debugging."""
            import gc
            tensors = []
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) and obj.is_cuda:
                        tensors.append((obj.numel() * obj.element_size(),
                                        obj.shape, obj.dtype))
                except Exception:
                    pass
            tensors.sort(reverse=True)
            total = sum(t[0] for t in tensors)
            print(f"  [CUDA-TENSORS {tag}] count={len(tensors)} "
                  f"total={total/1024**3:.2f}GB", flush=True)
            for sz, shape, dtype in tensors[:top_n]:
                print(f"    {sz/1024**2:8.1f} MB  {str(dtype):15s}  {shape}",
                      flush=True)

        # Unpack params
        size = SIZE_CONFIGS[p.size]
        frame_num = p.frame_num
        seed = p.seed
        offload_model = p.model.offload_model
        s = p.sampling

        guide_scale = s.guide_scale
        guide_scale = (guide_scale, guide_scale) if isinstance(
            guide_scale, float) else guide_scale
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        n_prompt = p.n_prompt or self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

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

        # evaluation mode — torch.no_grad() wraps the whole block.
        # FG steps temporarily exit no_grad via torch.enable_grad().
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_low_noise(),
                no_sync_high_noise(),
        ):
            boundary = self.boundary * self.num_train_timesteps

            # Force Euler scheduler when FG is enabled
            effective_solver = s.solver
            if p.fg.enable:
                effective_solver = 'unipc'

            if effective_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    s.steps, device=self.device, shift=s.shift)
                timesteps = sample_scheduler.timesteps
            elif effective_solver == 'dpm++':
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

            total_steps = len(timesteps)

            # Print sigma sequence for debugging
            if self.rank == 0:
                sigmas = [get_sigma_at_step(sample_scheduler, i).item() for i in range(total_steps + 1)]
                print(f"  [SIGMA] sequence ({len(sigmas)} values): "
                      f"first5={sigmas[:5]} last5={sigmas[-5:]}", flush=True)
                print(f"  [SIGMA] boundary={self.boundary}, boundary_t={boundary:.0f}", flush=True)

            # sample videos
            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            # --- FG setup ------------------------------------------------
            fg = self.setup_fg(
                fg_params=p.fg,
                offload_model=offload_model,
                frame_num=frame_num,
                total_steps=total_steps,
                target_h=size[1],
                target_w=size[0],
                mode="t2v",
            )

            # -------------------------------------------------------
            # Denoising loop
            # -------------------------------------------------------
            if self.rank == 0:
                a = torch.cuda.memory_allocated(self.device) / 1024**3
                r = torch.cuda.memory_reserved(self.device) / 1024**3
                h_dev = next(self.high_noise_model.parameters()).device
                l_dev = next(self.low_noise_model.parameters()).device
                vae_dev = next(self.vae.model.parameters()).device
                print(f"  [BASELINE] alloc={a:.2f}GB reserved={r:.2f}GB "
                      f"high={h_dev} low={l_dev} vae={vae_dev}", flush=True)
            for step_idx, t in enumerate(tqdm(timesteps)):
                current_latent = latents[0]

                model = self._prepare_model_for_timestep(
                    t, boundary, offload_model)
                sample_guide_scale = guide_scale[1] if t.item(
                ) >= boundary else guide_scale[0]

                timestep_tensor = torch.stack([t]).to(self.device)

                if fg.enabled:
                    sigma = get_sigma_at_step(sample_scheduler, step_idx)
                    sigma_next = get_sigma_at_step(sample_scheduler, step_idx + 1) \
                        if step_idx + 1 < total_steps else torch.tensor(0.0, device=self.device, dtype=noise.dtype)
                    n_repeats = fg.step_schedule[step_idx]
                else:
                    n_repeats = 0

                def _vram(tag=""):
                    if self.rank == 0:
                        a = torch.cuda.memory_allocated(self.device) / 1024**3
                        r = torch.cuda.memory_reserved(self.device) / 1024**3
                        print(f"  [VRAM] {tag}: alloc={a:.2f}GB reserved={r:.2f}GB", flush=True)

                def _model_devices():
                    if self.rank == 0:
                        h_dev = next(self.high_noise_model.parameters()).device
                        l_dev = next(self.low_noise_model.parameters()).device
                        vae_dev = next(self.vae.model.parameters()).device
                        csd_dev = "N/A"
                        if hasattr(self, '_fg_csd') and self._fg_csd is not None:
                            csd_dev = next(self._fg_csd.parameters()).device
                        print(f"  [DEVICES] high={h_dev} low={l_dev} vae={vae_dev} csd={csd_dev}", flush=True)

                for rep in range(n_repeats + 1):
                    need_grad = fg.enabled and n_repeats > 0 and rep < n_repeats

                    if fg.enabled:
                        current_latent = current_latent.detach().clone().requires_grad_(need_grad)

                    if need_grad:
                        # Offload idle transformer to free ~26GB for grad computation
                        if rep == 0:
                            idle_name = 'low_noise_model' if t.item() >= boundary else 'high_noise_model'
                            idle_m = getattr(self, idle_name)
                            if next(idle_m.parameters()).device.type == 'cuda':
                                idle_m.cpu()
                                torch.cuda.empty_cache()
                                if self.rank == 0:
                                    print(f"  [OFFLOAD] {idle_name} → cpu", flush=True)
                            if self.rank == 0 and step_idx <= 3:
                                _model_devices()
                                _vram(f"step={step_idx} after_idle_offload")

                        # With bf16 model (~27GB) + idle offloaded, dual-path
                        # grad fits: 27GB + 15.5GB×2 + VAE/CSD ~10GB ≈ 68GB.
                        with torch.enable_grad():
                            noise_pred_cond = model(
                                [current_latent], t=timestep_tensor, **arg_c)[0]
                            noise_pred_uncond = model(
                                [current_latent], t=timestep_tensor, **arg_null)[0]
                            noise_pred = noise_pred_uncond + sample_guide_scale * (
                                noise_pred_cond - noise_pred_uncond)
                            pred_x0 = current_latent - sigma * noise_pred

                            # --- Diagnostic: computation graph ---
                            if rep == 0 and self.rank == 0:
                                if step_idx <= 3 or step_idx % 10 == 0:
                                    _vram(f"step={step_idx} after_dual_grad")
                                    print(f"  [GRAPH] current_latent requires_grad={current_latent.requires_grad} "
                                          f"noise_pred requires_grad={noise_pred.requires_grad} "
                                          f"pred_x0 requires_grad={pred_x0.requires_grad}", flush=True)
                                    print(f"  [GRAPH] noise_pred grad_fn={type(noise_pred.grad_fn).__name__} "
                                          f"pred_x0 grad_fn={type(pred_x0.grad_fn).__name__}", flush=True)
                                    print(f"  [STATS] noise_pred: min={noise_pred.min().item():.3f} "
                                          f"max={noise_pred.max().item():.3f} "
                                          f"norm={noise_pred.norm().item():.1f}", flush=True)
                                    print(f"  [STATS] pred_x0: min={pred_x0.min().item():.3f} "
                                          f"max={pred_x0.max().item():.3f} "
                                          f"norm={pred_x0.norm().item():.1f} "
                                          f"mean={pred_x0.mean().item():.4f}", flush=True)
                                    print(f"  [STATS] sigma={sigma:.6f} guide_scale={sample_guide_scale}", flush=True)
                    else:
                        with torch.no_grad():
                            noise_pred_cond = model(
                                [current_latent], t=timestep_tensor, **arg_c)[0]
                            noise_pred_uncond = model(
                                [current_latent], t=timestep_tensor, **arg_null)[0]
                            noise_pred = noise_pred_uncond + sample_guide_scale * (
                                noise_pred_cond - noise_pred_uncond)
                            if fg.enabled:
                                pred_x0 = current_latent - sigma * noise_pred

                    if need_grad:
                        grad = self.compute_fg_step_loss(
                            fg, pred_x0, current_latent,
                            frame_num, step_idx, rep)
                        current_latent = self.apply_fg_gradient(
                            fg, current_latent, grad, noise_pred,
                            sigma, sigma_next, step_idx, rep)

                # Scheduler step
                if fg.enabled:
                    # sigma_next already computed above
                    with torch.no_grad():
                        euler_delta = (sigma_next - sigma) * noise_pred
                        latents = [current_latent + euler_delta]
                    if n_repeats > 0 and self.rank == 0:
                        print(f"  [FG-DEBUG] step={step_idx} sigma={sigma:.4f} sigma_next={sigma_next:.4f} "
                              f"|euler_delta|={euler_delta.norm().item():.4f} "
                              f"|current_latent|={current_latent.norm().item():.4f} "
                              f"|latent_after|={latents[0].norm().item():.4f} "
                              f"euler/fg_ratio={euler_delta.norm().item()/10.0:.2f}", flush=True)
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

            # --- FG teardown ------------------------------------------
            self.teardown_fg(fg)

            print("[T2V] Denoising loop done, preparing for VAE decode...", flush=True)

            # Detach final latent from any computation graph
            print("[T2V] Detaching latent...", flush=True)
            x0 = [latents[0].detach().clone()]
            del latents
            print("[T2V] Latent detached.", flush=True)

            # Skip transformer offload to CPU — .cpu() on large models triggers
            # SIGKILL in this container environment. Instead, keep the active
            # transformer on GPU (~53 GB) and decode VAE alongside it.
            # 96 GB GPU can hold transformer (53) + VAE (1) + decode workspace (~20).

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

        del noise
        del sample_scheduler
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
