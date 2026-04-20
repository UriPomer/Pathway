"""Frame Guidance (FG) shared utilities for I2V and T2V pipelines.

This module centralises all FG-related helpers so that WanI2V and WanT2V
can inherit them via FGMixin instead of duplicating code.
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as _F
import torchvision.transforms as T
from PIL import Image as PILImage

# CSD model setup lives in the frame-guidance repo
_FRAME_GUIDANCE_DIR = Path(__file__).resolve().parents[2] / "frame-guidance"
if str(_FRAME_GUIDANCE_DIR) not in sys.path:
    sys.path.insert(0, str(_FRAME_GUIDANCE_DIR))

from pipelines.utils.models import setup_csd  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FG_VALID_LOSS_FNS = {"style", "sketch", "loop"}

# ---------------------------------------------------------------------------
# Pure utility functions
# ---------------------------------------------------------------------------


def downscale_spatial(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Downsample only the spatial (H, W) dims of a 4-D ``[C,T,H,W]`` or
    5-D ``[B,C,T,H,W]`` tensor."""
    has_batch = (x.ndim == 5)
    if not has_batch:
        x = x.unsqueeze(0)
    B, C, T_, H, W = x.shape
    x = x.permute(0, 2, 1, 3, 4).reshape(B * T_, C, H, W)
    x = _F.interpolate(x, scale_factor=1 / factor, mode='bilinear',
                        align_corners=False)
    x = x.reshape(B, T_, C, x.shape[-2], x.shape[-1]).permute(0, 2, 1, 3, 4)
    if not has_batch:
        x = x.squeeze(0)
    return x


def get_sigma_at_step(scheduler, step_idx: int) -> torch.Tensor:
    """Extract sigma value at a given step index from a flow-matching scheduler."""
    sigmas = getattr(scheduler, 'sigmas', None)
    if sigmas is not None and step_idx < len(sigmas):
        return sigmas[step_idx].to(scheduler.timesteps.device)
    if step_idx >= len(scheduler.timesteps):
        return torch.tensor(0.0)
    idx = scheduler.timesteps[step_idx].item()
    return torch.tensor(idx / 1000.0)


def weighted_mse_loss(pred, target, weight):
    return (weight * (pred - target) ** 2).mean()


def load_fg_image(
    path: str,
    device: torch.device,
    target_hw: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Load an image file to a ``[1, 3, 1, H, W]`` float tensor in [-1, 1]."""
    img = PILImage.open(path).convert("RGB")
    if target_hw is not None:
        img = img.resize((target_hw[1], target_hw[0]), PILImage.BILINEAR)
    arr = T.ToTensor()(img) * 2.0 - 1.0
    return arr.unsqueeze(0).unsqueeze(2).to(device)


# ---------------------------------------------------------------------------
# Differentiable augmenter
# ---------------------------------------------------------------------------


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
        sobel = torch.stack([sobel_x, sobel_y])
        self.register_buffer("sobel", sobel.unsqueeze(1))

    def forward(self, x: torch.Tensor, mode: str = "scribble") -> torch.Tensor:
        if mode == "scribble":
            g = (x * self.rgb2gray).sum(1, keepdim=True)
            gxgy = _F.conv2d(g, self.sobel, padding=1)
            grad_mag = torch.sqrt(gxgy.square().sum(1, keepdim=True) + 1e-6)
            scribble = torch.tanh(2.5 * grad_mag)
            return scribble.repeat(1, 3, 1, 1)
        raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# FGConfig — single source of truth for all FG runtime state
# ---------------------------------------------------------------------------


@dataclass
class FGConfig:
    """Bundles every piece of FG state so ``generate()`` only passes one object."""

    enabled: bool = False
    loss_fn: str = "style"
    fixed_frames: List[int] = field(default_factory=list)
    frame_weights: Dict[int, float] = field(default_factory=dict)
    step_schedule: List[int] = field(default_factory=list)
    lr_schedule: List[float] = field(default_factory=list)
    loss_ctx: Dict[str, Any] = field(default_factory=dict)
    travel_time: Tuple[int, int] = (-1, -1)
    guidance_lr: float = 5.0
    grad_ckpt_enabled: bool = False
    mode: str = "t2v"  # "i2v" or "t2v"
    additional_inputs: Optional[Dict] = None


# ---------------------------------------------------------------------------
# FGMixin — mixed into WanI2V / WanT2V
# ---------------------------------------------------------------------------


class FGMixin:
    """Frame Guidance methods shared by both I2V and T2V pipelines.

    Requires the host class to expose:
        self.device, self.param_dtype, self.rank,
        self.vae (Wan2_1_VAE), self.high_noise_model, self.low_noise_model
    """

    # ---- public entry-points used by generate() --------------------------

    def setup_fg(
        self,
        *,
        fg_params,  # FGParams from configs.param_types
        offload_model: bool,
        frame_num: int,
        total_steps: int,
        target_h: int,
        target_w: int,
        mode: str,  # "i2v" | "t2v"
    ) -> FGConfig:
        """One-shot FG initialisation.  Returns an ``FGConfig`` that the
        denoising loop passes around instead of a bag of loose variables."""

        # Store on self so private helpers can read them
        self.fg_offload_model = offload_model
        self.fg_downscale_factor = fg_params.downscale_factor

        if not fg_params.enable:
            return FGConfig(
                enabled=False,
                step_schedule=[0] * total_steps,
                lr_schedule=[0.0] * total_steps,
            )

        fg_loss_fn = fg_params.loss_fn
        fg_guidance_lr = fg_params.guidance_lr
        fg_additional_inputs = fg_params.additional_inputs

        # --- validate -------------------------------------------------
        if fg_loss_fn not in FG_VALID_LOSS_FNS:
            raise ValueError(
                f"fg_loss_fn must be one of {sorted(FG_VALID_LOSS_FNS)}, "
                f"got '{fg_loss_fn}'"
            )

        # --- gradient checkpointing -----------------------------------
        grad_ckpt = False
        for m in [self.high_noise_model, self.low_noise_model]:
            if m is not None:
                m.gradient_checkpointing = True
                grad_ckpt = True
        if grad_ckpt and self.rank == 0:
            print("[FG] Gradient checkpointing enabled for transformer",
                  flush=True)

        # --- effective fixed frames -----------------------------------
        fixed_frames = _compute_fixed_frames(
            fg_loss_fn, fg_params.fixed_frames, frame_num, mode)

        # --- per-frame weights ----------------------------------------
        frame_weights = _compute_frame_weights(
            fg_loss_fn, fixed_frames, frame_num, mode)

        # --- loss context (CSD / scribble / loop) ---------------------
        loss_ctx: Dict[str, Any] = {}
        if fixed_frames:
            loss_ctx = self._prepare_fg_loss_context(
                fg_loss_fn, fg_additional_inputs, None, target_h, target_w)

        # --- step / lr schedules --------------------------------------
        if fixed_frames:
            step_schedule, lr_schedule = _build_schedules(
                total_steps, fg_guidance_lr)
        else:
            step_schedule = [0] * total_steps
            lr_schedule = [0.0] * total_steps

        # --- summary log ----------------------------------------------
        if self.rank == 0:
            guided = sum(1 for s in step_schedule if s > 0)
            total_reps = sum(step_schedule)
            print(f"[FG] Schedule: {guided}/{total_steps} steps guided, "
                  f"{total_reps} total reps, schedule={step_schedule}",
                  flush=True)
            print(f"[FG] fixed_frames={fixed_frames} "
                  f"frame_weights={frame_weights} lr={fg_guidance_lr}",
                  flush=True)
            # VAE scale parameters
            scale = self.vae.scale
            print(f"[FG] VAE scale[0] (mean): {scale[0].tolist()}", flush=True)
            print(f"[FG] VAE scale[1] (1/std): {scale[1].tolist()}", flush=True)
            print(f"[FG] VAE model dtype: {next(self.vae.model.parameters()).dtype}", flush=True)
            # Transformer model info
            h_dtype = next(self.high_noise_model.parameters()).dtype
            l_dtype = next(self.low_noise_model.parameters()).dtype
            h_params = sum(p.numel() for p in self.high_noise_model.parameters()) / 1e9
            l_params = sum(p.numel() for p in self.low_noise_model.parameters()) / 1e9
            print(f"[FG] high_noise_model: dtype={h_dtype} params={h_params:.2f}B", flush=True)
            print(f"[FG] low_noise_model: dtype={l_dtype} params={l_params:.2f}B", flush=True)
            print(f"[FG] param_dtype={self.param_dtype} downscale={self.fg_downscale_factor}", flush=True)

        return FGConfig(
            enabled=True,
            loss_fn=fg_loss_fn,
            fixed_frames=fixed_frames,
            frame_weights=frame_weights,
            step_schedule=step_schedule,
            lr_schedule=lr_schedule,
            loss_ctx=loss_ctx,
            travel_time=fg_params.travel_time,
            guidance_lr=fg_guidance_lr,
            grad_ckpt_enabled=grad_ckpt,
            mode=mode,
            additional_inputs=fg_additional_inputs,
        )

    def teardown_fg(self, fg: FGConfig) -> None:
        """Post-loop cleanup: disable grad-ckpt, release CSD."""
        if fg.grad_ckpt_enabled:
            for m in [self.high_noise_model, self.low_noise_model]:
                if m is not None:
                    m.gradient_checkpointing = False
        if fg.enabled and hasattr(self, '_fg_csd') and self._fg_csd is not None:
            del self._fg_csd
            self._fg_csd = None
            torch.cuda.empty_cache()

    def compute_fg_step_loss(
        self,
        fg: FGConfig,
        pred_x0: torch.Tensor,
        current_latent: torch.Tensor,
        frame_num: int,
        step_idx: int,
        rep: int,
    ) -> torch.Tensor:
        """Compute FG loss for one repetition, call backward, print diagnostics.

        Returns the detached gradient tensor (``current_latent.grad``).
        The caller is responsible for applying the gradient update.
        """
        with torch.enable_grad():
            if fg.loss_fn == "loop":
                total_loss = self._compute_fg_loss(
                    fg.loss_fn, pred_x0, None, fg.fixed_frames,
                    fg.loss_ctx, fg.additional_inputs, frame_num,
                    step_idx, rep)
            else:
                if fg.loss_fn == "style":
                    sample_frames = random.sample(
                        fg.fixed_frames,
                        min(4, len(fg.fixed_frames)))
                else:
                    sample_frames = fg.fixed_frames
                total_loss = 0.0
                for ff in sample_frames:
                    ff_loss = self._compute_fg_loss(
                        fg.loss_fn, pred_x0, ff, fg.fixed_frames,
                        fg.loss_ctx, fg.additional_inputs, frame_num,
                        step_idx, rep)
                    if ff_loss is not None:
                        w = (fg.frame_weights.get(ff, 1.0)
                             if fg.loss_fn == "style" else 1.0)
                        total_loss = total_loss + w * ff_loss

            if isinstance(total_loss, float):
                # No valid loss computed (all ff_loss were None)
                print(f"  [FG] step={step_idx} rep={rep} WARNING: no valid loss", flush=True)
                return torch.zeros_like(current_latent)

            total_loss.backward()

        print(f"  [FG] step={step_idx} rep={rep} loss={total_loss.item():.6f} "
              f"|current_latent.grad|="
              f"{current_latent.grad.norm().item():.6f}", flush=True)

        # --- diagnostics (first rep only) -----------------------------
        if self.rank == 0 and rep == 0:
            g = current_latent.grad
            per_frame_norms = [g[:, ti, :, :].norm().item()
                               for ti in range(g.shape[1])]
            print(f"  [FG-DIAG] grad per-frame norms: "
                  f"{[f'{n:.4f}' for n in per_frame_norms]}", flush=True)
            total_els = g.numel()
            near_zero = (g.abs() < 1e-8).sum().item()
            print(f"  [FG-DIAG] grad sparsity: {near_zero}/{total_els} "
                  f"({100*near_zero/total_els:.1f}%) near-zero", flush=True)
            print(f"  [FG-DIAG] grad min={g.min().item():.6f} "
                  f"max={g.max().item():.6f} "
                  f"mean={g.mean().item():.6f} "
                  f"std={g.std().item():.6f}", flush=True)
            del g

        del total_loss, pred_x0

        self._offload_fg_models_after_backward(fg.loss_fn)
        torch.cuda.empty_cache()

        grad = current_latent.grad.detach().clone()
        current_latent.grad = None
        return grad

    def apply_fg_gradient(
        self,
        fg: FGConfig,
        current_latent: torch.Tensor,
        grad: torch.Tensor,
        noise_pred: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        step_idx: int,
        rep: int,
    ) -> torch.Tensor:
        """Normalise gradient, optionally apply temporal mask, update latent.

        Returns the updated ``current_latent`` (detached).
        """
        # --- I2V temporal gradient mask (protect early frames) --------
        if fg.mode == "i2v" and fg.loss_fn == "style" and grad.dim() >= 2:
            T = grad.shape[1]
            tw = torch.linspace(0, 1, T, device=grad.device,
                                dtype=grad.dtype) ** 2
            grad = grad * tw.reshape(1, T, 1, 1)

        grad_norm = grad.norm(2)
        rho = 1.0 / grad_norm if grad_norm > 0 else 0.0
        latent_before = current_latent.detach().clone()

        with torch.no_grad():
            in_travel = fg.travel_time[0] <= step_idx <= fg.travel_time[1]
            lr = fg.lr_schedule[step_idx]
            if in_travel:
                pred_x0_no = (current_latent.detach()
                              - sigma * noise_pred.detach())
                noise_g = torch.randn_like(current_latent)
                current_latent = (sigma * noise_g
                                  + (1 - sigma) * pred_x0_no)

            # Simple constant-lr update (matching official pipeline)
            # The sigma schedule is forced to shift=1.0 when FG is enabled,
            # ensuring uniform Euler steps that don't overpower guidance.
            current_latent = current_latent - lr * rho * grad

        fg_delta = (current_latent - latent_before).norm().item()
        euler_magnitude = (
            abs((sigma_next - sigma).item())
            * noise_pred.detach().norm().item()
        )
        print(f"  [FG] step={step_idx} rep={rep} "
              f"|grad|={grad_norm.item():.4f} "
              f"lr={lr:.4f} rho={rho:.4f} |fg_delta|={fg_delta:.4f} "
              f"|euler_mag|={euler_magnitude:.4f} "
              f"|latent|={current_latent.norm().item():.4f} "
              f"travel={in_travel}", flush=True)

        del grad
        torch.cuda.empty_cache()
        return current_latent

    # ---- private helpers (used by the public methods above) -----------

    def _prepare_fg_loss_context(
        self,
        fg_loss_fn: str,
        fg_additional_inputs: Optional[Dict],
        fg_video: Optional[torch.Tensor],
        target_h: int,
        target_w: int,
    ) -> Dict[str, Any]:
        """Prepare pre-computed context for the chosen FG loss function."""
        if fg_additional_inputs is None:
            fg_additional_inputs = {}
        ctx: Dict[str, Any] = {}
        device = self.device

        if fg_loss_fn == "style":
            style_image = fg_additional_inputs.get("style_image")
            assert style_image is not None, \
                "style loss requires 'style_image' in fg_additional_inputs"
            if not hasattr(self, '_fg_csd') or self._fg_csd is None:
                self._fg_csd, self._fg_preprocess = setup_csd(device=device)
                import torchvision
                self._fg_clip_preprocess_tensor = (
                    torchvision.transforms.Compose([
                        torchvision.transforms.Resize(
                            224,
                            interpolation=torchvision.transforms
                            .InterpolationMode.BICUBIC),
                        torchvision.transforms.CenterCrop(224),
                        torchvision.transforms.Normalize(
                            mean=(0.48145466, 0.4578275, 0.40821073),
                            std=(0.26862954, 0.26130258, 0.27577711)),
                    ]))
                if self.fg_offload_model:
                    self._fg_csd.cpu()
                    torch.cuda.empty_cache()
            ctx["cosine_loss"] = nn.CosineSimilarity(dim=1)
            ctx["style_image"] = style_image
            ctx["style_weight"] = fg_additional_inputs.get("style_weight", 1.0)
            ctx["content_image"] = fg_additional_inputs.get(
                "content_image", None)
            ctx["content_weight"] = fg_additional_inputs.get(
                "content_weight", 0.0)

        elif fg_loss_fn == "sketch":
            sketch_pil = fg_additional_inputs.get("sketch_image")
            assert sketch_pil is not None, \
                "sketch loss requires 'sketch_image' (PIL Image)"
            if not hasattr(self, '_fg_aug') or self._fg_aug is None:
                self._fg_aug = (DifferentiableAugmenter()
                                .to(device).requires_grad_(False))
            # Convert sketch to Sobel-compatible target:
            # - Resize to target resolution
            # - To tensor [0,1], grayscale, normalize
            # - Do NOT apply Sobel (sketch IS already edge-like)
            # - Replicate to 3-ch to match Sobel output shape
            sketch_resized = sketch_pil.resize(
                (target_w, target_h), PILImage.BILINEAR)
            sketch_tensor = T.ToTensor()(sketch_resized).to(device)  # [3, H, W]
            gray = sketch_tensor.mean(dim=0, keepdim=True)  # [1, H, W]
            gray = gray / (gray.max() + 1e-8)  # normalize to [0,1]
            sketch_target = gray.repeat(3, 1, 1).unsqueeze(0)  # [1, 3, H, W]
            if self.fg_downscale_factor > 1:
                sketch_target = _F.interpolate(
                    sketch_target,
                    scale_factor=1 / self.fg_downscale_factor,
                    mode='bilinear', align_corners=False)
            ctx["sketch_target"] = sketch_target.to(self.param_dtype)
            # Mask: 1 where sketch has content, 0 elsewhere (with dilation)
            sketch_mask = (gray > 0.05).float().unsqueeze(0)  # [1, 1, H, W]
            kernel_size = 11
            sketch_mask = _F.max_pool2d(
                sketch_mask, kernel_size=kernel_size,
                stride=1, padding=kernel_size // 2)
            if self.fg_downscale_factor > 1:
                sketch_mask = _F.interpolate(
                    sketch_mask,
                    scale_factor=1 / self.fg_downscale_factor,
                    mode='bilinear', align_corners=False)
            ctx["sketch_mask"] = sketch_mask.to(self.param_dtype)

        elif fg_loss_fn == "loop":
            pass

        return ctx

    def _offload_fg_models_after_backward(self, fg_loss_fn: str) -> None:
        """Offload FG auxiliary models (VAE, CSD) to CPU after backward."""
        if not self.fg_offload_model:
            return
        self.vae.model.cpu()
        if fg_loss_fn == "style" and hasattr(self, '_fg_csd'):
            self._fg_csd.cpu()
        torch.cuda.empty_cache()

    def _compute_fg_loss(
        self,
        fg_loss_fn: str,
        pred_x0: torch.Tensor,
        ff: Optional[int],
        fg_fixed_frames: list,
        fg_loss_ctx: Dict,
        fg_additional_inputs: Optional[Dict],
        frame_num: int,
        step_idx: int,
        rep_idx: int,
    ):
        """Compute FG loss for a single fixed frame or loop loss."""
        offload = self.fg_offload_model
        _diag = (self.rank == 0 and rep_idx == 0 and (step_idx <= 3 or step_idx % 10 == 0))  # verbose diagnostics

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
                    pair = downscale_spatial(pair, self.fg_downscale_factor)
                if offload:
                    self.vae.model.to(self.device)
                with torch.amp.autocast('cuda', enabled=False):
                    decoded = self.vae.model.decode(
                        pair.float().unsqueeze(0), self.vae.scale
                    ).float().squeeze(0)
                pred_f = decoded[:, rel:rel + 1]
                predicted_images.append(pred_f)
                del pair, decoded
            loss = _F.mse_loss(predicted_images[0], predicted_images[1])
            del predicted_images
            return loss

        # Single-frame losses (style / sketch)
        if ff is not None and ff <= 0:
            return None

        lf = (ff - 1) // 4 + 1
        pair = pred_x0[:, lf - 1:lf + 1]
        if self.fg_downscale_factor > 1:
            pair = downscale_spatial(pair, self.fg_downscale_factor)
        if offload:
            self.vae.model.to(self.device)

        # --- Diagnostic: pred_x0 slice stats ---
        if _diag and ff == fg_fixed_frames[0]:
            print(f"    [DIAG] pred_x0 shape={pred_x0.shape} "
                  f"requires_grad={pred_x0.requires_grad} "
                  f"grad_fn={type(pred_x0.grad_fn).__name__ if pred_x0.grad_fn else 'None'}", flush=True)
            print(f"    [DIAG] pair shape={pair.shape} "
                  f"min={pair.min().item():.3f} max={pair.max().item():.3f} "
                  f"mean={pair.mean().item():.3f} std={pair.std().item():.3f}", flush=True)

        with torch.amp.autocast('cuda', enabled=False):
            pair_f32 = pair.float()
            decoded = self.vae.model.decode(
                pair_f32.unsqueeze(0), self.vae.scale
            ).squeeze(0)
            decoded = decoded.float()

        # --- Diagnostic: VAE decode output ---
        if _diag and ff == fg_fixed_frames[0]:
            print(f"    [DIAG] decoded shape={decoded.shape} "
                  f"requires_grad={decoded.requires_grad} "
                  f"grad_fn={type(decoded.grad_fn).__name__ if decoded.grad_fn else 'None'}", flush=True)
            print(f"    [DIAG] decoded min={decoded.min().item():.3f} "
                  f"max={decoded.max().item():.3f} "
                  f"mean={decoded.mean().item():.3f} "
                  f"std={decoded.std().item():.3f}", flush=True)

        rel = (ff - 1) % 4 + 1
        pred_f = decoded[:, rel:rel + 1]
        del pair

        if fg_loss_fn == "style":
            pred_f_sq = pred_f[:, 0].unsqueeze(0)  # [1, 3, H, W]
            if offload and hasattr(self, '_fg_csd'):
                self._fg_csd.to(self.device)

            # --- Diagnostic: CSD input stats ---
            if _diag and ff == fg_fixed_frames[0]:
                print(f"    [DIAG] pred_f_sq shape={pred_f_sq.shape} "
                      f"min={pred_f_sq.min().item():.3f} "
                      f"max={pred_f_sq.max().item():.3f} "
                      f"requires_grad={pred_f_sq.requires_grad}", flush=True)

            with torch.amp.autocast('cuda', enabled=False):
                pred_f_input = self._fg_clip_preprocess_tensor(
                    pred_f_sq.float())

                # --- Diagnostic: after CLIP preprocess ---
                if _diag and ff == fg_fixed_frames[0]:
                    print(f"    [DIAG] clip_input shape={pred_f_input.shape} "
                          f"min={pred_f_input.min().item():.3f} "
                          f"max={pred_f_input.max().item():.3f} "
                          f"requires_grad={pred_f_input.requires_grad} "
                          f"grad_fn={type(pred_f_input.grad_fn).__name__ if pred_f_input.grad_fn else 'None'}", flush=True)

                _, content_emb, style_emb = self._fg_csd(pred_f_input)

                # --- Diagnostic: CSD output ---
                if _diag and ff == fg_fixed_frames[0]:
                    print(f"    [DIAG] style_emb shape={style_emb.shape} "
                          f"norm={style_emb.norm().item():.4f} "
                          f"requires_grad={style_emb.requires_grad} "
                          f"grad_fn={type(style_emb.grad_fn).__name__ if style_emb.grad_fn else 'None'}", flush=True)

            with torch.no_grad():
                if "style_embed" not in fg_additional_inputs:
                    with torch.amp.autocast('cuda', enabled=False):
                        _, _, style_emb_ref = self._fg_csd(
                            self._fg_preprocess(
                                fg_loss_ctx["style_image"]
                            ).unsqueeze(0).to(self.device))
                    fg_additional_inputs["style_embed"] = style_emb_ref

                    # --- Diagnostic: ref embedding (only first time) ---
                    if _diag:
                        print(f"    [DIAG] style_emb_ref shape={style_emb_ref.shape} "
                              f"norm={style_emb_ref.norm().item():.4f}", flush=True)

                style_emb_ref = fg_additional_inputs["style_embed"].to(
                    self.device)
                if (fg_loss_ctx.get("content_image") is not None
                        and fg_loss_ctx.get("content_weight", 0) > 0):
                    if "content_embed" not in fg_additional_inputs:
                        with torch.amp.autocast('cuda', enabled=False):
                            _, content_emb_ref, _ = self._fg_csd(
                                self._fg_preprocess(
                                    fg_loss_ctx["content_image"]
                                ).unsqueeze(0).to(self.device))
                        fg_additional_inputs["content_embed"] = (
                            content_emb_ref)

            cos = fg_loss_ctx["cosine_loss"]
            style_sim = cos(style_emb, style_emb_ref)
            print(f"    style_similarity: {style_sim.item():.3f}", flush=True)
            loss = fg_loss_ctx["style_weight"] * (1 - style_sim)

            # --- Diagnostic: loss grad_fn chain ---
            if _diag and ff == fg_fixed_frames[0]:
                print(f"    [DIAG] loss={loss.item():.4f} "
                      f"requires_grad={loss.requires_grad} "
                      f"grad_fn={type(loss.grad_fn).__name__ if loss.grad_fn else 'None'}", flush=True)
                # Walk grad_fn chain (first 8 nodes)
                node = loss.grad_fn
                chain = []
                for _ in range(8):
                    if node is None:
                        break
                    chain.append(type(node).__name__)
                    parents = node.next_functions
                    node = parents[0][0] if parents else None
                print(f"    [DIAG] grad_fn chain: {' → '.join(chain)}", flush=True)

            if (fg_loss_ctx.get("content_image") is not None
                    and fg_loss_ctx.get("content_weight", 0) > 0):
                content_emb_ref = fg_additional_inputs["content_embed"].to(
                    self.device)
                content_sim = cos(content_emb, content_emb_ref)
                loss = loss + fg_loss_ctx["content_weight"] * (
                    1 - content_sim)

        elif fg_loss_fn == "sketch":
            # Normalize VAE output to [0,1] before Sobel edge detection
            pred_frame_01 = pred_f[:, 0].unsqueeze(0).float()
            pred_frame_01 = pred_frame_01.clamp(-1, 1)
            pred_frame_01 = (pred_frame_01 + 1.0) / 2.0
            sketch_obs = self._fg_aug(pred_frame_01, mode="scribble")
            target = fg_loss_ctx["sketch_target"].float()
            mask = fg_loss_ctx["sketch_mask"].float()
            # Resize target/mask to match decoded spatial size
            # (pixel_res / downscale != lat_h / downscale * vae_stride)
            obs_h, obs_w = sketch_obs.shape[-2], sketch_obs.shape[-1]
            if target.shape[-2] != obs_h or target.shape[-1] != obs_w:
                target = _F.interpolate(
                    target, size=(obs_h, obs_w),
                    mode='bilinear', align_corners=False)
                mask = _F.interpolate(
                    mask, size=(obs_h, obs_w),
                    mode='bilinear', align_corners=False)
            # Masked MSE: only compute loss where sketch has content
            diff_sq = (sketch_obs - target) ** 2
            loss = (diff_sq * mask).sum() / (mask.sum() + 1e-8)

        del decoded, pred_f
        return loss

    # ---- (end of _compute_fg_loss) -----------------------------------


# ---------------------------------------------------------------------------
# Module-private helpers (used by setup_fg)
# ---------------------------------------------------------------------------


def _compute_fixed_frames(
    loss_fn: str,
    user_frames: Optional[List[int]],
    frame_num: int,
    mode: str,
) -> List[int]:
    """Determine which pixel-space frames get FG loss."""
    if loss_fn == "loop":
        return [0, frame_num - 1]

    if loss_fn not in ("style", "sketch"):
        return user_frames or []

    # Sketch always constrains the last frame only
    if loss_fn == "sketch":
        return [frame_num - 1]

    # User provided explicit frames
    if user_frames is not None and len(user_frames) > 0:
        return user_frames

    last = frame_num - 1
    n_fixed = 4

    if mode == "i2v":
        # I2V: concentrate on back half (avoid fighting first-frame cond)
        mid = last // 2
        span = last - mid
        frames = []
        for i in range(n_fixed):
            t_ratio = (i + 1) / n_fixed
            ff = mid + int(span * t_ratio * t_ratio)
            ff = min(ff, last)
            frames.append(ff)
        frames[-1] = last
    else:
        # T2V: uniform distribution matching the paper
        frames = [int(last * i / n_fixed) for i in range(1, n_fixed + 1)]

    # Deduplicate while preserving order
    seen: set = set()
    return [f for f in frames if not (f in seen or seen.add(f))]


def _compute_frame_weights(
    loss_fn: str,
    fixed_frames: List[int],
    frame_num: int,
    mode: str,
) -> Dict[int, float]:
    """Per-frame loss weights (only for style)."""
    if loss_fn != "style" or not fixed_frames:
        return {}
    if mode == "i2v":
        last = frame_num - 1
        return {ff: (ff / last if last > 0 else 1.0) for ff in fixed_frames}
    # T2V: equal weight
    return {ff: 1.0 for ff in fixed_frames}


def _build_schedules(
    total_steps: int,
    guidance_lr: float,
) -> Tuple[List[int], List[float]]:
    """Build step-schedule and lr-schedule aligned with the paper.

    Official (50 steps): [0]*2 + [5]*8 + [3]*20 + [1]*10 + [0]*10
    We scale proportionally and keep the last ~20% as cooldown (no guidance),
    matching the official approach.  Low-sigma guidance is ineffective and
    can cause latent divergence.
    """
    skip = 2
    s = max(1, round(total_steps * 0.15))    # heavy (5 reps)
    m = max(1, round(total_steps * 0.40))    # medium (3 reps)
    lo = max(1, round(total_steps * 0.20))   # light (1 rep)
    r = max(0, total_steps - skip - s - m - lo)  # cooldown (0 reps)
    step_sched = ([0] * skip + [5] * s + [3] * m + [1] * lo + [0] * r)
    step_sched = step_sched[:total_steps]
    lr_sched = [guidance_lr] * total_steps
    return step_sched, lr_sched
