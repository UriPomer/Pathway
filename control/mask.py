from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    # The FramePainter repository is vendored into this project. Import lazily to
    # avoid making it a hard runtime dependency when the user does not enable mask
    # control.
    from FramePainter.modules.sparse_control_encoder import SparseControlEncoder  # type: ignore
except Exception:  # pragma: no cover - best effort import
    SparseControlEncoder = None  # type: ignore


Tensorable = Union[torch.Tensor, np.ndarray, Image.Image]


def _to_tensor(data: Tensorable, *, device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert PIL/numpy/tensor inputs to a float32 tensor in [0, 1]."""
    if isinstance(data, torch.Tensor):
        tensor = data.float()
    elif isinstance(data, Image.Image):
        tensor = torch.from_numpy(np.array(data))
    else:
        tensor = torch.from_numpy(np.asarray(data))

    if tensor.ndim == 2:  # (H, W)
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim == 3 and tensor.size(-1) in (1, 3, 4):
        tensor = tensor.permute(2, 0, 1)  # C, H, W
    elif tensor.ndim == 3:
        # Interpret as (T, H, W)
        tensor = tensor.unsqueeze(1)  # T, 1, H, W
    elif tensor.ndim == 4 and tensor.shape[0] not in (1, 3, 4):
        # Assume (T, H, W, C)
        tensor = tensor.permute(0, 3, 1, 2)

    tensor = tensor.float()
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    if device is not None:
        tensor = tensor.to(device)
    return tensor.clamp(0.0, 1.0)


def _gaussian_feather(mask: torch.Tensor, radius: float) -> torch.Tensor:
    """Apply a gentle Gaussian blur to soften mask boundaries."""
    if radius <= 0:
        return mask
    # Compute kernel size as an odd integer
    kernel = int(max(1, round(radius * 4)))
    if kernel % 2 == 0:
        kernel += 1
    mask = mask.unsqueeze(0)  # add batch -> (1, C, H, W)
    grid = torch.arange(kernel, device=mask.device, dtype=mask.dtype) - kernel // 2
    g = torch.exp(-(grid ** 2) / (2 * radius ** 2))
    gaussian = (g[:, None] @ g[None, :])
    gaussian = gaussian / gaussian.sum()
    filt = gaussian.unsqueeze(0).unsqueeze(0)
    channels = mask.shape[1]
    filt = filt.repeat(channels, 1, 1, 1)
    padding = kernel // 2
    smoothed = F.conv2d(mask, filt, padding=padding, groups=channels)
    return smoothed.squeeze(0)


def _ensure_frames(mask: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Broadcast the mask to the requested number of frames."""
    if mask.ndim == 4:  # (T, C, H, W)
        if mask.size(0) == num_frames:
            return mask
        if mask.size(0) == 1:
            return mask.repeat(num_frames, 1, 1, 1)
        raise ValueError(f"Mask has {mask.size(0)} frames but {num_frames} were requested.")
    if mask.ndim == 3:  # (C, H, W)
        return mask.unsqueeze(0).repeat(num_frames, 1, 1, 1)
    raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")


@dataclass
class SparseControlConfig:
    """Configuration bundle for sparse control guidance."""

    mask: Tensorable
    feather_radius: float = 4.0
    strength: float = 1.0
    num_frames: int = 1
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None


class SparseControlAdapter:
    """Thin wrapper over the FramePainter sparse control encoder.

    The adapter normalises ROI masks, optionally feathers their edges, and downsamples
    them to the latent resolution before pushing them through the encoder. The encoded
    features are then rescaled to the current latent resolution and injected as a
    residual, i.e. h' = h + M * f_ctrl.
    """

    def __init__(self, encoder: Optional["SparseControlEncoder"] = None):
        if encoder is not None:
            self.encoder = encoder
        elif SparseControlEncoder is not None:
            self.encoder = SparseControlEncoder()
        else:
            self.encoder = None
        self._prepared_mask: Optional[torch.Tensor] = None
        self._raw_mask: Optional[torch.Tensor] = None
        self._config: Optional[SparseControlConfig] = None

    @property
    def enabled(self) -> bool:
        return self.encoder is not None and self._prepared_mask is not None

    def reset(self) -> None:
        self._prepared_mask = None
        self._raw_mask = None
        self._config = None

    def snapshot(self):
        return self._prepared_mask, self._raw_mask, self._config

    def load_snapshot(self, snapshot) -> None:
        self._prepared_mask, self._raw_mask, self._config = snapshot

    def prepare(self, config: SparseControlConfig) -> None:
        """Convert and cache the control mask for later reuse."""
        if self.encoder is None:
            raise RuntimeError(
                "SparseControlEncoder is unavailable. "
                "Ensure FramePainter/modules is on PYTHONPATH."
            )
        device = config.device or getattr(self.encoder, "device", torch.device("cpu"))
        mask = _to_tensor(config.mask, device=device)
        if mask.ndim == 4 and mask.size(1) not in (1, 3):
            # Merge potential temporal dimension if provided as frames first
            mask = mask.mean(dim=1, keepdim=True)

        if mask.ndim == 3 and mask.size(0) not in (1, 3):
            mask = mask.mean(dim=0, keepdim=True)

        if mask.size(0) == 1:
            mask = mask.repeat(3, 1, 1)
        elif mask.size(0) > 3:
            mask = mask[:3]

        mask = _gaussian_feather(mask, config.feather_radius)
        mask = mask.clamp(0.0, 1.0)
        mask = _ensure_frames(mask, config.num_frames)

        batch = 1
        mask = mask.unsqueeze(0)  # (1, T, C, H, W)

        self._prepared_mask = mask
        self._raw_mask = mask
        self._config = config

    def _resize_mask_like(self, sample: torch.Tensor) -> torch.Tensor:
        if self._prepared_mask is None:
            raise RuntimeError("SparseControlAdapter.prepare must be called before applying control.")
        batch, frames = sample.shape[:2]
        mask = self._prepared_mask.to(sample.device, sample.dtype)
        if mask.shape[1] != frames:
            mask = _ensure_frames(mask.squeeze(0), frames).unsqueeze(0)
        mask = mask.repeat(batch, 1, 1, 1, 1)
        mask_flat = mask.flatten(0, 1)
        target_hw = sample.shape[-2:]
        mask_flat = F.interpolate(mask_flat, size=target_hw, mode="bilinear", align_corners=False)
        mask = mask_flat.view(batch, frames, mask_flat.shape[1], *target_hw)
        return mask

    def inject(self, sample: torch.Tensor, timestep: Union[int, torch.Tensor]) -> torch.Tensor:
        """Apply sparse control residuals to the latent sample."""
        if not self.enabled:
            return sample
        if self._config is None:
            return sample

        mask = self._resize_mask_like(sample)
        prepared = self._raw_mask.to(sample.device, sample.dtype)
        prepared = prepared.repeat(sample.shape[0], 1, 1, 1, 1)

        encoded = self.encoder(prepared, timestep)
        features = encoded.get("output")
        scale = encoded.get("scale", 1.0)

        if isinstance(scale, (int, float)):
            scale_tensor = torch.tensor(scale, dtype=sample.dtype, device=sample.device)
        else:
            scale_tensor = scale.to(sample.device, sample.dtype)

        if features.dim() == 4:
            frames = prepared.shape[1]
            features = features.view(sample.shape[0], frames, features.shape[1], features.shape[2], features.shape[3])
        elif features.dim() != 5:
            raise RuntimeError(f"SparseControlEncoder returned unexpected shape {features.shape}.")

        features_flat = features.flatten(0, 1)
        features_flat = F.interpolate(features_flat, size=sample.shape[-2:], mode="bilinear", align_corners=False)
        features = features_flat.view_as(mask)

        while scale_tensor.ndim < features.ndim:
            scale_tensor = scale_tensor.unsqueeze(-1)

        guided = sample + self._config.strength * mask * features * scale_tensor
        return guided


class CrossAttentionMask:
    """Utility to prepare ROI alpha maps for cross-attention blending."""

    def __init__(self, roi_mask: Tensorable, num_frames: int = 1, feather_radius: float = 4.0):
        mask = _to_tensor(roi_mask)
        mask = mask.mean(dim=0, keepdim=True) if mask.ndim == 3 and mask.size(0) > 1 else mask
        if mask.ndim == 3:
            mask = _ensure_frames(mask, num_frames)
        elif mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0).repeat(num_frames, 1, 1, 1)
        elif mask.ndim == 4 and mask.shape[0] != num_frames:
            mask = _ensure_frames(mask, num_frames)
        self.mask = mask.clamp(0.0, 1.0)
        if feather_radius > 0:
            flattened = self.mask.view(-1, *self.mask.shape[-2:])
            blurred = _gaussian_feather(flattened, feather_radius)
            self.mask = blurred.view_as(self.mask).clamp(0.0, 1.0)

    def downsample(self, spatial: Tuple[int, int], *, flatten: bool = True, device=None, dtype=None) -> torch.Tensor:
        tensor = self.mask.to(device=device, dtype=dtype or torch.float32)
        tensor = tensor.view(-1, 1, *tensor.shape[-2:])
        tensor = F.interpolate(tensor, size=spatial, mode="bilinear", align_corners=False)
        if flatten:
            tensor = tensor.flatten(-2)
        return tensor


def build_alpha_map(
    roi_mask: Tensorable,
    *,
    spatial: Tuple[int, int],
    heads: int,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
    feather_radius: float = 2.0,
) -> torch.Tensor:
    """Create a broadcast-friendly alpha tensor for attention map fusion."""
    mask = CrossAttentionMask(roi_mask, num_frames=1, feather_radius=feather_radius)
    alpha = mask.downsample(spatial, device=device, dtype=dtype, flatten=True)
    alpha = alpha.unsqueeze(0)  # (1, query)
    alpha = alpha.unsqueeze(-1)  # (1, query, 1)
    alpha = alpha.repeat(batch * heads, 1, 1)
    return alpha.clamp(0.0, 1.0)
