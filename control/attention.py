from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Union

import math

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import AttentionProcessor


@dataclass
class CrossAttentionCache:
    """Container holding cached cross-attention probability tensors."""

    maps: Dict[str, torch.Tensor] = field(default_factory=dict)

    def to(self, device: torch.device, dtype: torch.dtype) -> "CrossAttentionCache":
        return CrossAttentionCache(
            maps={name: tensor.to(device=device, dtype=dtype) for name, tensor in self.maps.items()}
        )


@dataclass
class TokenGatingConfig:
    """Configures which prompt tokens should retain source or target attention."""

    target_token_indices: Iterable[int] = ()
    frozen_token_indices: Iterable[int] = ()
    blend_bias: float = 1.0

    def build(self, key_length: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.ones(key_length, dtype=dtype, device=device)
        for idx in self.frozen_token_indices:
            if 0 <= idx < key_length:
                mask[idx] = 0.0
        for idx in self.target_token_indices:
            if 0 <= idx < key_length:
                mask[idx] = self.blend_bias
        return mask


class ControlledAttnProcessor(AttentionProcessor):
    """Drop-in attention processor that routes attention probabilities through a controller."""

    def __init__(self, name: str, controller: "CrossAttentionController"):
        super().__init__()
        self.name = name
        self.controller = controller

    def __call__(  # type: ignore[override]
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        attention_probs = self.controller.modify(self.name, attention_probs, key_length=key.shape[-2])

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            hidden_states = attn.to_out[1](hidden_states, temb=temb)

        hidden_states = hidden_states + residual

        return hidden_states


class CrossAttentionController:
    """High-level coordinator that records and blends cross-attention maps."""

    def __init__(self, *, verbose: bool = False):
        self.verbose = verbose
        self._original_processors: Dict[str, AttentionProcessor] = {}
        self._mode: str = "idle"
        self._active_tag: Optional[str] = None
        self._record_store: Dict[str, Dict[str, torch.Tensor]] = {}
        self._inject_store: Dict[str, Dict[str, torch.Tensor]] = {}
        self._token_gating: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ hooks
    def install(self, unet) -> None:
        """Swap the U-Net cross-attention processors with controlled variants."""
        if self._original_processors:
            return
        processors = dict(unet.attn_processors)
        replacement: Dict[str, AttentionProcessor] = {}
        for name, processor in processors.items():
            if "attn2" not in name:
                replacement[name] = processor
                continue
            replacement[name] = ControlledAttnProcessor(name, controller=self)
            self._original_processors[name] = processor
        unet.set_attn_processor(replacement)

    def uninstall(self, unet) -> None:
        """Restore the original attention processors."""
        if not self._original_processors:
            return
        processors = dict(unet.attn_processors)
        for name, original in self._original_processors.items():
            processors[name] = original
        unet.set_attn_processor(processors)
        self._original_processors.clear()

    # ---------------------------------------------------------------- recording
    def start_record(self, tag: str) -> None:
        self._mode = "record"
        self._active_tag = tag
        self._record_store[tag] = {}

    def stop_record(self) -> CrossAttentionCache:
        cache = CrossAttentionCache(maps=self._record_store.get(self._active_tag or "", {}).copy())
        self._mode = "idle"
        self._active_tag = None
        return cache

    # ---------------------------------------------------------------- injection
    def prepare_injection(
        self,
        source: CrossAttentionCache,
        target: CrossAttentionCache,
        *,
        alpha: Optional[Dict[str, torch.Tensor]] = None,
        token_gating: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        self._inject_store.clear()
        self._token_gating = token_gating or {}
        for name, src_tensor in source.maps.items():
            tgt_tensor = target.maps.get(name)
            if tgt_tensor is None:
                continue
            if alpha and name in alpha:
                cur_alpha = alpha[name]
            else:
                cur_alpha = None
            self._inject_store[name] = {
                "src": src_tensor,
                "tgt": tgt_tensor,
                "alpha": cur_alpha,
            }
        self._mode = "inject"

    def clear(self) -> None:
        self._mode = "idle"
        self._inject_store.clear()
        self._token_gating.clear()

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _resize_mask(mask: torch.Tensor, query_length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(0)
        mask = mask.to(device=device, dtype=dtype)

        height = max(1, int(math.sqrt(query_length)))
        width = max(1, math.ceil(query_length / height))
        resized = F.interpolate(mask, size=(height, width), mode="bilinear", align_corners=False)
        flattened = resized.flatten(-2)
        if flattened.shape[-1] > query_length:
            flattened = flattened[..., :query_length]
        elif flattened.shape[-1] < query_length:
            pad = query_length - flattened.shape[-1]
            flattened = F.pad(flattened, (0, pad), value=0.0)
        return flattened

    def _compute_alpha(
        self,
        payload_alpha: Optional[torch.Tensor],
        query_length: int,
        device: torch.device,
        dtype: torch.dtype,
        batch_heads: int,
    ) -> Optional[torch.Tensor]:
        if payload_alpha is None:
            return None
        base = self._resize_mask(payload_alpha, query_length, device, dtype)
        base = base.repeat(batch_heads, 1)
        base = base.unsqueeze(-1)
        return base

    # ---------------------------------------------------------------- modifier
    def modify(self, name: str, attention_probs: torch.Tensor, *, key_length: int) -> torch.Tensor:
        if self._mode == "record" and self._active_tag:
            self._record_store[self._active_tag][name] = attention_probs.detach().cpu()
            return attention_probs

        if self._mode != "inject":
            return attention_probs

        payload = self._inject_store.get(name)
        if payload is None:
            return attention_probs

        src = payload["src"].to(attention_probs.device, attention_probs.dtype)
        tgt = payload["tgt"].to(attention_probs.device, attention_probs.dtype)

        if src.shape != attention_probs.shape or tgt.shape != attention_probs.shape:
            # Fall back to the live probabilities when shapes do not match.
            if self.verbose:
                print(f"[CrossAttentionController] shape mismatch on {name}, skipping injection.")
            return attention_probs

        blend = self._compute_alpha(
            payload.get("alpha"),
            query_length=attention_probs.shape[1],
            device=attention_probs.device,
            dtype=attention_probs.dtype,
            batch_heads=attention_probs.shape[0],
        )
        if blend is None:
            fused = tgt
        else:
            fused = (1.0 - blend) * src + blend * tgt

        gate_obj = self._token_gating.get(name)
        if isinstance(gate_obj, TokenGatingConfig):
            gate = gate_obj.build(
                key_length,
                device=attention_probs.device,
                dtype=attention_probs.dtype,
            )
        else:
            gate = gate_obj
        if gate is not None:
            if gate.shape[-1] != key_length:
                gate = gate[..., :key_length]
            gate = gate.to(attention_probs.device, attention_probs.dtype)
            if gate.ndim == 1:
                gate = gate.unsqueeze(0).unsqueeze(0)
            elif gate.ndim == 2:
                gate = gate.unsqueeze(0)
            gate = gate.repeat(fused.shape[0], fused.shape[1], 1)
            fused = fused * gate
            fused = fused / fused.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        return fused
