"""
Control tooling for ROI-aware diffusion editing.

This package groups the sparse control encoder wrapper and cross-attention
controllers used to integrate mask-conditioned behaviour into the Pathway
pipeline.
"""

from .mask import SparseControlConfig, SparseControlAdapter, build_alpha_map, CrossAttentionMask
from .attention import (
    CrossAttentionController,
    CrossAttentionCache,
    TokenGatingConfig,
)

__all__ = [
    "SparseControlConfig",
    "SparseControlAdapter",
    "build_alpha_map",
    "CrossAttentionMask",
    "CrossAttentionController",
    "CrossAttentionCache",
    "TokenGatingConfig",
]
