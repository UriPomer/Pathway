"""Local patched diffusers pipelines.

Contains a modified WanImageToVideoPipeline that supports `condition` in
callback_on_step_end, enabling Temporal Latent Dropout (TLD) without
rewriting the denoising loop.
"""

from .pipeline_wan_i2v import WanImageToVideoPipeline

__all__ = ["WanImageToVideoPipeline"]
