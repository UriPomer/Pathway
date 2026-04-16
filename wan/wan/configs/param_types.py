"""Structured parameter types for the generation pipeline.

These dataclasses replace the flat ``argparse.Namespace`` that was passed
from the UI through ``generate()`` into the model's ``generate()`` method.
Each group of related parameters is bundled into its own dataclass, and
``GenerateParams`` composes them all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class FGParams:
    """Frame Guidance input parameters (UI -> generate -> model)."""

    enable: bool = False
    loss_fn: str = "style"
    fixed_frames: Optional[List[int]] = None
    guidance_lr: float = 5.0
    travel_time: Tuple[int, int] = (-1, -1)
    downscale_factor: int = 4
    additional_inputs: Optional[Dict[str, Any]] = None


@dataclass
class IFEditParams:
    """IF-Edit Temporal Latent Dropout parameters (I2V only)."""

    use_tld: bool = False
    tld_threshold_ratio: float = 0.5
    tld_step_k: int = 2


@dataclass
class LooplessParams:
    """Loopless / Mobius cinemagraph parameters (I2V only)."""

    enable: bool = False
    shift_skip: int = 6
    shift_stop_step: int = 4


@dataclass
class SamplingParams:
    """Diffusion sampler parameters."""

    solver: str = "unipc"
    steps: int = 40
    shift: float = 3.0
    guide_scale: float = 5.0


@dataclass
class ModelParams:
    """Model loading / offloading parameters."""

    offload_model: bool = True
    t5_cpu: bool = False


@dataclass
class GenerateParams:
    """Top-level generation parameters — composes all sub-groups."""

    task: str = "i2v-A14B"
    image: Optional[str] = None
    prompt: str = ""
    n_prompt: str = ""
    size: str = "1280*720"
    ckpt_dir: str = ""
    save_file: Optional[str] = None
    frame_num: int = 81
    seed: int = -1

    sampling: SamplingParams = field(default_factory=SamplingParams)
    model: ModelParams = field(default_factory=ModelParams)
    fg: FGParams = field(default_factory=FGParams)
    ifedit: IFEditParams = field(default_factory=IFEditParams)
    loopless: LooplessParams = field(default_factory=LooplessParams)
