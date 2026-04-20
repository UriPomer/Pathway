# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import re
import subprocess
import sys
import warnings
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

if not os.environ.get("CUDA_VISIBLE_DEVICES") and os.environ.get("WORLD_SIZE", "1") == "1":
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0:
            cand = []
            for line in out.stdout.strip().splitlines():
                m = re.match(r"\s*(\d+)\s*,\s*(\d+)", line)
                if m:
                    idx, used = int(m.group(1)), int(m.group(2))
                    if used < 8000:
                        cand.append((idx, used))
            if cand:
                cand.sort(key=lambda x: x[1])
                os.environ["CUDA_VISIBLE_DEVICES"] = str(cand[0][0])
    except Exception:
        pass

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.configs.param_types import (
    FGParams, GenerateParams, IFEditParams, LooplessParams, ModelParams, SamplingParams,
)
from wan.distributed.util import init_distributed_group
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import merge_video_audio, save_video, str2bool


EXAMPLE_PROMPT = {
    "t2v-A14B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "i2v-A14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
    "ti2v-5B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "animate-14B": {
        "prompt": "视频中的人在做动作",
        "video": "",
        "pose": "",
        "mask": "",
    },
    "s2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
        "audio":
            "examples/talk.wav",
        "tts_prompt_audio":
            "examples/zero_shot_prompt.wav",
        "tts_prompt_text":
            "希望你以后能够做的比我还好呦。",
        "tts_text":
            "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
    },
}


def _validate_params(p):
    "Validate a GenerateParams instance."
    assert p.ckpt_dir, "Please specify the checkpoint directory."
    assert p.task in WAN_CONFIGS, f"Unsupport task: {p.task}"

    if p.task == "i2v-A14B" and p.image is None:
        assert p.prompt, "T2V-like mode (no image) requires a prompt."

    assert 0.0 <= p.ifedit.tld_threshold_ratio <= 1.0,         "ifedit.tld_threshold_ratio must be in [0, 1]."
    assert p.ifedit.tld_step_k >= 1, "ifedit.tld_step_k must be >= 1."
    assert p.loopless.shift_skip >= 0, "loopless.shift_skip must be >= 0."
    assert p.loopless.shift_stop_step >= 0, "loopless.shift_stop_step must be >= 0."
    assert not (p.ifedit.use_tld and p.loopless.enable),         "ifedit.use_tld and loopless.enable are mutually exclusive."
    if 's2v' not in p.task:
        assert p.size in SUPPORTED_SIZES[p.task],             f"Unsupport size {p.size} for task {p.task}"


def _validate_backend_params(p):
    "Validate backend-specific parameters."
    pass

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument("--task", type=str, default="t2v-A14B", choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument("--backend", type=str, default="wan22", choices=["wan22", "diffusers"],
        help="Backend: 'wan22' (full features) or 'diffusers' (standard format, single GPU)")
    parser.add_argument("--size", type=str, default="1280*720", choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video.")
    parser.add_argument("--frame_num", type=int, default=None, help="Number of frames")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Checkpoint directory.")
    parser.add_argument("--offload_model", type=str2bool, default=None, help="Offload model to CPU.")
    parser.add_argument("--t5_cpu", action="store_true", default=False, help="Place T5 on CPU.")
    parser.add_argument("--save_file", type=str, default=None, help="Save file path.")
    parser.add_argument("--prompt", type=str, default=None, help="The prompt.")
    parser.add_argument("--base_seed", type=int, default=-1, help="The seed.")
    parser.add_argument("--image", type=str, default=None, help="Input image path.")
    parser.add_argument("--sample_solver", type=str, default='unipc', choices=['unipc', 'dpm++'], help="Solver.")
    parser.add_argument("--sample_steps", type=int, default=None, help="Sampling steps.")
    parser.add_argument("--sample_shift", type=float, default=None, help="Sampling shift.")
    parser.add_argument("--sample_guide_scale", type=float, default=None, help="Guidance scale.")
    parser.add_argument("--ifedit_use_tld", action="store_true", default=False, help="Enable TLD.")
    parser.add_argument("--loopless_enable", action="store_true", default=False, help="Enable Loopless.")
    args = parser.parse_args()

    if args.task in EXAMPLE_PROMPT:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
            args.image = EXAMPLE_PROMPT[args.task]["image"]

    cfg = WAN_CONFIGS[args.task]
    params = GenerateParams(
        task=args.task,
        image=args.image,
        prompt=args.prompt or "",
        size=args.size,
        ckpt_dir=args.ckpt_dir or "",
        save_file=args.save_file,
        frame_num=args.frame_num if args.frame_num is not None else cfg.frame_num,
        seed=args.base_seed,
        sampling=SamplingParams(
            solver=args.sample_solver,
            steps=args.sample_steps if args.sample_steps is not None else cfg.sample_steps,
            shift=args.sample_shift if args.sample_shift is not None else cfg.sample_shift,
            guide_scale=args.sample_guide_scale if args.sample_guide_scale is not None else cfg.sample_guide_scale,
        ),
        model=ModelParams(
            offload_model=args.offload_model if args.offload_model is not None else True,
            t5_cpu=args.t5_cpu,
            backend=args.backend,
        ),
        ifedit=IFEditParams(use_tld=args.ifedit_use_tld),
        loopless=LooplessParams(enable=args.loopless_enable),
    )
    _validate_params(params)
    return params



def make_i2v_args(image, prompt, size, ckpt_dir, save_file, frame_num=81, seed=-1,
    sampling=None, model=None, fg=None, ifedit=None, loopless=None):
    params = GenerateParams(
        task="i2v-A14B", image=image, prompt=prompt, size=size,
        ckpt_dir=ckpt_dir, save_file=save_file, frame_num=frame_num, seed=seed,
        sampling=sampling or SamplingParams(),
        model=model or ModelParams(),
        fg=fg or FGParams(),
        ifedit=ifedit or IFEditParams(),
        loopless=loopless or LooplessParams(),
    )
    _validate_params(params)
    return params


def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)



def _generate_wan22(p):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank

    if p.model.offload_model is None:
        p.model.offload_model = False if world_size > 1 else True

    if world_size > 1 and rank == 0:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

    cfg = WAN_CONFIGS[p.task]
    logging.info("Generation params: task=%s size=%s frames=%s steps=%s", 
        p.task, p.size, p.frame_num, p.sampling.steps)

    if dist.is_initialized():
        base_seed = [p.seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        p.seed = base_seed[0]

    img = None
    if p.image is not None:
        img = Image.open(p.image).convert("RGB")
        logging.info(f"Input image: {p.image}")

    if "t2v" in p.task:
        logging.info("Creating WanT2V pipeline.")
        wan_t2v = wan.WanT2V(
            config=cfg, checkpoint_dir=p.ckpt_dir, device_id=device,
            rank=rank, t5_cpu=p.model.t5_cpu,
        )
        video = wan_t2v.generate(p)
    else:
        if img is None:
            logging.info("No input image — switching to WanT2V pipeline.")
            t2v_cfg = WAN_CONFIGS["t2v-A14B"]
            t2v_ckpt_dir = os.path.join(os.path.dirname(p.ckpt_dir.rstrip("/")), "Wan2.2-T2V-A14B")
            if not os.path.isdir(os.path.join(t2v_ckpt_dir, "low_noise_model")):
                raise FileNotFoundError(f"T2V checkpoint not found at {t2v_ckpt_dir}")
            logging.info(f"T2V checkpoint dir: {t2v_ckpt_dir}")

            t2v_params = GenerateParams(
                task="t2v-A14B", prompt=p.prompt, n_prompt=p.n_prompt, size=p.size,
                ckpt_dir=t2v_ckpt_dir, save_file=p.save_file, frame_num=p.frame_num, seed=p.seed,
                sampling=SamplingParams(
                    solver=p.sampling.solver, steps=p.sampling.steps,
                    shift=t2v_cfg.sample_shift, guide_scale=t2v_cfg.sample_guide_scale,
                ),
                model=p.model, fg=p.fg,
            )
            wan_t2v = wan.WanT2V(
                config=t2v_cfg, checkpoint_dir=t2v_ckpt_dir, device_id=device,
                rank=rank, t5_cpu=p.model.t5_cpu, convert_model_dtype=True,
            )
            video = wan_t2v.generate(t2v_params)
        else:
            logging.info("Creating WanI2V pipeline.")
            wan_i2v = wan.WanI2V(
                config=cfg, checkpoint_dir=p.ckpt_dir, device_id=device,
                rank=rank, t5_cpu=p.model.t5_cpu,
            )
            logging.info("Generating video ...")
            if p.fg.enable:
                logging.info("Frame Guidance enabled: loss=%s, lr=%s", p.fg.loss_fn, p.fg.guidance_lr)
            video = wan_i2v.generate(p, img)

    return video



def _generate_diffusers(p, img):
    from diffusers import AutoencoderKLWan
    from diffusers.utils import export_to_video
    from transformers import CLIPVisionModel
    from pipelines.pipeline_wan_i2v import WanImageToVideoPipeline
    from model_utils import ensure_wan21_model

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))

    if world_size > 1:
        logging.warning("[Diffusers] Distributed generation not supported. Using rank 0 GPU.")
        if rank != 0:
            return None

    device_id = p.model.diffusers_device_id
    device = torch.device(f"cuda:{device_id}")

    logging.info(f"[Diffusers] Loading Wan 2.1 I2V model (device={device})")
    model_id = ensure_wan21_model("i2v-14B")
    logging.info(f"[Diffusers] Model ID: {model_id}")

    image_encoder = CLIPVisionModel.from_pretrained(
        model_id, subfolder="image_encoder", torch_dtype=torch.float32, ignore_mismatched_sizes=True)
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16)

    pipe = WanImageToVideoPipeline.from_pretrained(model_id, vae=vae, image_encoder=image_encoder,
        torch_dtype=torch.bfloat16)
    pipe = pipe.to(device)
    pipe.transformer.enable_gradient_checkpointing()
    logging.info("[Diffusers] Model loaded.")

    seed_val = p.seed if p.seed >= 0 else random.randint(0, 2**31 - 1)
    num_frames = p.frame_num
    target_h, target_w = 480, 832
    steps = p.sampling.steps

    # -- FG parameters (matching run_sketch_verify.py's proven approach) --
    loss_fn = "sketch"
    guidance_step = [0] * steps
    guidance_lr_list = [0.0] * steps
    fixed_frames = []
    additional_inputs = {}
    latent_downscale_factor = 4
    travel_time = (-1, -1)

    if p.fg.enable:
        loss_fn = p.fg.loss_fn
        fixed_frames = p.fg.fixed_frames if p.fg.fixed_frames else []
        guidance_lr_list = [p.fg.guidance_lr] * steps
        # Integer rep schedule matching official notebooks
        guidance_step = [5]*5 + [3]*10 + [1]*20 + [0] * max(0, steps - 35)
        guidance_step = (guidance_step + [0] * steps)[:steps]
        additional_inputs = dict(p.fg.additional_inputs) if p.fg.additional_inputs else {}
        latent_downscale_factor = p.fg.downscale_factor
        travel_time = p.fg.travel_time if p.fg.travel_time else (-1, -1)
        logging.info("[Diffusers] FG: loss=%s lr=%s frames=%s downscale=%d travel_time=%s",
            loss_fn, p.fg.guidance_lr, fixed_frames, latent_downscale_factor, travel_time)

    # -- Build init video --
    if img is not None:
        init_image = img.convert("RGB").resize((target_w, target_h), Image.BILINEAR)
    else:
        init_image = Image.new("RGB", (target_w, target_h), color=(0, 0, 0))
    video_init = [init_image] * num_frames

    # -- TLD / Loopless params --
    tld_enable = bool(p.ifedit.use_tld)
    loopless_enable = bool(p.loopless.enable)
    if tld_enable:
        logging.info("[Diffusers] TLD: threshold=%.2f step_k=%d",
            p.ifedit.tld_threshold_ratio, p.ifedit.tld_step_k)
    if loopless_enable:
        logging.info("[Diffusers] Loopless: skip=%d stop_step=%d",
            p.loopless.shift_skip, p.loopless.shift_stop_step)

    logging.info("[Diffusers] Generating: frames=%d steps=%d guide_scale=%.1f seed=%d",
        num_frames, steps, p.sampling.guide_scale, seed_val)

    # Build pipe() kwargs — don't pass travel_time for sketch/style,
    # letting the pipeline use its default (0, 50).
    # This matches run_sketch_verify.py which does NOT pass travel_time.
    # Only loop/keyframe have explicit travel_time=(15,20) from official notebooks.
    pipe_kwargs = dict(
        prompt=p.prompt,
        video=video_init,
        negative_prompt=p.n_prompt or "low quality, blurry, camera motion, camera shake, pan, zoom",
        num_videos_per_prompt=1,
        num_inference_steps=steps,
        guidance_scale=p.sampling.guide_scale,
        generator=torch.Generator(device=device).manual_seed(seed_val),
        # FG parameters
        loss_fn=loss_fn,
        fixed_frames=fixed_frames,
        guidance_step=guidance_step,
        guidance_lr=guidance_lr_list,
        additional_inputs=additional_inputs if additional_inputs else None,
        latent_downscale_factor=latent_downscale_factor,
        # Spatial
        height=target_h,
        width=target_w,
        num_frames=num_frames,
        # TLD
        tld_enable=tld_enable,
        tld_threshold_ratio=float(p.ifedit.tld_threshold_ratio),
        tld_step_k=int(p.ifedit.tld_step_k),
        # Loopless
        loopless_enable=loopless_enable,
        loopless_shift_skip=int(p.loopless.shift_skip),
        loopless_shift_stop_step=int(p.loopless.shift_stop_step),
    )
    # Only pass travel_time for loop mode (official: (15,20)).
    # For sketch/style, let pipe use its default (0,50) — matches run_sketch_verify.py.
    if p.fg.enable and p.fg.loss_fn == "loop":
        pipe_kwargs["travel_time"] = travel_time

    video = pipe(**pipe_kwargs).frames[0]

    # Convert output to list of PIL Images
    if isinstance(video, (list, tuple)) and len(video) > 0:
        if isinstance(video[0], Image.Image):
            frames = video
        else:
            import numpy as np
            frames = [Image.fromarray(v) if isinstance(v, np.ndarray) else v for v in video]
    else:
        frames = video

    logging.info(f"[Diffusers] Video generated: {len(frames)} frames")
    return frames



def generate(p):
    rank = int(os.getenv("RANK", 0))
    _init_logging(rank)
    logging.info(f"[Backend] Selected backend: {p.model.backend}")
    _validate_backend_params(p)
    
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    device = int(os.getenv("LOCAL_RANK", 0))
    
    if rank == 0:
        gpu_dev = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        logging.info("Using GPU(s): %s", gpu_dev)
    
    if p.model.backend == "wan22":
        logging.info("[Wan22] Initializing Wan 2.2 native backend...")
        print("using Wan2.2....")
        
        img = None
        if p.image is not None:
            img = Image.open(p.image).convert("RGB")
            logging.info(f"Input image: {p.image}")
        
        video = _generate_wan22(p)
        
        if rank == 0 and video is not None:
            if p.save_file is None:
                formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                formatted_prompt = p.prompt.replace(" ", "_").replace("/", "_")[:50]
                p.save_file = f"{p.task}_{p.size.replace('*','x')}_1_{formatted_prompt}_{formatted_time}.mp4"
            
            logging.info(f"Saving generated video to {p.save_file}")
            cfg = WAN_CONFIGS[p.task]
            save_video(tensor=video[None], save_file=p.save_file, fps=cfg.sample_fps,
                nrow=1, normalize=True, value_range=(-1, 1))
        
        del video
        
    elif p.model.backend == "diffusers":
        logging.info("[Diffusers] Initializing Wan 2.1 diffusers backend...")
        
        if world_size > 1:
            logging.warning("[Diffusers] Distributed generation not supported. Using rank 0.")
            if rank != 0:
                return
        
        img = None
        if p.image is not None:
            img = Image.open(p.image).convert("RGB")
            logging.info(f"Input image: {p.image}")
        
        frames = _generate_diffusers(p, img)
        
        if rank == 0 and frames is not None:
            if p.save_file is None:
                formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                formatted_prompt = p.prompt.replace(" ", "_").replace("/", "_")[:50]
                p.save_file = f"wan21_i2v_{formatted_prompt}_{formatted_time}.mp4"
            
            logging.info(f"Saving generated video to {p.save_file}")
            from diffusers.utils import export_to_video
            os.makedirs(os.path.dirname(p.save_file) or ".", exist_ok=True)
            export_to_video(frames, p.save_file, fps=16)
        
        del frames
    
    else:
        raise ValueError(f"Unknown backend: {p.model.backend}")
    
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    params = _parse_args()
    from model_utils import ensure_wan_checkpoint
    if os.environ.get("WAN2_CKPT_DIR"):
        params.ckpt_dir = os.environ.get("WAN2_CKPT_DIR").strip()
    
    if params.model.backend == "wan22":
        params.ckpt_dir = ensure_wan_checkpoint(params.task, params.ckpt_dir)
    
    generate(params)

