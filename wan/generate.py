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


def _validate_params(p: GenerateParams):
    """Validate a GenerateParams instance."""
    assert p.ckpt_dir, "Please specify the checkpoint directory."
    assert p.task in WAN_CONFIGS, f"Unsupport task: {p.task}"

    if p.task == "i2v-A14B" and p.image is None:
        assert p.prompt, "T2V-like mode (no image) requires a prompt."

    assert 0.0 <= p.ifedit.tld_threshold_ratio <= 1.0, \
        "ifedit.tld_threshold_ratio must be in [0, 1]."
    assert p.ifedit.tld_step_k >= 1, "ifedit.tld_step_k must be >= 1."
    assert p.loopless.shift_skip >= 0, "loopless.shift_skip must be >= 0."
    assert p.loopless.shift_stop_step >= 0, "loopless.shift_stop_step must be >= 0."
    assert not (p.ifedit.use_tld and p.loopless.enable), \
        "ifedit.use_tld and loopless.enable are mutually exclusive."
    if 's2v' not in p.task:
        assert p.size in SUPPORTED_SIZES[p.task], \
            f"Unsupport size {p.size} for task {p.task}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames of video are generated. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=None,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--ifedit_use_tld",
        action="store_true",
        default=False,
        help="[IF-Edit] Enable Temporal Latent Dropout.")
    parser.add_argument(
        "--ifedit_tld_threshold_ratio",
        type=float,
        default=0.5,
        help="[IF-Edit] Apply TLD at sampling_steps * ratio.")
    parser.add_argument(
        "--ifedit_tld_step_k",
        type=int,
        default=2,
        help="[IF-Edit] Temporal sub-sampling stride for TLD.")
    parser.add_argument(
        "--loopless_enable",
        action="store_true",
        default=False,
        help="[Loopless] Enable Mobius-like latent temporal shift for seamless loop generation.")
    parser.add_argument(
        "--loop_shift_skip",
        type=int,
        default=6,
        help="[Loopless] Temporal latent shift step in each denoising iteration.")
    parser.add_argument(
        "--loop_shift_stop_step",
        type=int,
        default=4,
        help="[Loopless] Stop latent shift in last N denoising steps. 0 means no stop.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")

    # animate
    parser.add_argument(
        "--src_root_path",
        type=str,
        default=None,
        help="The file of the process output path. Default None.")
    parser.add_argument(
        "--refert_num",
        type=int,
        default=77,
        help="How many frames used for temporal guidance. Recommended to be 1 or 5."
    )
    parser.add_argument(
        "--replace_flag",
        action="store_true",
        default=False,
        help="Whether to use replace.")
    parser.add_argument(
        "--use_relighting_lora",
        action="store_true",
        default=False,
        help="Whether to use relighting lora.")
    
    # following args only works for s2v
    parser.add_argument(
        "--num_clip",
        type=int,
        default=None,
        help="Number of video clips to generate, the whole video will not exceed the length of audio."
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to the audio file, e.g. wav, mp3")
    parser.add_argument(
        "--enable_tts",
        action="store_true",
        default=False,
        help="Use CosyVoice to synthesis audio")
    parser.add_argument(
        "--tts_prompt_audio",
        type=str,
        default=None,
        help="Path to the tts prompt audio file, e.g. wav, mp3. Must be greater than 16khz, and between 5s to 15s.")
    parser.add_argument(
        "--tts_prompt_text",
        type=str,
        default=None,
        help="Content to the tts prompt audio. If provided, must exactly match tts_prompt_audio")
    parser.add_argument(
        "--tts_text",
        type=str,
        default=None,
        help="Text wish to synthesize")
    parser.add_argument(
        "--pose_video",
        type=str,
        default=None,
        help="Provide Dw-pose sequence to do Pose Driven")
    parser.add_argument(
        "--start_from_ref",
        action="store_true",
        default=False,
        help="whether set the reference image as the starting point for generation"
    )
    parser.add_argument(
        "--infer_frames",
        type=int,
        default=80,
        help="Number of frames per clip, 48 or 80 or others (must be multiple of 4) for 14B s2v"
    )
    args = parser.parse_args()

    # Fill defaults from EXAMPLE_PROMPT for CLI usage
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
        ),
        ifedit=IFEditParams(
            use_tld=args.ifedit_use_tld,
            tld_threshold_ratio=args.ifedit_tld_threshold_ratio,
            tld_step_k=args.ifedit_tld_step_k,
        ),
        loopless=LooplessParams(
            enable=args.loopless_enable,
            shift_skip=args.loop_shift_skip,
            shift_stop_step=args.loop_shift_stop_step,
        ),
    )
    _validate_params(params)

    return params


def make_i2v_args(
    image: str,
    prompt: str,
    size: str,
    ckpt_dir: str,
    save_file: str,
    frame_num: int = 81,
    seed: int = -1,
    sampling: SamplingParams = None,
    model: ModelParams = None,
    fg: FGParams = None,
    ifedit: IFEditParams = None,
    loopless: LooplessParams = None,
) -> GenerateParams:
    """Build a GenerateParams for the I2V (or T2V-like) pipeline.

    Called by ``app.py`` to construct structured parameters.
    """
    params = GenerateParams(
        task="i2v-A14B",
        image=image,
        prompt=prompt,
        size=size,
        ckpt_dir=ckpt_dir,
        save_file=save_file,
        frame_num=frame_num,
        seed=seed,
        sampling=sampling or SamplingParams(),
        model=model or ModelParams(),
        fg=fg or FGParams(),
        ifedit=ifedit or IFEditParams(),
        loopless=loopless or LooplessParams(),
    )
    _validate_params(params)
    return params


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(p: GenerateParams):
    print("using Wan2.2....")

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if p.model.offload_model is None:
        p.model.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {p.model.offload_model}.")
    if rank == 0:
        gpu_dev = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        logging.info("Using GPU(s): %s", gpu_dev)
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)

    cfg = WAN_CONFIGS[p.task]

    logging.info(
        "Generation params: task=%s size=%s frames=%s steps=%s shift=%s guide=%s seed=%s offload=%s t5_cpu=%s image=%s save=%s",
        p.task, p.size, p.frame_num, p.sampling.steps, p.sampling.shift,
        p.sampling.guide_scale, p.seed, p.model.offload_model, p.model.t5_cpu,
        p.image, p.save_file,
    )

    if dist.is_initialized():
        base_seed = [p.seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        p.seed = base_seed[0]

    logging.info("Input prompt received (len=%s)", len(p.prompt) if p.prompt else 0)
    img = None
    if p.image is not None:
        img = Image.open(p.image).convert("RGB")
        logging.info(f"Input image: {p.image}")

    if "t2v" in p.task:
        logging.info("Creating WanT2V pipeline.")
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=p.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_cpu=p.model.t5_cpu,
        )

        logging.info(f"Generating video ...")
        video = wan_t2v.generate(p)
    else:
        if img is None:
            # No input image: use T2V pipeline with T2V checkpoint
            logging.info("No input image — switching to WanT2V pipeline.")
            t2v_cfg = WAN_CONFIGS["t2v-A14B"]

            # Resolve T2V checkpoint dir (sibling of I2V ckpt_dir)
            t2v_ckpt_dir = os.path.join(
                os.path.dirname(p.ckpt_dir.rstrip("/")), "Wan2.2-T2V-A14B")
            if not os.path.isdir(os.path.join(t2v_ckpt_dir, "low_noise_model")):
                raise FileNotFoundError(
                    f"T2V checkpoint not found at {t2v_ckpt_dir}. "
                    f"Run `python model_utils.py` to download it.")
            logging.info(f"T2V checkpoint dir: {t2v_ckpt_dir}")

            # Override sampling params with T2V-specific config
            t2v_params = GenerateParams(
                task="t2v-A14B",
                prompt=p.prompt,
                n_prompt=p.n_prompt,
                size=p.size,
                ckpt_dir=t2v_ckpt_dir,
                save_file=p.save_file,
                frame_num=p.frame_num,
                seed=p.seed,
                sampling=SamplingParams(
                    solver=p.sampling.solver,
                    steps=p.sampling.steps,
                    shift=t2v_cfg.sample_shift,
                    guide_scale=t2v_cfg.sample_guide_scale,
                ),
                model=p.model,
                fg=p.fg,
            )

            wan_t2v = wan.WanT2V(
                config=t2v_cfg,
                checkpoint_dir=t2v_ckpt_dir,
                device_id=device,
                rank=rank,
                t5_cpu=p.model.t5_cpu,
            )
            video = wan_t2v.generate(t2v_params)
        else:
            logging.info("Creating WanI2V pipeline.")
            wan_i2v = wan.WanI2V(
                config=cfg,
                checkpoint_dir=p.ckpt_dir,
                device_id=device,
                rank=rank,
                t5_cpu=p.model.t5_cpu,
            )
            logging.info("Generating video ...")
            if p.fg.enable:
                logging.info(
                    "Frame Guidance enabled: loss=%s, fixed_frames=%s, lr=%s, travel_time=%s, downscale=%s",
                    p.fg.loss_fn, p.fg.fixed_frames, p.fg.guidance_lr,
                    p.fg.travel_time, p.fg.downscale_factor,
                )
            video = wan_i2v.generate(p, img)

    if rank == 0:
        if p.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = p.prompt.replace(" ", "_").replace("/",
                                                                     "_")[:50]
            suffix = '.mp4'
            p.save_file = f"{p.task}_{p.size.replace('*','x') if sys.platform=='win32' else p.size}_1_{formatted_prompt}_{formatted_time}" + suffix

        logging.info(f"Saving generated video to {p.save_file}")
        save_video(
            tensor=video[None],
            save_file=p.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
    del video

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    params = _parse_args()
    # Auto-download checkpoint if needed
    from model_utils import ensure_wan_checkpoint
    if os.environ.get("WAN2_CKPT_DIR"):
        params.ckpt_dir = os.environ.get("WAN2_CKPT_DIR").strip()
    params.ckpt_dir = ensure_wan_checkpoint(params.task, params.ckpt_dir)
    generate(params)
