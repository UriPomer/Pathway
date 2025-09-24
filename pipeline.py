import torch
try:  # compatibility for type checker
    from diffusers import DiffusionPipeline, __version__ as diffusers_version  # type: ignore
except Exception:  # pragma: no cover
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore
    from diffusers import __version__ as diffusers_version  # type: ignore
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import time
import math
from typing import List, Dict, Tuple, Optional

class Frame2FramePipeline:
    def __init__(self,
                 device: str = "cuda",
                 model_name: str = "zai-org/CogVideoX1.5-5B-I2V",
                 model_cache_dir: Optional[str] = None):
        """Frame2FramePipeline

        Args:
            device: 运行设备。
            model_name: HuggingFace 上的模型名称。
            model_cache_dir: 可选，自定义缓存目录。如果为 None（默认），使用系统默认
                             `~/.cache/huggingface`。可通过环境变量
                             `FRAME2FRAME_MODEL_DIR` 强制指定。
        """
        self.device = device
        self.model_name = model_name
        env_override = os.environ.get("FRAME2FRAME_MODEL_DIR")
        if env_override:
            model_cache_dir = env_override
        if model_cache_dir is not None:
            model_cache_dir = os.path.abspath(model_cache_dir)
            os.makedirs(model_cache_dir, exist_ok=True)
        self.model_cache_dir = model_cache_dir

        print(f"[INFO] diffusers version: {diffusers_version}")
        print(f"[INFO] Loading video model: {model_name}")

        # 尝试下载 +加载
        start_time = time.time()
        video_kwargs = dict(torch_dtype=torch.float16)
        if self.model_cache_dir is not None:
            video_kwargs["cache_dir"] = self.model_cache_dir  # type: ignore
        self.video_pipe = DiffusionPipeline.from_pretrained(model_name, **video_kwargs)

        print(f"[INFO] Model loaded in {time.time() - start_time:.1f}s")

        # 启用优化以节省显存
        try:
            self.video_pipe.enable_sequential_cpu_offload()
            if hasattr(self.video_pipe, "vae"):
                self.video_pipe.vae.enable_slicing()
                self.video_pipe.vae.enable_tiling()
            print("[INFO] Memory optimizations (offload + slicing + tiling) enabled.")
        except Exception as e:
            print("[WARN] Some memory optimizations failed:", e)

        # CLIP 模型加载
        print("[INFO] Loading CLIP model for frame selection...")
        clip_kwargs = {}
        if self.model_cache_dir is not None:
            clip_subdir = os.path.join(self.model_cache_dir, "clip-vit-base-patch16")
            os.makedirs(clip_subdir, exist_ok=True)
            clip_kwargs["cache_dir"] = clip_subdir
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", **clip_kwargs).to(self.device)  # type: ignore
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", **clip_kwargs)

    # ------------------------------------------------------------------
    # Prompt Handling
    # ------------------------------------------------------------------

    def temporal_caption(self, prompt_text: str) -> str:
        return f"Image reference fixed. Gradually apply: {prompt_text}"

    def generate_video(self, image: Image.Image, prompt_text: str,
                       num_frames: int = 4,
                       height: int = 256,
                       width: int = 256,
                       guidance_scale: float = 6.0,
                       num_inference_steps: int = 25,
                       generator: Optional[torch.Generator] = None):
        edit_prompt = self.temporal_caption(prompt_text)
        print(f"[INFO] Generating video: {num_frames} frames, {width}×{height}, prompt: {edit_prompt}")
        # Some static analyzers fail to see DiffusionPipeline.__call__; resolve explicitly
        pipe_call = getattr(self.video_pipe, "__call__")
        result = pipe_call(
            prompt=edit_prompt,
            image=image,
            num_frames=num_frames,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        )
        if hasattr(result, "frames"):
            return result.frames
        elif isinstance(result, list):
            return result
        else:
            raise RuntimeError("Video generation result missing `.frames` or not a list.")

    def select_frame(self, frames: List[Image.Image], prompt_text: str):
        inputs = self.clip_processor(
            text=[prompt_text],
            images=frames,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        with torch.no_grad():
            out = self.clip_model(**inputs)
            logits = out.logits_per_image.squeeze(0)
            best_idx = logits.argmax().item()
        return frames[best_idx]

    # ------------------------------------------------------------------
    # Feature & Metrics
    # ------------------------------------------------------------------
    def _clip_image_features(self, frames: List[Image.Image]) -> torch.Tensor:
        proc = self.clip_processor(text=["placeholder"], images=frames, return_tensors="pt", padding=True).to(self.device)
        # clip_processor 需要 text 占位，有些版本不允许纯 image
        with torch.no_grad():
            out = self.clip_model(
                input_ids=proc.get('input_ids'),
                attention_mask=proc.get('attention_mask'),
                pixel_values=proc.get('pixel_values')
            )
        if hasattr(out, 'image_embeds') and getattr(out, 'image_embeds') is not None:
            feats = out.image_embeds
        else:  # fallback
            feats = out.vision_model_output.last_hidden_state.mean(dim=1) if hasattr(out, 'vision_model_output') else out.last_hidden_state.mean(dim=1)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats  # (N, D)

    def _clip_text_feature(self, prompt_text: str) -> torch.Tensor:
        proc = self.clip_processor(text=[prompt_text], images=[Image.new('RGB', (32,32), 'black')], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.clip_model(
                input_ids=proc.get('input_ids'),
                attention_mask=proc.get('attention_mask'),
                pixel_values=proc.get('pixel_values')
            )
        if hasattr(out, 'text_embeds') and getattr(out, 'text_embeds') is not None:
            feat = out.text_embeds[0]
        else:
            feat = out.text_model_output.last_hidden_state.mean(dim=1)[0] if hasattr(out, 'text_model_output') else out.last_hidden_state.mean(dim=1)[0]
        feat = torch.nn.functional.normalize(feat, dim=-1)
        return feat  # (D,)

    def compute_path_metrics(self, frames: List[Image.Image], prompt_text: str) -> Dict[str, object]:
        if len(frames) < 2:
            return {"path_length": 0.0, "smoothness": 0.0, "semantic_scores": [0.0], "identity_drift": [0.0]}
        feats = self._clip_image_features(frames)  # (N,D)
        text_feat = self._clip_text_feature(prompt_text)
        cos = (feats @ text_feat)  # (N,)
        # path length (L2 in feature space)
        diffs = feats[1:] - feats[:-1]
        path_length = diffs.norm(dim=-1).sum().item()
        # smoothness second finite difference
        if len(feats) > 2:
            sec = feats[2:] - 2 * feats[1:-1] + feats[:-2]
            smoothness = sec.norm(dim=-1).mean().item()
        else:
            smoothness = 0.0
        # identity drift relative to first frame
        base = feats[0]
        identity_drift = (feats - base).norm(dim=-1).tolist()
        return {
            "path_length": path_length,
            "smoothness": smoothness,
            "semantic_scores": cos.tolist(),
            "identity_drift": identity_drift
        }

    # ------------------------------------------------------------------
    # Iterative Path Construction (Simple Approximation of Paper Idea)
    # ------------------------------------------------------------------
    def iterative_edit(self,
                       image: Image.Image,
                       prompt_text: str,
                       steps: int = 6,
                       candidates_per_step: int = 4,
                       height: int = 256,
                       width: int = 256,
                       guidance_scale: float = 6.0,
                       num_inference_steps: int = 25,
                       w_sem: float = 1.0,
                       w_step: float = 0.2,
                       w_id: float = 0.1,
                       generator: Optional[torch.Generator] = None) -> Tuple[List[Image.Image], Dict[str, object]]:
        """Construct a path by sequentially sampling short candidate clips.

        Energy(frame) = w_sem * (1 - cosSim) + w_step * ||f - f_prev||_2 + w_id * ||f - f_start||_2
        (All distances in CLIP embedding space.)
        """
        path = [image]
        text_feat = self._clip_text_feature(prompt_text)
        start_feat = None
        prev_feat = None
        for t in range(1, steps + 1):
            print(f"[PATH] Step {t}/{steps} generating candidates ...")
            frames = self.generate_video(
                image=path[-1],
                prompt_text=prompt_text,
                num_frames=candidates_per_step,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator
            )
            feats = self._clip_image_features(frames)
            if start_feat is None:
                start_feat = self._clip_image_features([path[0]])[0]
            if prev_feat is None:
                prev_feat = self._clip_image_features([path[-1]])[0]
            cos = feats @ text_feat
            # L2 distances
            step_dist = (feats - prev_feat).norm(dim=-1)
            id_dist = (feats - start_feat).norm(dim=-1)
            energy = w_sem * (1 - cos) + w_step * step_dist + w_id * id_dist
            best_idx = int(energy.argmin().item())
            chosen = frames[best_idx]
            path.append(chosen)
            prev_feat = feats[best_idx]
            # small stochasticity: advance generator if provided
        metrics = self.compute_path_metrics(path, prompt_text)
        return path, metrics

    def run(self, image: Image.Image, prompt_text: str,
            num_frames: int = 4, height: int = 256, width: int = 256,
            guidance_scale: float = 6.0,
            num_inference_steps: int = 25,
            generator: Optional[torch.Generator] = None,
            save_dir: str = "outputs",
            use_iterative: bool = False,
            iterative_steps: int = 6,
            candidates_per_step: int = 4,
            w_sem: float = 1.0,
            w_step: float = 0.2,
            w_id: float = 0.1):
        os.makedirs(save_dir, exist_ok=True)
        if use_iterative:
            path, metrics = self.iterative_edit(
                image=image,
                prompt_text=prompt_text,
                steps=iterative_steps,
                candidates_per_step=candidates_per_step,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                w_sem=w_sem,
                w_step=w_step,
                w_id=w_id,
                generator=generator
            )
            best_frame = path[-1]
            best_frame.save(os.path.join(save_dir, "final_frame.png"))
            # Optionally save intermediate frames
            for i, fr in enumerate(path):
                fr.save(os.path.join(save_dir, f"path_{i:02d}.png"))
            print(f"[INFO] Iterative path saved to {save_dir} (length={len(path)})")
            # Write metrics
            try:
                import json
                with open(os.path.join(save_dir, "metrics.json"), 'w') as f:
                    json.dump(metrics, f, indent=2)
            except Exception as e:
                print("[WARN] Failed to write metrics.json:", e)
            return best_frame, path, metrics
        else:
            frames = self.generate_video(image, prompt_text,
                                         num_frames=num_frames,
                                         height=height,
                                         width=width,
                                         guidance_scale=guidance_scale,
                                         num_inference_steps=num_inference_steps,
                                         generator=generator)
            best_frame = self.select_frame(frames, prompt_text)
            best_frame.save(os.path.join(save_dir, "best_frame.png"))
            print(f"[INFO] Saved best_frame to {save_dir}/best_frame.png")
            return best_frame, frames, None
