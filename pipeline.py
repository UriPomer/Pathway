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

try:  # 仅在可用时导入，避免硬依赖错误
    from huggingface_hub import HfApi, HfHubHTTPError, snapshot_download  # type: ignore
except Exception:  # pragma: no cover
    HfApi = None  # type: ignore
    HfHubHTTPError = Exception  # type: ignore
    snapshot_download = None  # type: ignore

class Frame2FramePipeline:
    def __init__(self,
                 device: str = "cuda",
                 model_name: str = "zai-org/CogVideoX1.5-5B-I2V",
                 model_cache_dir: Optional[str] = None,
                 hf_token: Optional[str] = None,
                 validate_access: bool = True,
                 local_model_subdir: Optional[str] = None,
                 prefer_local: bool = True):
        """Frame2FramePipeline

        Args:
            device: 运行设备
            model_name: HF 仓库名
            model_cache_dir: 模型主缓存目录（默认 ./model）
            hf_token: 手动指定 token（否则自动从环境与登录缓存获取）
            validate_access: 是否预检访问权限
            local_model_subdir: 指定子目录直接本地加载（存在 model_index.json）
            prefer_local: 若找到本地目录则不再触发远程下载
        """
        self.device = device
        self.model_name = model_name

        # 解析 / 发现 token
        env_token = (hf_token or os.environ.get("HUGGINGFACE_HUB_TOKEN") or
                     os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN") or
                     os.environ.get("HUGGINGFACE_TOKEN"))
        if not env_token:
            default_token_path = os.path.expanduser("~/.cache/huggingface/token")
            try:
                if os.path.isfile(default_token_path):
                    with open(default_token_path, 'r') as f:
                        lines = [ln.strip() for ln in f if ln.strip()]
                        if lines:
                            env_token = lines[-1]
                            print("[INFO] 读取到本地登录 token (来自 ~/.cache/huggingface/token)")
            except Exception:
                pass
        self.hf_token = env_token

        # 目录确定
        env_override = os.environ.get("FRAME2FRAME_MODEL_DIR")
        if env_override:
            model_cache_dir = env_override
        if model_cache_dir is None:
            project_root = os.path.abspath(os.path.dirname(__file__))
            model_cache_dir = os.path.join(project_root, "model")
        model_cache_dir = os.path.abspath(model_cache_dir)
        os.makedirs(model_cache_dir, exist_ok=True)
        self.model_cache_dir = model_cache_dir
        print(f"[INFO] Using local model cache dir: {self.model_cache_dir}")

        print(f"[INFO] diffusers version: {diffusers_version}")
        print(f"[INFO] Loading video model: {model_name}")
        if self.hf_token:
            print(f"[INFO] 使用 HuggingFace token (长度={len(self.hf_token)}, 前8位={self.hf_token[:8]}) 进行鉴权")
        else:
            print("[WARN] 未检测到 token；若模型为 gated/private 将触发 401。可设置 HUGGINGFACE_HUB_TOKEN")

        # 访问预检
        if validate_access and self.hf_token and HfApi is not None:
            api = HfApi()
            try:
                info = api.model_info(model_name, token=self.hf_token, timeout=15)  # type: ignore
                print(f"[INFO] 访问验证成功: gated={getattr(info,'gated',False)}, private={getattr(info,'private',False)}, sha={getattr(info,'sha','N/A')[:8]}")
            except HfHubHTTPError as he:  # type: ignore
                status = getattr(getattr(he, 'response', None), 'status_code', None)
                if status in (401, 403):
                    raise RuntimeError("预检失败(状态码=%s)。请确认模型访问授权与 token 权限。" % status) from he
                else:
                    print(f"[WARN] 预检失败但继续: {he}")
            except Exception as ve:
                print(f"[WARN] 预检异常(忽略): {ve}")

        # 分片完整性工具
        def _list_missing_shards(root_dir: str):
            missing: List[str] = []
            for dirpath, _, filenames in os.walk(root_dir):
                for fn in filenames:
                    if fn.endswith('.index.json'):
                        idx_path = os.path.join(dirpath, fn)
                        try:
                            import json
                            with open(idx_path, 'r') as f:
                                data = json.load(f)
                            for shard in set((data.get('weight_map') or {}).values()):
                                shard_path = shard if os.path.isabs(shard) else os.path.join(dirpath, shard)
                                if not os.path.exists(shard_path):
                                    missing.append(shard_path)
                        except Exception:
                            continue
            return sorted(set(missing))

        # 预下载（可断点续）
        if snapshot_download is not None and not prefer_local:  # 若已经 prefer_local 就先尝试本地
            try:
                print("[INFO] 预下载模型快照 (resume_download=True)...")
                snap = snapshot_download(
                    repo_id=model_name,
                    cache_dir=self.model_cache_dir,
                    token=self.hf_token,
                    resume_download=True,
                    local_files_only=False,
                    force_download=False
                )
                print(f"[INFO] 快照路径: {snap}")
                miss = _list_missing_shards(snap)
                if miss:
                    print(f"[WARN] 缺失分片 {len(miss)} 个，尝试补拉...")
                    for p in miss:
                        try:
                            rel = os.path.relpath(p, snap)
                        except ValueError:
                            rel = p
                        try:
                            snapshot_download(repo_id=model_name, cache_dir=self.model_cache_dir, token=self.hf_token,
                                              resume_download=True, allow_patterns=rel, local_files_only=False)
                        except Exception as se:
                            print(f"[WARN] 补拉 {rel} 失败: {se}")
                    miss2 = _list_missing_shards(snap)
                    if miss2:
                        print("[ERROR] 仍缺分片: ")
                        for m in miss2:
                            print("  -", os.path.relpath(m, snap))
                    else:
                        print("[INFO] 缺失分片已补齐。")
            except Exception as pe:
                print(f"[WARN] 预下载阶段异常: {pe}")

        # 加载逻辑（本地优先）
        start_time = time.time()
        candidate_dirs: List[str] = []
        if local_model_subdir:
            candidate_dirs.append(os.path.join(self.model_cache_dir, local_model_subdir))
        candidate_dirs.append(self.model_cache_dir)
        candidate_dirs.append(os.path.join(self.model_cache_dir, 'cogvideox'))
        local_dir_found = None
        if prefer_local:
            for d in candidate_dirs:
                if os.path.isfile(os.path.join(d, 'model_index.json')):
                    local_dir_found = d
                    break
        if local_dir_found:
            print(f"[INFO] 本地发现模型目录：{local_dir_found} -> 直接加载 (local_files_only)")
            load_kwargs = dict(torch_dtype=torch.float16, trust_remote_code=True, local_files_only=True)
            self.video_pipe = DiffusionPipeline.from_pretrained(local_dir_found, **load_kwargs)
        else:
            load_kwargs = dict(torch_dtype=torch.float16, trust_remote_code=True, cache_dir=self.model_cache_dir)
            if self.hf_token:
                load_kwargs['token'] = self.hf_token
            try:
                self.video_pipe = DiffusionPipeline.from_pretrained(model_name, **load_kwargs)
            except AttributeError as e:
                if 'CogVideoX' in str(e):
                    raise RuntimeError("缺少 CogVideoX 自定义 pipeline 类，建议升级 diffusers 或检查 trust_remote_code。") from e
                raise
            except Exception as e:
                msg = str(e)
                if ('401' in msg or 'Unauthorized' in msg):
                    if not self.hf_token:
                        raise RuntimeError("401 未授权：缺少 token，先执行 hf auth login 或导出 HUGGINGFACE_HUB_TOKEN。") from e
                    else:
                        raise RuntimeError("401 未授权：token 无效或未获许可；尝试 classic read token / 重新申请访问。") from e
                raise

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
