import os
import time
from typing import List, Dict, Tuple, Optional, Any, Sequence, TYPE_CHECKING

import torch
from diffusers import DiffusionPipeline, __version__ as diffusers_version  # type: ignore
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

from vlm import (
    FrameSelector,
    FrameSelectionResult,
    TemporalPromptPlan,
    TemporalCaptionGenerator,
    build_frame_collage,
)

if TYPE_CHECKING:  # pragma: no cover
    from vlm import (  # type: ignore
        FrameSelector as FrameSelectorType,
        FrameSelectionResult as FrameSelectionResultType,
        TemporalCaptionGenerator as TemporalCaptionGeneratorType,
    )
else:
    FrameSelectorType = Any
    FrameSelectionResultType = Any
    TemporalCaptionGeneratorType = Any

from utils import pad_lr_to_720x480, crop_center_square, postprocess_to_512  # type: ignore
from huggingface_hub import HfApi, snapshot_download

class Frame2FramePipeline:
    """封装 CogVideoX 图生视频（参考帧引导）+ 简单迭代路径逻辑。

    特性:
    - 本地优先加载 (指定 cache 目录 / 子目录)
    - 兼容不同版本的参数名 (video_length / num_frames)
    - 黑帧检测 + 可选一次自动重试 (fp16 -> fp32, guidance 调整, 步数增加)
    - CLIP 用于选择最佳帧 & 路径能量评估
    - 迭代模式：逐步扩展路径 (简化的 energy 公式)
    """

    def __init__(self,
                 device: str = "cuda",
                 model_name: str = "zai-org/CogVideoX1.5-5B-I2V",
                 model_cache_dir: Optional[str] = None,
                 hf_token: Optional[str] = None,
                 validate_access: bool = True,
                 local_model_subdir: Optional[str] = None,
                 prefer_local: bool = True,
                 caption_generator: Optional[TemporalCaptionGeneratorType] = None,
                 frame_selector: Optional[FrameSelectorType] = None,
                 frame_sample_every: int = 4,
                 max_selector_tiles: Optional[int] = None):
        self.device = device
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "hf")
        os.makedirs(self.model_cache_dir, exist_ok=True)
        # 解析 token
        self.hf_token = hf_token or os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        if validate_access and self.hf_token and HfApi is not None:
            try:
                _ = HfApi().whoami(token=self.hf_token)
                print("[INFO] HF token 校验成功。")
            except Exception as e:
                print("[WARN] HF token 验证失败:", e)
        self.local_model_subdir = local_model_subdir
        self.prefer_local = prefer_local
        if caption_generator is not None:
            self.caption_generator = caption_generator
        else:
            self.caption_generator = TemporalCaptionGenerator()
        self._last_temporal_plan = None

        if frame_selector is not None:
            self.frame_selector = frame_selector
        else:
            self.frame_selector = FrameSelector()
        self.frame_sample_every = max(1, frame_sample_every)
        self.max_selector_tiles = max_selector_tiles

        # ---------------- 分片缺失检测工具 ----------------
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

        # ---------------- 预下载（若不强制本地优先） ----------------
        if snapshot_download is not None and not prefer_local:
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
                    print(f"[WARN] 缺失分片 {len(miss)} 个，尝试补拉 ...")
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
                        print("[ERROR] 仍缺分片:")
                        for m in miss2:
                            try:
                                print("  -", os.path.relpath(m, snap))
                            except ValueError:
                                print("  -", m)
                    else:
                        print("[INFO] 缺失分片已补齐。")
            except Exception as e:
                print("[WARN] 预下载阶段异常:", e)

        # ---------------- 搜索本地目录 ----------------
        candidate_dirs: List[str] = []
        if local_model_subdir:
            candidate_dirs.append(os.path.join(self.model_cache_dir, local_model_subdir))
        candidate_dirs.append(self.model_cache_dir)
        candidate_dirs.append(os.path.join(self.model_cache_dir, 'cogvideox'))
        self.local_dir_found: Optional[str] = None
        if prefer_local:
            for d in candidate_dirs:
                if os.path.isfile(os.path.join(d, 'model_index.json')):
                    self.local_dir_found = d
                    break

        # ---------------- 加载 pipeline ----------------
        def _choose_dtype():
            if os.environ.get("FRAME2FRAME_FORCE_FP32") == "1":
                return torch.float32
            return torch.float16 if torch.cuda.is_available() else torch.float32

        def _load_pipe(dtype):
            if self.local_dir_found:
                print(f"[INFO] 本地发现模型目录: {self.local_dir_found} (dtype={dtype})")
                return DiffusionPipeline.from_pretrained(self.local_dir_found, torch_dtype=dtype, trust_remote_code=True, local_files_only=True)
            load_kwargs = dict(torch_dtype=dtype, trust_remote_code=True, cache_dir=self.model_cache_dir)
            if self.hf_token:
                load_kwargs['token'] = self.hf_token
            return DiffusionPipeline.from_pretrained(self.model_name, **load_kwargs)

        start = time.time()
        self._loaded_dtype = _choose_dtype()
        print(f"[INFO] Loading pipeline with dtype={self._loaded_dtype}")
        self.video_pipe = _load_pipe(self._loaded_dtype)
        print(f"[INFO] Model loaded in {time.time() - start:.1f}s")

        # 显存优化（可通过 FRAME2FRAME_DISABLE_OFFLOAD=1 禁用）
        try:
            if os.environ.get('FRAME2FRAME_DISABLE_OFFLOAD') == '1':
                print('[INFO] Offload disabled by env FRAME2FRAME_DISABLE_OFFLOAD=1')
            else:
                self.video_pipe.enable_sequential_cpu_offload()
                if hasattr(self.video_pipe, 'vae'):
                    self.video_pipe.vae.enable_slicing(); self.video_pipe.vae.enable_tiling()
                print("[INFO] Memory optimizations enabled.")
        except Exception as e:
            print("[WARN] Memory optimization failed:", e)

        # CLIP 模型（用于选择与特征）
        print("[INFO] Loading CLIP (openai/clip-vit-base-patch16) ...")
        clip_kwargs = {}
        clip_cache = os.path.join(self.model_cache_dir, "clip-vit-base-patch16")
        os.makedirs(clip_cache, exist_ok=True)
        clip_kwargs['cache_dir'] = clip_cache
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", **clip_kwargs).to(self.device)  # type: ignore
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", use_fast=True, **clip_kwargs)

    # ---------------- Prompt 包装 ----------------
    def build_temporal_caption(self, image: Image.Image, prompt_text: str) -> str:
        prompt_text = (prompt_text or "").strip()
        
        if not prompt_text:
            raise ValueError("Target edit text cannot be empty.")

        caption_text = self.caption_generator.generate(image, prompt_text)
        self._last_temporal_plan = getattr(self.caption_generator, 'last_plan', None)
        return caption_text

    # ---------------- 视频生成核心 ----------------
    def generate_video(self, image: Image.Image, prompt_text: str,
                       num_frames: int = 49,
                       height: int = 480,
                       width: int = 720,
                       guidance_scale: float = 6.0,
                       num_inference_steps: int = 50,
                       generator: Optional[torch.Generator] = None,
                       temporal_caption: Optional[str] = None,
                       _allow_retry: bool = True,
                       _square_fallback_used: bool = False) -> List[Image.Image]:
        # --- 预处理 ---
        if image is None:
            raise ValueError("输入图像为空，请上传图像。")
        original_for_caption = image
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if (height, width) == (480, 720):
            try:
                if pad_lr_to_720x480 is not None:
                    image = pad_lr_to_720x480(image)
                else:
                    image = image.resize((width, height), Image.Resampling.LANCZOS)
            except Exception:
                pass

        edit_prompt = temporal_caption or self.build_temporal_caption(original_for_caption, prompt_text)
        print(f"[INFO] Generating: frames={num_frames}, size={width}x{height}, prompt='{edit_prompt}'")
        debug_mode = os.environ.get('FRAME2FRAME_DEBUG') == '1'

        pipe_call = getattr(self.video_pipe, '__call__')
        base_kwargs = dict(prompt=edit_prompt, image=image, height=height, width=width,
                           guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                           generator=generator)

        # 调整 generator 设备以匹配执行设备（避免 cpu tensor + cuda generator 报错）
        if generator is not None:
            exec_dev = getattr(self.video_pipe, "_execution_device", None)
            if exec_dev is not None and hasattr(generator, 'device'):
                try:
                    gen_dev = generator.device
                    if exec_dev.type == 'cpu' and gen_dev.type != 'cpu':
                        print("[DEBUG] Rebuilding generator on CPU to match execution device.")
                        try:
                            base_kwargs['generator'] = torch.Generator(device='cpu')
                        except Exception as _ge:
                            print("[DEBUG] CPU generator rebuild failed, drop generator:", _ge)
                            base_kwargs['generator'] = None
                except Exception:
                    pass

        attempt_params: List[Tuple[str, int]] = []
        varnames = getattr(getattr(pipe_call, '__code__', None), 'co_varnames', ())
        if 'num_frames' in varnames:
            attempt_params.append(('num_frames', num_frames))
        if 'video_length' in varnames:
            attempt_params.append(('video_length', num_frames))
        if not attempt_params:
            # Fallback if introspection fails
            attempt_params = [('num_frames', num_frames), ('video_length', num_frames)]

        result = None
        last_error: Optional[Exception] = None
        step_stats: List[Dict[str, float]] = [] if debug_mode else []

        # 暂不启用 callback_on_step_end（某些 diffusers 版本不支持该签名或类型检查不兼容）
        callback_fn = None

        for key, val in attempt_params:
            try:
                kw = dict(base_kwargs)
                kw[key] = val
                print(f"[DEBUG] call pipeline with {key}={val}")
                result = pipe_call(**kw)
                # Success, break the loop
                last_error = None
                break
            except TypeError as te:
                if 'unexpected keyword argument' in str(te):
                    print(f"[DEBUG] Argument '{key}' is not accepted, trying next.")
                    last_error = te
                    continue
                else:
                    # Re-raise other TypeErrors
                    raise
            except Exception as e:
                print(f"[WARN] pipeline call failed with {key}: {e}")
                last_error = e
                # Break on non-TypeError, as it's likely a deeper issue
                break
        
        if result is None:
            raise RuntimeError("底层 pipeline 调用失败 (尝试了 num_frames/video_length)") from last_error

        # ---- 解析输出 ----
        def _to_frames(obj):
            from PIL import Image as _Img; import numpy as np
            if obj is None: return []
            if isinstance(obj, list):
                merged = []
                for it in obj:
                    if isinstance(it, list): merged.extend(it)
                    else: merged.append(it)
                return [f for f in merged if isinstance(f, _Img.Image)]
            if torch.is_tensor(obj):
                t = obj
                if t.dim() == 5: t = t[0]
                frames_ = []
                for fr in t:
                    fr = fr.detach().float().cpu().clamp(0,1)
                    arr = (fr.permute(1,2,0).numpy()*255).round().astype('uint8')
                    frames_.append(_Img.fromarray(arr))
                return frames_
            return []

        frames: List[Image.Image] = []
        if hasattr(result, 'frames'): frames = _to_frames(getattr(result,'frames'))
        if not frames and hasattr(result, 'videos'): frames = _to_frames(getattr(result,'videos'))
        if not frames and hasattr(result, 'images'): frames = _to_frames(getattr(result,'images'))
        if not frames and isinstance(result, list): frames = _to_frames(result)
        if not frames:
            raise RuntimeError("未解析到任何帧")
        print(f"[DEBUG] generate_video got {len(frames)} frame(s)")

        # ---- 黑帧 / NaN 检测 + 回退策略 ----
        import numpy as np
        sample_means: List[float] = []
        sample_all_zero: List[bool] = []
        sample_has_nan: List[bool] = []
        for f in frames[:min(4, len(frames))]:
            arr = np.array(f)
            sample_means.append(float(arr.mean()))
            sample_all_zero.append(bool((arr == 0).all()))
            sample_has_nan.append(bool(np.isnan(arr).any()))
        if any(sample_has_nan):
            print('[WARN] Detected NaN in frame pixels (decoder produced invalid values).')
        if sample_means and all(m < 2 for m in sample_means):
            print('[WARN] 可能黑帧: means=', [round(m,2) for m in sample_means], 'all_zero=', sample_all_zero, 'nan_flags=', sample_has_nan)
            # 分辨率方形回退（尚未尝试且当前为宽屏）
            if _allow_retry and not _square_fallback_used and width != height:
                print('[RETRY] 方形回退尝试 480x480 ...')
                try:
                    return self.generate_video(image, prompt_text, num_frames,
                                               480, 480, guidance_scale, num_inference_steps,
                                               generator, _allow_retry=_allow_retry, _square_fallback_used=True)
                except Exception as e:
                    print('[RETRY-WARN] 方形回退失败:', e)
            if _allow_retry and self._loaded_dtype == torch.float16:
                print('[RETRY] 尝试 fp32 重载与参数调整 ...')
                os.environ['FRAME2FRAME_FORCE_FP32'] = '1'
                try:
                    reload_dir = self.local_dir_found or (os.path.isfile(os.path.join(self.model_cache_dir,'model_index.json')) and self.model_cache_dir) or self.model_name
                    self.video_pipe = DiffusionPipeline.from_pretrained(reload_dir, torch_dtype=torch.float32, trust_remote_code=True, local_files_only=bool(self.local_dir_found), cache_dir=self.model_cache_dir)
                    self._loaded_dtype = torch.float32
                except Exception as e:
                    print('[RETRY-WARN] fp32 重载失败:', e)
                try:
                    return self.generate_video(image, prompt_text, num_frames, height, width,
                                               max(2.5, guidance_scale), num_inference_steps,
                                               generator, _allow_retry=False, _square_fallback_used=_square_fallback_used)
                except Exception as e:
                    print('[RETRY-FAIL] 重试仍失败:', e)
            print('[HINT] 可尝试: 英文 prompt / 降低 guidance / 提高 steps / 确保图像非纯色 / FRAME2FRAME_FORCE_FP32=1 / FRAME2FRAME_DEBUG=1')

        # debug 输出
        if debug_mode:
            try:
                os.makedirs('outputs', exist_ok=True)
                import json
                payload = {
                    'step_stats': step_stats,
                    'sample_means': sample_means,
                    'sample_all_zero': sample_all_zero,
                    'sample_has_nan': sample_has_nan,
                    'final_frame_count': len(frames),
                    'dtype': str(self._loaded_dtype),
                    'width': width,
                    'height': height
                }
                with open('outputs/debug_stats.json','w') as f:
                    json.dump(payload, f, indent=2)
                print('[DEBUG] 写入 outputs/debug_stats.json')
            except Exception as e:
                print('[DEBUG] debug_stats 写入失败:', e)
        return frames

    # ---------------- 帧选择 ----------------
    def select_frame(self, frames: List[Image.Image], prompt_text: str):
        inputs = self.clip_processor(text=[prompt_text], images=frames, return_tensors='pt', padding=True).to(self.device)
        with torch.no_grad():
            out = self.clip_model(**inputs)
            logits = out.logits_per_image.squeeze(0)
            best_idx = logits.argmax().item()
        return frames[best_idx]

    # ---------------- 特征与指标 ----------------
    def _clip_image_features(self, frames: List[Image.Image]) -> torch.Tensor:
        flat = []
        for f in frames:
            if isinstance(f, list): flat.extend(f)
            else: flat.append(f)
        proc = self.clip_processor(text=['placeholder'], images=flat, return_tensors='pt', padding=True).to(self.device)
        with torch.no_grad():
            out = self.clip_model(input_ids=proc.get('input_ids'), attention_mask=proc.get('attention_mask'), pixel_values=proc.get('pixel_values'))
        if hasattr(out,'image_embeds') and getattr(out,'image_embeds') is not None:
            feats = out.image_embeds
        else:
            feats = out.vision_model_output.last_hidden_state.mean(dim=1) if hasattr(out,'vision_model_output') else out.last_hidden_state.mean(dim=1)
        return torch.nn.functional.normalize(feats, dim=-1)

    def _clip_text_feature(self, prompt_text: str) -> torch.Tensor:
        proc = self.clip_processor(text=[prompt_text], images=[Image.new('RGB',(32,32),'black')], return_tensors='pt', padding=True).to(self.device)
        with torch.no_grad():
            out = self.clip_model(input_ids=proc.get('input_ids'), attention_mask=proc.get('attention_mask'), pixel_values=proc.get('pixel_values'))
        if hasattr(out,'text_embeds') and getattr(out,'text_embeds') is not None:
            feat = out.text_embeds[0]
        else:
            feat = out.text_model_output.last_hidden_state.mean(dim=1)[0] if hasattr(out,'text_model_output') else out.last_hidden_state.mean(dim=1)[0]
        return torch.nn.functional.normalize(feat, dim=-1)

    def _clip_text_features(self, prompts: Sequence[str]) -> Tuple[torch.Tensor, List[str]]:
        cleaned: List[str] = []
        for prompt in prompts:
            text = (prompt or "").strip()
            if text and text not in cleaned:
                cleaned.append(text)
        if not cleaned:
            raise ValueError("At least one non-empty prompt is required for CLIP scoring.")
        dummy = Image.new('RGB', (32, 32), 'black')
        images = [dummy] * len(cleaned)
        proc = self.clip_processor(text=cleaned, images=images, return_tensors='pt', padding=True).to(self.device)
        with torch.no_grad():
            out = self.clip_model(
                input_ids=proc.get('input_ids'),
                attention_mask=proc.get('attention_mask'),
                pixel_values=proc.get('pixel_values'),
            )
        if hasattr(out, 'text_embeds') and getattr(out, 'text_embeds') is not None:
            feats = out.text_embeds
        else:
            text_output = getattr(out, 'text_model_output', None)
            if text_output is not None:
                feats = text_output.last_hidden_state.mean(dim=1)
            else:
                feats = out.last_hidden_state.mean(dim=1)
        return torch.nn.functional.normalize(feats, dim=-1), cleaned

    def _build_selection_prompts(
        self,
        prompt_text: str,
        temporal_caption: Optional[str],
        plan: Optional[TemporalPromptPlan],
    ) -> List[str]:
        prompts: List[str] = []

        def _append(text: Optional[str]) -> None:
            if not text:
                return
            cleaned = text.strip()
            if cleaned and cleaned not in prompts:
                prompts.append(cleaned)

        _append(prompt_text)
        _append(temporal_caption)
        if plan is not None:
            _append(plan.anchor)
            if plan.deltas:
                sorted_deltas = sorted(plan.deltas, key=lambda d: d.end)
                for delta in sorted_deltas:
                    delta_text = delta.description
                    if delta.focus_tokens:
                        delta_text = f"{delta_text} (focus: {', '.join(delta.focus_tokens)})"
                    if plan.anchor:
                        stage_prompt = f"{plan.anchor} -> {delta_text}"
                    else:
                        stage_prompt = delta_text
                    _append(stage_prompt)
                final_delta = sorted_deltas[-1]
                final_desc = final_delta.description
                if final_delta.focus_tokens:
                    final_desc = f"Final state of {', '.join(final_delta.focus_tokens)}: {final_delta.description}"
                else:
                    final_desc = f"Final state: {final_delta.description}"
                _append(final_desc)
            if plan.notes:
                _append(plan.notes)
        return prompts

    def _binary_search_frame_selection(
        self,
        frames: List[Image.Image],
        base_prompt: str,
        temporal_caption: Optional[str],
        plan: Optional[TemporalPromptPlan],
        threshold_ratio: float = 0.85,
    ) -> Dict[str, Any]:
        if not frames:
            return {
                'prompts': [],
                'aggregated_scores': [],
                'per_prompt_scores': [],
                'base_score': 0.0,
                'max_score': 0.0,
                'max_index': 0,
                'threshold_ratio': threshold_ratio,
                'threshold_score': None,
                'first_good_index': None,
            }

        prompts = self._build_selection_prompts(base_prompt, temporal_caption, plan)
        text_feats, prompts_used = self._clip_text_features(prompts)
        image_feats = self._clip_image_features(frames)
        score_matrix = image_feats @ text_feats.T
        aggregated = score_matrix.mean(dim=1)
        aggregated_cpu = aggregated.detach().cpu()
        aggregated_list = aggregated_cpu.tolist()
        base_score = aggregated_list[0] if aggregated_list else 0.0
        max_score = max(aggregated_list) if aggregated_list else base_score
        max_index = int(aggregated_cpu.argmax().item()) if aggregated_cpu.numel() > 0 else 0
        threshold_score: Optional[float] = None
        first_good_index: Optional[int] = None
        if aggregated_list and max_score > base_score + 1e-5:
            threshold_score = base_score + (max_score - base_score) * threshold_ratio
            running_max: List[float] = []
            current = float('-inf')
            for value in aggregated_list:
                current = max(current, value)
                running_max.append(current)
            lo, hi = 0, len(running_max) - 1
            candidate: Optional[int] = None
            while lo <= hi:
                mid = (lo + hi) // 2
                if running_max[mid] >= threshold_score:
                    candidate = mid
                    hi = mid - 1
                else:
                    lo = mid + 1
            if candidate is not None:
                for idx in range(candidate + 1):
                    if aggregated_list[idx] >= threshold_score:
                        first_good_index = idx
                        break
                if first_good_index is None:
                    first_good_index = candidate
        per_prompt_scores = score_matrix.detach().cpu().tolist()
        return {
            'prompts': prompts_used,
            'aggregated_scores': aggregated_list,
            'per_prompt_scores': per_prompt_scores,
            'base_score': base_score,
            'max_score': max_score,
            'max_index': max_index,
            'threshold_ratio': threshold_ratio,
            'threshold_score': threshold_score,
            'first_good_index': first_good_index,
        }

    def select_frame_with_vlm(self,
                               source_image: Image.Image,
                               frames: List[Image.Image],
                               prompt_text: str,
                               save_dir: str) -> Dict[str, Any]:
        collage, mapping = build_frame_collage(
            source_image,
            frames,
            sample_step=self.frame_sample_every,
            max_tiles=self.max_selector_tiles,
        )
        os.makedirs(save_dir, exist_ok=True)
        collage_path = os.path.join(save_dir, 'frame_collage.png')
        collage.save(collage_path)
        selection_result = self.frame_selector.select(collage, prompt_text, mapping)
        return {
            'collage_path': collage_path,
            'mapping': mapping,
            'selection': selection_result,
        }

    def finalize_output_frame(self,
                               frame: Image.Image,
                               save_dir: str,
                               prefix: str) -> Dict[str, Any]:
        os.makedirs(save_dir, exist_ok=True)
        if frame.mode != 'RGB':
            frame = frame.convert('RGB')

        outputs: Dict[str, Any] = {}
        full_path = os.path.join(save_dir, f'{prefix}_full.png')
        try:
            frame.save(full_path)
            outputs['full_path'] = full_path
        except Exception as exc:
            print(f"[WARN] 保存 {full_path} 失败: {exc}")

        crop_img = None
        crop_path = os.path.join(save_dir, f'{prefix}_crop480.png')
        if crop_center_square is not None:
            try:
                crop_img = crop_center_square(frame, 480)
                crop_img.save(crop_path)
                outputs['crop480_path'] = crop_path
            except Exception as exc:
                print(f"[WARN] 保存 {crop_path} 失败: {exc}")
                crop_img = None
        if crop_img is None:
            crop_img = frame

        resized_img = crop_img
        resized_path = os.path.join(save_dir, f'{prefix}_512.png')
        if postprocess_to_512 is not None:
            try:
                resized_img = postprocess_to_512(frame)
                resized_img.save(resized_path)
                outputs['resized512_path'] = resized_path
            except Exception as exc:
                print(f"[WARN] 保存 {resized_path} 失败: {exc}")
        else:
            try:
                resized_img = crop_img.resize((512, 512), Image.Resampling.LANCZOS)
                resized_img.save(resized_path)
                outputs['resized512_path'] = resized_path
            except Exception as exc:
                print(f"[WARN] 保存 {resized_path} 失败: {exc}")

        outputs['image'] = resized_img
        outputs['base_frame_size'] = frame.size
        return outputs

    def compute_path_metrics(self, frames: List[Image.Image], prompt_text: str) -> Dict[str, object]:
        if len(frames) < 2:
            return {"path_length":0.0, "smoothness":0.0, "semantic_scores":[0.0], "identity_drift":[0.0]}
        feats = self._clip_image_features(frames)
        text_feat = self._clip_text_feature(prompt_text)
        cos = feats @ text_feat
        diffs = feats[1:] - feats[:-1]
        path_length = diffs.norm(dim=-1).sum().item()
        if len(feats) > 2:
            sec = feats[2:] - 2*feats[1:-1] + feats[:-2]
            smoothness = sec.norm(dim=-1).mean().item()
        else:
            smoothness = 0.0
        base = feats[0]
        identity_drift = (feats - base).norm(dim=-1).tolist()
        return {"path_length": path_length, "smoothness": smoothness, "semantic_scores": cos.tolist(), "identity_drift": identity_drift}

    def _save_video_from_frames(self, frames: List[Image.Image], save_path: str, fps: int = 8):
        if not frames:
            return
        try:
            import cv2
            import numpy as np
        except ImportError:
            print("[WARN] opencv-python or numpy not installed. Cannot save video.")
            return

        width, height = frames[0].size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            print(f"[WARN] Failed to open video writer for path: {save_path}")
            return

        print(f"[INFO] Saving generated video to {save_path} ({len(frames)} frames, {fps} fps)...")
        for frame in frames:
            frame_np = np.array(frame.convert('RGB'))
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        writer.release()


    # ---------------- 迭代路径 ----------------
    def iterative_edit(self,
                       image: Image.Image,
                       prompt_text: str,
                       steps: int = 6,
                       candidates_per_step: int = 4,
                       height: int = 480,
                       width: int = 720,
                       guidance_scale: float = 6.0,
                       num_inference_steps: int = 50,
                       w_sem: float = 1.0,
                       w_step: float = 0.2,
                       w_id: float = 0.1,
                       generator: Optional[torch.Generator] = None,
                       temporal_caption: Optional[str] = None) -> Tuple[List[Image.Image], Dict[str, object]]:
        path = [image]
        text_feat = self._clip_text_feature(prompt_text)
        start_feat = None; prev_feat = None
        for t in range(1, steps+1):
            print(f"[PATH] Step {t}/{steps} ...")
            frames = self.generate_video(path[-1], prompt_text, candidates_per_step, height, width,
                                         guidance_scale, num_inference_steps, generator,
                                         temporal_caption=temporal_caption)
            if len(frames) <= 1:
                print("[PATH-WARN] 候选帧数 <=1，提前结束。"); break
            feats = self._clip_image_features(frames)
            if start_feat is None: start_feat = self._clip_image_features([path[0]])[0]
            if prev_feat is None: prev_feat = self._clip_image_features([path[-1]])[0]
            cos = feats @ text_feat
            step_dist = (feats - prev_feat).norm(dim=-1)
            id_dist = (feats - start_feat).norm(dim=-1)
            energy = w_sem*(1-cos) + w_step*step_dist + w_id*id_dist
            idx = int(energy.argmin().item())
            path.append(frames[idx])
            prev_feat = feats[idx]
        metrics = self.compute_path_metrics(path, prompt_text)
        return path, metrics

    # ---------------- 统一入口 ----------------
    def run(self, image: Image.Image, prompt_text: str,
        num_frames: int = 49, height: int = 480, width: int = 720,
        guidance_scale: float = 6.0, num_inference_steps: int = 50,
            generator: Optional[torch.Generator] = None, save_dir: str = 'outputs',
            use_iterative: bool = False, iterative_steps: int = 6, candidates_per_step: int = 4,
            w_sem: float = 1.0, w_step: float = 0.2, w_id: float = 0.1):
        os.makedirs(save_dir, exist_ok=True)
        temporal_caption = self.build_temporal_caption(image, prompt_text)
        run_info: Dict[str, Any] = {
            'temporal_caption': temporal_caption,
            'prompt': prompt_text,
            'mode': 'iterative' if use_iterative else 'baseline',
        }
        if getattr(self, '_last_temporal_plan', None) is not None:
            try:
                run_info['temporal_prompt_plan'] = self._last_temporal_plan.to_dict()
            except Exception as exc:
                print('[WARN] Failed to serialize temporal prompt plan:', exc)
        if use_iterative:
            path, metrics = self.iterative_edit(image, prompt_text, iterative_steps, candidates_per_step,
                                                height, width, guidance_scale, num_inference_steps,
                                                w_sem, w_step, w_id, generator,
                                                temporal_caption=temporal_caption)
            best = path[-1]
            if path:
                video_save_path = os.path.join(save_dir, f"path_{int(time.time())}.mp4")
                self._save_video_from_frames(path, video_save_path, fps=3)
            for i, fr in enumerate(path):
                try:
                    fr.save(os.path.join(save_dir, f'path_{i:02d}.png'))
                except Exception as exc:
                    print(f"[WARN] 保存路径帧失败 path_{i:02d}.png: {exc}")
            try:
                import json
                json.dump(metrics, open(os.path.join(save_dir, 'metrics.json'), 'w'), indent=2)
            except Exception as e:
                print('[WARN] 写入 metrics.json 失败:', e)
            processed = self.finalize_output_frame(best, save_dir, prefix='final_frame')
            run_info['clip_path_metrics'] = metrics
            run_info['outputs'] = {k: v for k, v in processed.items() if k != 'image'}
            run_info['outputs']['mode'] = 'iterative'
            best_processed = processed['image']
            return best_processed, path, run_info
        else:
            frames = self.generate_video(image, prompt_text, num_frames, height, width,
                                         guidance_scale, num_inference_steps, generator,
                                         temporal_caption=temporal_caption)
            if not frames:
                raise ValueError("Video generation returned no frames to select from.")
            video_save_path = os.path.join(save_dir, f"generation_{int(time.time())}.mp4")
            self._save_video_from_frames(frames, video_save_path, fps=8)

            plan_obj = getattr(self, '_last_temporal_plan', None)
            binary_info = self._binary_search_frame_selection(
                frames,
                prompt_text,
                temporal_caption,
                plan_obj,
            )
            binary_entry = dict(binary_info)
            binary_entry['used_for_selection'] = False

            vlm_info = self.select_frame_with_vlm(image, frames, prompt_text, save_dir)
            run_info['frame_collage'] = {
                'path': vlm_info.get('collage_path'),
                'mapping': vlm_info.get('mapping'),
            }
            run_info['binary_search'] = binary_entry

            aggregated_scores = binary_info.get('aggregated_scores') or []
            if aggregated_scores:
                run_info['clip_scores'] = aggregated_scores

            selected_frame = None
            final_index: Optional[int] = None
            strategy = 'vlm'

            selection = vlm_info.get('selection')
            if selection is not None and hasattr(selection, 'frame_index') and 0 <= selection.frame_index < len(frames):
                final_index = selection.frame_index
                candidate_idx = binary_info.get('first_good_index')
                if candidate_idx is not None and 0 <= candidate_idx < len(frames):
                    final_index = int(candidate_idx)
                    strategy = 'vlm+binary'
                    run_info['binary_search']['used_for_selection'] = True
                selected_frame = frames[final_index]
                run_info['frame_selection'] = {
                    'strategy': strategy,
                    'frame_id': f'T{final_index:02d}',
                    'frame_index': final_index,
                    'raw_response': getattr(selection, 'raw_response', None),
                    'collage_path': vlm_info.get('collage_path'),
                    'vlm_frame_index': getattr(selection, 'frame_index', None),
                }
            else:
                candidate_idx = binary_info.get('first_good_index')
                if candidate_idx is None or candidate_idx >= len(frames):
                    candidate_idx = binary_info.get('max_index', 0)
                candidate_idx = int(candidate_idx or 0)
                candidate_idx = max(0, min(len(frames) - 1, candidate_idx))
                final_index = candidate_idx
                selected_frame = frames[final_index]
                strategy = 'clip-binary'
                run_info['binary_search']['used_for_selection'] = True
                run_info['frame_selection'] = {
                    'strategy': strategy,
                    'frame_id': f'T{final_index:02d}',
                    'frame_index': final_index,
                }

            if aggregated_scores and 0 <= final_index < len(aggregated_scores):
                run_info['frame_selection']['aggregated_score'] = aggregated_scores[final_index]
            if binary_info.get('threshold_score') is not None:
                run_info['frame_selection']['threshold_score'] = binary_info['threshold_score']
                run_info['frame_selection']['threshold_ratio'] = binary_info.get('threshold_ratio')
            run_info['frame_selection']['prompts_used'] = binary_info.get('prompts', [])

            processed = self.finalize_output_frame(selected_frame, save_dir, prefix='best_frame')
            run_info['outputs'] = {k: v for k, v in processed.items() if k != 'image'}
            run_info['outputs']['mode'] = 'baseline'
            selected_processed = processed['image']
            return selected_processed, frames, run_info
