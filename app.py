from __future__ import annotations

import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file FIRST, before any other local imports
load_dotenv()

import gradio as gr
from pipeline import Frame2FramePipeline
from vlm import FrameSelector, TemporalCaptionGenerator
from utils import run_pipeline_dispatch  # 引入统一调度


# =============================================================
# 0. 模型初始化（只做一次）
# =============================================================
# 强制使用 fp32 模式，避免黑帧重试，提高稳定性
os.environ['FRAME2FRAME_FORCE_FP32'] = '1'

MODEL_REPO_URL = "https://modelscope.cn/models/ZhipuAI/CogVideoX1.5-5B-I2V.git"  # ModelScope git mirror
MODEL_INDEX_FILENAME = "model_index.json"

def _find_diffusers_root(base_path: Path) -> Path | None:
    candidates = [base_path]
    try:
        candidates.extend(child for child in base_path.iterdir() if child.is_dir())
    except FileNotFoundError:
        return None
    for candidate in candidates:
        if (candidate / MODEL_INDEX_FILENAME).is_file():
            return candidate
    return None

def ensure_model_checkout(target_dir: str, repo_url: str) -> str:
    base_path = Path(target_dir).expanduser()
    resolved = _find_diffusers_root(base_path)
    if resolved is not None:
        return str(resolved)

    needs_clone = False
    if base_path.exists():
        try:
            has_entries = any(base_path.iterdir())
        except OSError:
            has_entries = True
        if has_entries:
            try:
                deep_candidate = next((path.parent for path in base_path.rglob(MODEL_INDEX_FILENAME)), None)
            except OSError:
                deep_candidate = None
            if deep_candidate is not None:
                print(f"[INFO] Using CogVideoX assets detected at {deep_candidate}")
                return str(deep_candidate)
            print(f"[WARN] CogVideoX directory {base_path} exists but missing {MODEL_INDEX_FILENAME}.")
        else:
            needs_clone = True
    else:
        base_path.parent.mkdir(parents=True, exist_ok=True)
        needs_clone = True

    if needs_clone:
        print(f"[INFO] CogVideoX model not found; cloning from {repo_url} into {base_path}...")
        try:
            subprocess.run(['git', 'clone', repo_url, str(base_path)], check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"Failed to clone CogVideoX repository: {exc}") from exc

    resolved = _find_diffusers_root(base_path)
    if resolved is not None:
        return str(resolved)

    try:
        deep_candidate = next((path.parent for path in base_path.rglob(MODEL_INDEX_FILENAME)), None)
    except OSError:
        deep_candidate = None
    if deep_candidate is not None:
        print(f"[INFO] Falling back to CogVideoX assets at {deep_candidate}")
        return str(deep_candidate)

    raise FileNotFoundError(
        f"Failed to locate {MODEL_INDEX_FILENAME} under {base_path}. Please download the CogVideoX weights manually or set FRAME2FRAME_MODEL_DIR."
    )

DEFAULT_MODEL_DIR = os.environ.get("FRAME2FRAME_MODEL_DIR") or str((Path(__file__).resolve().parent / "models" / "cogvideox").resolve())  # 模型缓存目录
MODEL_DIR = ensure_model_checkout(DEFAULT_MODEL_DIR, MODEL_REPO_URL)
os.environ['FRAME2FRAME_MODEL_DIR'] = MODEL_DIR

caption_generator = TemporalCaptionGenerator()
frame_selector = FrameSelector()

pipeline = Frame2FramePipeline(
    model_name=MODEL_DIR,      # 指向本地目录 / 远程仓库名都可以
    validate_access=False,     # 不做 HuggingFace 权限预检（本地已缓存）
    prefer_local=True,         # 优先从本地加载
    caption_generator=caption_generator,
    frame_selector=frame_selector,
)


# =============================================================
# 2. 调度与执行
# =============================================================
def run_pipeline(image, prompt, num_frames, guidance_scale, seed,
                 use_iterative, iterative_steps, candidates_per_step,
                 w_sem, w_step, w_id, num_inference_steps):
    return run_pipeline_dispatch(
        pipeline=pipeline,
        image=image,
        prompt=prompt,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        seed=seed,
        use_iterative=use_iterative,
        iterative_steps=iterative_steps,
        candidates_per_step=candidates_per_step,
        w_sem=w_sem,
        w_step=w_step,
        w_id=w_id,
        num_inference_steps=num_inference_steps,
    )


# =============================================================
# 3. 构建 UI
# =============================================================
with gr.Blocks(title="Frame2Frame 简化演示") as demo:
    gr.Markdown("""# Frame2Frame 演示
**模式说明**  
Baseline: 一次生成多帧，自动选最符合文本的那一帧。  
Iterative: 多步生成 + 能量函数筛选，路径更平滑、可控。  
""")

    with gr.Row():
        # 左侧：输入与参数
        with gr.Column(scale=1):
            gr.Markdown("### 1. 输入 & 基础参数")
            image = gr.Image(type="pil", label="输入图像(参考帧)")
            prompt = gr.Textbox(label="编辑文本 (prompt)", value="The wooden door slowly swings closed, the light from the room gradually dims until it disappears completely, the camera remains static.")
            num_frames = gr.Slider(4, 160, value=81, step=1, label="(Baseline) 视频总帧数 (视频长度)")
            guidance_scale = gr.Slider(1, 12, value=6, step=0.5, label="Guidance Scale 文本引导强度")
            num_inference_steps = gr.Slider(10, 100, value=50, step=1, label="单帧生成质量 (步数)")
            seed = gr.Number(value=-1, label="随机种子 (-1 表示自动)")
            use_iterative = gr.Checkbox(value=False, label="启用 Iterative 迭代路径模式")

            with gr.Accordion("(可选) 迭代参数", open=False):
                iterative_steps = gr.Slider(2, 12, value=5, step=1, label="迭代优化次数")
                candidates_per_step = gr.Slider(2, 8, value=3, step=1, label="每步候选帧数 candidates")
                w_sem = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="w_sem 语义权重 ↑更贴合文本")
                w_step = gr.Slider(0.0, 1.0, value=0.2, step=0.01, label="w_step 相邻平滑权重")
                w_id = gr.Slider(0.0, 1.0, value=0.1, step=0.01, label="w_id 保持身份(与首帧一致)")

            run_btn = gr.Button("▶ 运行生成")

        # 右侧：输出
        with gr.Column(scale=1):
            gr.Markdown("### 2. 输出")
            result = gr.Image(type="pil", label="最终结果帧")
            out_seed = gr.Number(label="实际使用的种子")
            info = gr.Textbox(label="运行信息")

    # 事件绑定
    run_btn.click(
        fn=run_pipeline,
        inputs=[image, prompt, num_frames, guidance_scale, seed, use_iterative,
                iterative_steps, candidates_per_step, w_sem, w_step, w_id, num_inference_steps],
        outputs=[result, out_seed, info]
    )

if __name__ == "__main__":
    # 监听全部网卡便于在远程机器访问
    demo.launch(server_name="0.0.0.0", server_port=7860)


