from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file FIRST, before any other local imports
load_dotenv()

import gradio as gr
from pipeline import Frame2FramePipeline
from vlm import FrameSelector, TemporalCaptionGenerator
from utils import run_pipeline_dispatch

ENABLE_CPU_OFFLOAD = True  # 改为 True 启用 offload
os.environ['DISABLE_OFFLOAD'] = '0' if ENABLE_CPU_OFFLOAD else '1'
print(f"[CONFIG] CPU Offload: {'启用 (省显存)' if ENABLE_CPU_OFFLOAD else '禁用 (快)'}")

# =============================================================
# 0. 模型初始化（只做一次）
# =============================================================

MODEL_REPO_ID = "ZhipuAI/CogVideoX1.5-5B-I2V"  # ModelScope 仓库 ID
MODEL_INDEX_FILENAME = "model_index.json"

def _find_diffusers_root(base_path: Path) -> Path | None:
    """在指定目录中查找包含 model_index.json 的 diffusers 根目录"""
    candidates = [base_path]
    try:
        candidates.extend(child for child in base_path.iterdir() if child.is_dir())
    except FileNotFoundError:
        return None
    for candidate in candidates:
        if (candidate / MODEL_INDEX_FILENAME).is_file():
            return candidate
    return None

def ensure_model_checkout(target_dir: str, repo_id: str) -> str:
    base_path = Path(target_dir).expanduser()
    
    # 首先检查是否已存在
    resolved = _find_diffusers_root(base_path)
    if resolved is not None:
        print(f"[INFO] 使用已缓存的 CogVideoX 模型: {resolved}")
        return str(resolved)

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
                print(f"[INFO] 使用检测到的 CogVideoX 资源: {deep_candidate}")
                return str(deep_candidate)

    print(f"[INFO] CogVideoX 模型未找到，正在从 ModelScope 下载...")
    print(f"[INFO] 仓库: {repo_id}")
    print(f"[INFO] 目标目录: {base_path}")
    
    try:
        from modelscope import snapshot_download
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_dir = snapshot_download(
            repo_id,
            cache_dir=str(base_path.parent),
            local_dir=str(base_path),
        )
        
        print(f"[INFO] 模型下载成功: {model_dir}")
        
    except ImportError as exc:
        raise RuntimeError(
            "需要安装 modelscope 包。请运行: pip install modelscope"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"从 ModelScope 下载模型失败: {exc}") from exc

    resolved = _find_diffusers_root(base_path)
    if resolved is not None:
        return str(resolved)

    try:
        deep_candidate = next((path.parent for path in base_path.rglob(MODEL_INDEX_FILENAME)), None)
    except OSError:
        deep_candidate = None
    if deep_candidate is not None:
        print(f"[INFO] 回退到 CogVideoX 资源: {deep_candidate}")
        return str(deep_candidate)

    raise FileNotFoundError(
        f"在 {base_path} 下未找到 {MODEL_INDEX_FILENAME}。"
        f"请手动下载 CogVideoX 权重或设置 FRAME2FRAME_MODEL_DIR 环境变量。"
    )

DEFAULT_MODEL_DIR = os.environ.get("FRAME2FRAME_MODEL_DIR") or str(
    (Path(__file__).resolve().parent / "models" / "cogvideox").resolve()
)
MODEL_DIR = ensure_model_checkout(DEFAULT_MODEL_DIR, MODEL_REPO_ID)
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
