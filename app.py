from __future__ import annotations

import os
import gradio as gr
from pipeline import Frame2FramePipeline
try:
    from vlm import FrameSelector, TemporalCaptionGenerator
except Exception:
    FrameSelector = None  # type: ignore
    TemporalCaptionGenerator = None  # type: ignore
from utils import run_pipeline_dispatch  # 引入统一调度

# =============================================================
# 0. 模型初始化（只做一次）
# =============================================================
# 强制使用 fp32 模式，避免黑帧重试，提高稳定性
os.environ['FRAME2FRAME_FORCE_FP32'] = '1'

MODEL_DIR = "/root/autodl-tmp/Workspace/Pathway/model/cogvideox"  # 已下载模型目录

caption_generator = TemporalCaptionGenerator() if TemporalCaptionGenerator else None
frame_selector = FrameSelector() if FrameSelector else None

pipeline = Frame2FramePipeline(
    model_name=MODEL_DIR,      # 指向本地目录 / 远程仓库名都可以
    validate_access=False,     # 不做 HuggingFace 权限预检（本地已缓存）
    prefer_local=True,         # 优先从本地加载
    caption_generator=caption_generator,
    frame_selector=frame_selector,
)


# =============================================================
 # 统一调度函数来自 utils.run_pipeline_dispatch
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
            prompt = gr.Textbox(label="编辑文本 (prompt)", value="make it a watercolor painting")
            num_frames = gr.Slider(4, 64, value=49, step=1, label="(Baseline) 视频总帧数 (视频长度)")
            guidance_scale = gr.Slider(1, 12, value=6, step=0.5, label="Guidance Scale 文本引导强度")
            num_inference_steps = gr.Slider(10, 100, value=50, step=1, label="单帧生成质量 (步数)")
            # 某些旧版本 gradio 的 Number 不支持 precision 参数，这里直接去掉
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
