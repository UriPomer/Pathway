# Frame2Frame-mini: Pathways on the Image Manifold (Baseline + Iterative Path)

一个使用开源视频扩散模型 (CogVideoX I2V) + CLIP 特征，构建 "图像编辑路径 (pathway)" 的轻量级实现原型。

> 目标：在无需训练新模型的前提下，用“逐步迭代 + 候选帧能量选择”近似论文《Pathways on the Image Manifold: Image Editing via Video Generation》思想：即在特征空间上从原始图像到目标语义的平滑、可控推进。

当前实现：
1. 基线模式：一次性生成 `N` 帧，用 CLIP 文本相关性选最佳帧。
2. 迭代路径模式 (iterative)：多步 (steps)，每步生成若干候选帧 (candidates_per_step)，利用能量函数选择下一个状态，形成路径并统计指标。
3. Temporal Caption + Frame Selection：内置可选的 GPT-4o 多模态工作流，自动生成“时间编辑描述”并在帧拼贴上挑选最早完成编辑的帧（无 Key 时自动回退到启发式）。
4. 指标：
   - path_length：CLIP 图像嵌入路径的累积 L2 长度
   - smoothness：二阶差分平均范数（越小越平滑）
   - semantic_scores：各帧与文本 CLIP 余弦相似度
   - identity_drift：各帧与起始帧特征距离

仍未实现（论文额外可能包含的更高级内容）：
* 多尺度/多阶段调度
* 更细粒度的几何正则（曲率约束）
* ArcFace/DINO 身份保持特征
* 更精确的 geodesic 近似 / 优化器式修正
* 可视化投影 (t-SNE / PCA) + 批量实验脚本

## 环境安装
```bash
conda env create -f environment.yml
conda activate frame2frame
```

或者使用 `requirements.txt`：
```bash
pip install -r requirements.txt
```

## 运行 Gradio 基线
```bash
python app.py
```

### OpenAI 多模态支持（可选）
若希望复现论文中的“Temporal Caption + Frame Selection”全流程，请在运行前设置 OpenAI Key：

```bash
export OPENAI_API_KEY="sk-..."
# 可选：指定自定义模型
export FRAME2FRAME_VLM_MODEL="gpt-4o-mini"
python app.py
```

若未配置 Key，流水线会自动退化为内置的启发式 caption 与 CLIP 帧选择，依旧可以完整运行。
## 指定模型缓存目录
默认将所有权重缓存到当前仓库下 `./models/`，避免写入用户全局 `~/.cache/huggingface`。

两种方式覆盖：
1. 代码参数：
```python
from pipeline import Frame2FramePipeline
pipe = Frame2FramePipeline(model_cache_dir="./my_cache")
```
2. 环境变量（优先生效）：
```bash
export FRAME2FRAME_MODEL_DIR=./my_cache
python app.py
```

目录结构示例：
```
models/
	zai-org__CogVideoX1.5-5B-I2V/ ... (由 diffusers 自动命名)
	clip-vit-base-patch16/
```

浏览器访问：`http://localhost:7860`

界面参数：
* 帧数 (baseline 模式使用)
* Guidance Scale
* 随机种子（-1 表示自动）

## 使用迭代路径接口（命令行示例）
（临时示例，可在交互式 Python 中运行）
```python
from PIL import Image
from pipeline import Frame2FramePipeline
import torch

pipe = Frame2FramePipeline()
img = Image.open('your_image.jpg').convert('RGB')
g = torch.Generator(device=pipe.device).manual_seed(42)
best, path, metrics = pipe.run(
	image=img,
	prompt_text="make it a watercolor painting",
	use_iterative=True,
	iterative_steps=5,
	candidates_per_step=3,
	guidance_scale=6.0,
	generator=g
)
print(metrics)
```

输出：
* `outputs/final_frame.png` 最终编辑结果
* `outputs/path_XX.png` 中间帧
* `outputs/metrics.json` 指标

## 能量函数（迭代模式）
对候选帧的 CLIP 图像特征 `f`：
```
E = w_sem * (1 - cos(f, text)) + w_step * || f - f_prev ||_2 + w_id * || f - f_start ||_2
```
可在 `Frame2FramePipeline.iterative_edit` 中调参：`w_sem / w_step / w_id`。

## 路线图
| 阶段 | 内容 | 状态 |
|------|------|------|
| 0 | 一次性生成 & CLIP 选帧 | 完成 |
| 1 | 迭代路径 + 基础指标 | 完成 |
| 2 | 添加 ArcFace/DINO 身份保持 | TODO |
| 3 | 多尺度 / strength 调度 | TODO |
| 4 | 更复杂能量 (曲率、语义单调性约束) | TODO |
| 5 | 可视化与批量评估脚本 | TODO |

## 已知限制
* CogVideoX I2V 的接口抽象：暂未做 img2img strength 渐进策略，只是“以上一帧为输入再生成候选”的近似。
* CLIP 特征空间非严格流形度量，此处路径长度是启发式指标。
* 速度较慢（视频扩散本身体积大）；可减少 `num_inference_steps` 或分辨率。

## 许可证与致谢
* 模型权重归其各自开源项目所有。
* 本仓库仅用于研究与教学演示。

## 快速问题排查
| 现象 | 可能原因 | 解决 |
|------|----------|------|
| OOM | 显存不足 | 减小分辨率 / 步数 / 候选数；保持 offload | 
| 路径不平滑 | w_step 太低 | 增大 w_step 或减少 candidates_per_step |
| 语义不满足 | w_sem 太低 | 调高 w_sem 或增加 steps |
| 身份漂移大 | w_id 太低 | 增大 w_id, 引入专门身份特征 |

欢迎继续完善与补充。
