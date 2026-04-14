本仓库尝试复现论文：

Are Image-to-Video Models Good Zero-Shot Image Editors?

https://huggingface.co/papers/2511.19435

https://arxiv.org/pdf/2511.19435

并且在其基础上进行改进

当前阶段：
在已经完成的IFEdit+Wan2.2管线上加入Frame Guidance

./papers/Frame Guidance Paper.pdf 是论文的PDF
./frame-guidance/ 是Frame Guidance的代码

我希望用已经实现的IFEdit+Wan2.2（非diffusers）管线，加上Frame Guidance，实现一个完整的视频生成管线，目前在做style部分的实现