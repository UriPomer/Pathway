# 实验数据归档

## 实验信息
- **实验名称**: 6prompts_shift_stop_step_scan_20251228_013255
- **时间**: 20251228_013255
- **描述**: 6个prompt的shift_stop_step参数扫描实验

## 数据统计
- **Prompt 数量**: 6
- **测试的 shift_stop_step 值**: 0, 5, 10, 15, 20
- **每个prompt测试**: 5个不同的shift_stop_step值

## 目录结构
```
6prompts_shift_stop_step_scan_20251228_013255/
├── json/              # 所有 JSON 结果文件（6个prompt的测试结果）
├── charts/            # 生成的图表（PDF 和 PNG）
├── experiment_info.json  # 实验元信息
├── summary_statistics.json  # 汇总统计数据
└── README.md          # 本文件
```

**注意**: 视频文件未归档，原始视频保存在 `test/outputs/` 目录下。

## 指标说明
- **循环质量指标**: loop_mse, loop_lpips, loop_ssim
- **视频质量指标**: frame_psnr_mean, frame_ssim_mean, frame_lpips_mean, video_sharpness

## 最佳结果
根据测试结果，推荐使用 **shift_stop_step=5** 或 **shift_stop_step=10**。

详细分析请参考 charts/ 目录下的图表。
