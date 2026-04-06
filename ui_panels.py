# Copyright 2024-2025. UI panels for Mobius (Loopless) and IF-Edit modes.
"""Decoupled UI panels for Mobius (Loopless Cinemagraph) and IF-Edit modes."""

import gradio as gr


class FrameGuidancePanel:
    """UI and logic for Frame Guidance (multi-loss constrained generation).

    Supports **style** (all-frame CSD constraint) and **scribble**
    (last-frame edge constraint using the brush overlay from the main image editor).
    ``loop`` has been moved to :class:`MobiusPanel` as a sub-option.
    """

    ACCORDION_LABEL = "Frame Guidance (帧级约束)"

    LOSS_TYPES = [
        ("style", "风格化 (Style)"),
        ("scribble", "草图/线稿 (Scribble)"),
    ]

    @staticmethod
    def create_accordion():
        """Create Frame Guidance accordion.

        Returns (enable, loss_type, lr, downscale, style_image).

        - ``style``: shows ``style_image`` (风格参考图)
        - ``scribble``: shows a hint to use the brush editor; no extra upload needed
        """
        with gr.Accordion(FrameGuidancePanel.ACCORDION_LABEL, open=False):
            enable = gr.Checkbox(label="启用 Frame Guidance", value=False)
            loss_type = gr.Dropdown(
                label="Loss 类型",
                choices=[lt[1] for lt in FrameGuidancePanel.LOSS_TYPES],
                value=FrameGuidancePanel.LOSS_TYPES[0][1],
            )
            lr = gr.Slider(label="引导学习率", minimum=0.1, maximum=10.0, value=3.0, step=0.1)
            downscale = gr.Dropdown(label="Latent 降采样因子", choices=[1, 2, 4], value=4)

            scribble_hint = gr.Markdown(
                "**草图模式**：请在上方「轮廓笔刷」编辑器中绘制线稿，生成视频的最后一帧将被约束为匹配您的笔刷轮廓。",
                visible=False,
            )
            style_image = gr.Image(label="风格参考图", type="filepath", visible=True)

        return enable, loss_type, lr, downscale, scribble_hint, style_image

    @staticmethod
    def on_loss_type_change(loss_type_label: str):
        """Toggle visibility of condition inputs based on loss type.

        Returns (scribble_hint_update, style_image_update).
        """
        label_to_name = {lt[1]: lt[0] for lt in FrameGuidancePanel.LOSS_TYPES}
        loss_fn = label_to_name.get(loss_type_label, "style")

        return (
            gr.update(visible=(loss_fn == "scribble")),
            gr.update(visible=(loss_fn == "style")),
        )

    @staticmethod
    def get_loss_fn_name(label: str) -> str:
        """Convert display label to internal loss function name."""
        label_to_name = {lt[1]: lt[0] for lt in FrameGuidancePanel.LOSS_TYPES}
        return label_to_name.get(label, "style")


class MobiusPanel:
    """UI and logic for Loopless Cinemagraph (Mobius-like Latent Shift).

    Now includes an optional **FG Loop** sub-section that uses Frame Guidance
    loop loss (MSE between first & last frame) to reinforce seamless looping.
    """

    ACCORDION_LABEL = "Loopless Cinemagraph"
    DEFAULT_SKIP = 4
    DEFAULT_STOP_STEP = 0

    @staticmethod
    def create_accordion():
        """Create Loopless accordion components.
        Returns (enable, skip, stop_step, fg_loop_enable, fg_loop_lr, fg_loop_downscale)."""
        with gr.Accordion(MobiusPanel.ACCORDION_LABEL, open=False):
            enable = gr.Checkbox(
                label="启用无缝循环 (Mobius-like Latent Shift)",
                value=False,
            )
            skip = gr.Slider(
                label="Latent Shift Skip",
                minimum=1,
                maximum=16,
                value=MobiusPanel.DEFAULT_SKIP,
                step=1,
            )
            stop_step = gr.Slider(
                label="最后N步停止Shift",
                minimum=0,
                maximum=40,
                value=MobiusPanel.DEFAULT_STOP_STEP,
                step=1,
            )

            gr.Markdown("### FG Loop 约束 (可选)")
            fg_loop_enable = gr.Checkbox(
                label="启用 FG Loop (首尾帧 MSE 约束)",
                value=False,
                interactive=False,
            )
            fg_loop_lr = gr.Slider(
                label="FG Loop 引导学习率",
                minimum=0.1,
                maximum=10.0,
                value=3.0,
                step=0.1,
                interactive=False,
            )
            fg_loop_downscale = gr.Dropdown(
                label="FG Loop Latent 降采样因子",
                choices=[1, 2, 4],
                value=4,
                interactive=False,
            )

        return enable, skip, stop_step, fg_loop_enable, fg_loop_lr, fg_loop_downscale

    @staticmethod
    def on_enable_change(enabled):
        """When Mobius enable toggles, update FG Loop controls interactivity."""
        return (
            gr.update(interactive=bool(enabled)),
            gr.update(interactive=bool(enabled)),
            gr.update(interactive=bool(enabled)),
        )

    @staticmethod
    def on_enable_disable_ifedit(enabled):
        """When Mobius is enabled, disable all IF-Edit controls."""
        if enabled:
            return (
                gr.update(value=False, interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(value=False, interactive=False),
                gr.update(interactive=False),
                gr.update(value=False, interactive=False),
            )
        return (
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )

    @staticmethod
    def get_output_panel_updates():
        """Return (scpr_visible, video_label, image_label) for Mobius mode."""
        return (
            gr.update(visible=False),
            gr.update(label="生成视频 (无缝循环)"),
            gr.update(visible=False),
        )

    @staticmethod
    def format_run_info(prompt: str, skip: int, stop_step: int, best_idx: int, best_score: float) -> list[str]:
        """Build run info lines for Mobius mode."""
        return [
            f"实际输入 prompt: {prompt}",
            f"[Mobius] Loopless Cinemagraph: ON (skip={skip}, stop_last={stop_step})",
        ]


class IFEditPanel:
    """UI and logic for IF-Edit (CoT, TLD, SCPR)."""

    ACCORDION_LABEL = "IF-Edit"
    DEFAULT_TLD_K = 3
    DEFAULT_SCPR_RATIO = 0.2

    def __init__(self, tld_threshold_ratio: float = 0.9):
        self.tld_threshold_ratio = tld_threshold_ratio

    def create_accordion(self):
        """Create IF-Edit accordion components."""
        with gr.Accordion(IFEditPanel.ACCORDION_LABEL, open=False):
            use_cot = gr.Checkbox(label="CoT Prompt Enhancement", value=False)
            use_tld = gr.Checkbox(label="Temporal Latent Dropout (TLD)", value=False)
            tld_step_k = gr.Slider(
                label="TLD Step K",
                minimum=1,
                maximum=8,
                value=IFEditPanel.DEFAULT_TLD_K,
                step=1,
            )
            tld_threshold_ratio = gr.Slider(
                label="TLD Threshold Ratio",
                minimum=0.0,
                maximum=1.0,
                value=self.tld_threshold_ratio,
                step=0.05,
            )
            use_scpr = gr.Checkbox(label="Self-Consistent Post-Refinement (SCPR)", value=False)
            scpr_refinement_ratio = gr.Slider(
                label="SCPR Refinement Ratio",
                minimum=0.1,
                maximum=1.0,
                value=IFEditPanel.DEFAULT_SCPR_RATIO,
                step=0.05,
            )
        return use_cot, use_tld, tld_step_k, tld_threshold_ratio, use_scpr, scpr_refinement_ratio

    @staticmethod
    def on_enable_disable_mobius(enabled):
        """When any IF-Edit (TLD) is enabled, disable Mobius controls (including FG Loop)."""
        if enabled:
            return (
                gr.update(value=False, interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(value=False, interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )
        return (
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )

    @staticmethod
    def get_output_panel_updates(use_scpr: bool):
        """Return (scpr_visible, video_label, image_label) for IF-Edit mode."""
        return (
            gr.update(visible=bool(use_scpr)),
            gr.update(label="生成视频"),
            gr.update(label="SCPR 精修输出帧" if use_scpr else "主视频最清晰帧"),
        )

    @staticmethod
    def format_run_info_no_scpr(
        prompt: str,
        use_cot: bool,
        use_tld: bool,
        tld_k: int,
        tld_threshold: float,
        best_idx: int,
        best_score: float,
    ) -> list[str]:
        """Build run info for IF-Edit without SCPR."""
        return [
            f"实际输入 prompt: {prompt}",
            f"[IF-Edit] CoT: {'ON' if use_cot else 'OFF'} | TLD: {'ON' if use_tld else 'OFF'} "
            f"(K={tld_k}, threshold={tld_threshold:.2f})",
            f"[IF-Edit] SCPR: OFF",
            f"主视频后2/3最清晰帧: idx={best_idx}, score={best_score:.4f}",
        ]

    @staticmethod
    def format_run_info_with_scpr(
        prompt: str,
        use_cot: bool,
        use_tld: bool,
        tld_k: int,
        tld_threshold: float,
        main_idx: int,
        total_frames: int,
        main_score: float,
    ) -> list[str]:
        """Build run info for IF-Edit with SCPR (initial lines before SCPR result)."""
        return [
            f"实际输入 prompt: {prompt}",
            f"[IF-Edit] CoT: {'ON' if use_cot else 'OFF'} | TLD: {'ON' if use_tld else 'OFF'} "
            f"(K={tld_k}, threshold={tld_threshold:.2f})",
            f"[IF-Edit] SCPR: ON",
            f"SCPR 输入: 主视频后2/3最清晰帧 (第 {main_idx} 帧, 共 {total_frames} 帧, score={main_score:.4f})",
        ]


def get_output_panel_updates_by_mode(loopless_enable: bool, use_scpr: bool):
    """Dispatch output panel updates based on current mode (Mobius vs IF-Edit)."""
    if loopless_enable:
        return MobiusPanel.get_output_panel_updates()
    return IFEditPanel.get_output_panel_updates(use_scpr)
