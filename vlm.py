"""IF-Edit VLM Utilities: CoT Prompt Enhancement for Zero-Shot Image Editing.

Implements the Chain-of-Thought (CoT) temporal prompt enhancement from
"Are Image-to-Video Models Good Zero-Shot Image Editors?" (arXiv: 2511.19435).

The key insight: Instead of directly using the edit instruction as the I2V prompt,
we use a VLM to generate a temporal description of how the scene should evolve
from the source state to the edited state.

Two prompt enhancer backends:
1. IFEditPromptEnhancer — Uses OpenAI-compatible API (GPT-4o / Qwen-VL)
2. QwenVLPromptEnhancer — Dedicated Qwen3-VL local inference

Also retains the legacy TemporalCaptionGenerator for backward compatibility.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from dotenv import load_dotenv
load_dotenv()

from PIL import Image, ImageDraw, ImageFont

# Try importing OpenAI client
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_VLM_MODEL = os.environ.get("IFEDIT_VLM_MODEL", "gpt-4o-mini")
QWEN_VL_MODEL = os.environ.get("IFEDIT_QWEN_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")

# ---------------------------------------------------------------------------
# IF-Edit CoT Prompt System Prompt
# ---------------------------------------------------------------------------

IFEDIT_COT_SYSTEM_PROMPT = textwrap.dedent("""\
You are a temporal reasoning assistant for image editing via video generation.

Given a source image and an edit instruction, you must describe how the scene
would naturally evolve over time from its current state to the edited state,
as if it were a short video clip.

Rules:
1. Describe the transformation as a TEMPORAL PROCESS — what happens frame by
   frame, not just the end state.
2. Keep the description in 2-4 sentences.
3. Maintain all aspects of the original scene that are NOT being edited.
4. Use gradual, natural language: "slowly", "gradually", "begins to", etc.
5. Describe the camera as STATIC unless the edit explicitly requires camera motion.
6. Focus on the visual changes that would be observable in a video.

Example:
- Edit instruction: "Add sunglasses to the person"
- Good temporal description: "The scene remains still as a pair of stylish dark
  sunglasses gradually materializes on the person's face, settling naturally on
  the bridge of their nose. The rest of the scene stays completely static with
  no changes to lighting or background."

Example:
- Edit instruction: "Make it snowing"
- Good temporal description: "Gentle snowflakes begin to appear and fall softly
  through the scene, gradually increasing in density. A thin layer of white
  snow slowly accumulates on horizontal surfaces. The lighting shifts slightly
  cooler as the atmosphere becomes more wintry, while the main subject remains
  unchanged."

Respond with ONLY the temporal description, no additional commentary.
""").strip()

# ---------------------------------------------------------------------------
# Helper: image → base64
# ---------------------------------------------------------------------------

def _pil_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# OpenAI-compatible API wrapper
# ---------------------------------------------------------------------------

class _OpenAIWrapper:
    def __init__(self):
        self._api_key = os.environ.get("OPENAI_API_KEY")
        self._base_url = os.environ.get("OPENAI_BASE_URL")
        self._client = None

    @property
    def available(self) -> bool:
        return OpenAI is not None and bool(self._api_key)

    @property
    def client(self):
        if self._client is not None:
            return self._client
        if not self.available:
            return None
        try:
            kwargs = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
        except Exception as exc:
            print(f"[OpenAIWrapper] Init failed: {exc}")
            self._client = None
        return self._client


# ---------------------------------------------------------------------------
# IF-Edit Prompt Enhancer (main component)
# ---------------------------------------------------------------------------

class IFEditPromptEnhancer:
    """CoT Prompt Enhancement for IF-Edit.

    Converts a static edit instruction into a temporal description of how
    the scene evolves, suitable for driving an I2V model.

    Supports two backends:
    - OpenAI-compatible API (default, uses GPT-4o / any vision LLM)
    - Local Qwen-VL (via transformers, set IFEDIT_USE_LOCAL_VLM=1)
    """

    def __init__(
        self,
        model: str = DEFAULT_VLM_MODEL,
        use_local_vlm: bool = False,
    ):
        self.model = model
        self.use_local_vlm = use_local_vlm or os.environ.get("IFEDIT_USE_LOCAL_VLM") == "1"
        self._openai_wrapper = _OpenAIWrapper()
        self._local_vlm = None

        if self.use_local_vlm:
            print("[IFEditPromptEnhancer] Will use local VLM for CoT prompts.")
        elif self._openai_wrapper.available:
            print(f"[IFEditPromptEnhancer] Using OpenAI API (model={model})")
        else:
            print("[IFEditPromptEnhancer] No VLM backend available, will use heuristic fallback.")

    def enhance(self, image: Image.Image, edit_instruction: str) -> str:
        """Generate temporal CoT prompt from edit instruction.

        Args:
            image: Source image
            edit_instruction: e.g. "Add sunglasses to the person"

        Returns:
            Temporal description of the editing process.
        """
        if not edit_instruction or not edit_instruction.strip():
            raise ValueError("Edit instruction cannot be empty.")

        # Try local VLM first
        if self.use_local_vlm:
            result = self._enhance_local(image, edit_instruction)
            if result:
                return result

        # Try OpenAI API
        if self._openai_wrapper.available:
            result = self._enhance_openai(image, edit_instruction)
            if result:
                return result

        # Heuristic fallback
        return self._enhance_heuristic(image, edit_instruction)

    def _enhance_openai(self, image: Image.Image, edit_instruction: str) -> Optional[str]:
        """Use OpenAI-compatible VLM for CoT prompt generation."""
        client = self._openai_wrapper.client
        if client is None:
            return None

        image_b64 = _pil_to_base64(image)
        user_prompt = (
            f"Source image is attached. Edit instruction: \"{edit_instruction}\"\n\n"
            f"Describe how this scene would naturally transform over time to "
            f"achieve the edit, as a short video description."
        )

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": IFEDIT_COT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                            },
                        ],
                    },
                ],
                max_tokens=256,
            )
            result = response.choices[0].message.content.strip()
            print(f"[CoT-Prompt] VLM response: {result[:120]}...")
            return result
        except Exception as exc:
            print(f"[CoT-Prompt] OpenAI call failed: {exc}")
            return None

    def _enhance_local(self, image: Image.Image, edit_instruction: str) -> Optional[str]:
        """Use local Qwen-VL for CoT prompt generation.

        Requires: transformers, qwen-vl-utils
        """
        try:
            if self._local_vlm is None:
                self._local_vlm = self._load_local_vlm()
            if self._local_vlm is None:
                return None

            processor, model = self._local_vlm

            user_prompt = (
                f"Edit instruction: \"{edit_instruction}\"\n\n"
                f"Describe how this scene would naturally transform over time "
                f"to achieve the edit, as if watching a short video clip. "
                f"Keep the description to 2-4 sentences. Be specific about "
                f"visual changes and keep the camera static."
            )

            messages = [
                {"role": "system", "content": IFEDIT_COT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            generated_ids = model.generate(**inputs, max_new_tokens=256)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            print(f"[CoT-Prompt] Local VLM response: {output_text[:120]}...")
            return output_text

        except Exception as exc:
            print(f"[CoT-Prompt] Local VLM failed: {exc}")
            return None

    def _load_local_vlm(self):
        """Load local Qwen-VL model."""
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            import torch

            model_name = QWEN_VL_MODEL
            print(f"[CoT-Prompt] Loading local VLM: {model_name}")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            processor = AutoProcessor.from_pretrained(model_name)
            return (processor, model)
        except Exception as exc:
            print(f"[CoT-Prompt] Failed to load local VLM: {exc}")
            return None

    @staticmethod
    def _enhance_heuristic(image: Image.Image, edit_instruction: str) -> str:
        """Heuristic fallback when no VLM is available.

        Simply returns the original edit instruction as-is.
        """
        return edit_instruction.strip()


# ---------------------------------------------------------------------------
# Legacy: TemporalCaptionGenerator (backward compat wrapper)
# ---------------------------------------------------------------------------

class TemporalCaptionGenerator:
    """Backward-compatible wrapper around IFEditPromptEnhancer."""

    def __init__(self, model: str = DEFAULT_VLM_MODEL):
        self._enhancer = IFEditPromptEnhancer(model=model)
        self.last_plan = None

    def generate(self, image: Image.Image, target_prompt: str) -> str:
        return self._enhancer.enhance(image, target_prompt)


# ---------------------------------------------------------------------------
# Frame collage builder (retained for debugging / visualization)
# ---------------------------------------------------------------------------

def build_frame_collage(
    source_image: Image.Image,
    frames: Sequence[Image.Image],
    sample_step: int = 4,
    max_tiles: Optional[int] = None,
) -> Tuple[Image.Image, Dict[str, int]]:
    """Create annotated collage of source + sampled frames for visualization."""
    if not frames:
        raise ValueError("frames sequence is empty")

    indices = list(range(0, len(frames), max(1, sample_step)))
    if indices[-1] != len(frames) - 1:
        indices.append(len(frames) - 1)

    if max_tiles is None:
        max_tiles = 12
    if max_tiles > 0:
        indices = indices[:max_tiles - 1]
        if len(indices) < max_tiles and indices[-1] != len(frames) - 1:
            indices.append(len(frames) - 1)

    tiles = [("SRC", source_image)]
    id_to_index: Dict[str, int] = {}
    for idx_pos, frame_idx in enumerate(indices, start=1):
        frame_id = f"T{frame_idx:02d}"
        tiles.append((frame_id, frames[frame_idx]))
        id_to_index[frame_id] = frame_idx

    tile_width = 320
    tile_height = int(tile_width * (frames[0].height / frames[0].width)) if frames else 240
    label_height = 36
    padding = 20
    cols = 4
    rows = (len(tiles) + cols - 1) // cols

    collage_w = cols * tile_width + (cols + 1) * padding
    collage_h = rows * (tile_height + label_height) + (rows + 1) * padding

    collage = Image.new("RGB", (collage_w, collage_h), (18, 18, 22))
    draw = ImageDraw.Draw(collage, "RGBA")
    font = ImageFont.load_default()

    for idx, (frame_id, tile_image) in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        x0 = padding + col * (tile_width + padding)
        y0 = padding + row * (tile_height + label_height + padding)

        resized = tile_image.copy().resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        collage.paste(resized, (x0, y0))

        label_y = y0 + tile_height
        draw.rectangle(
            [(x0, label_y), (x0 + tile_width, label_y + label_height)],
            fill=(12, 94, 184),
        )
        text = frame_id
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.text(
            (x0 + (tile_width - text_w) / 2, label_y + (label_height - text_h) / 2),
            text, font=font, fill=(255, 255, 255),
        )

    return collage, id_to_index


# ---------------------------------------------------------------------------
# Legacy dataclasses (retained for backward compatibility)
# ---------------------------------------------------------------------------

@dataclass
class TemporalDelta:
    label: str
    start: float
    end: float
    description: str
    focus_tokens: Tuple[str, ...] = field(default_factory=tuple)
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label, "start": self.start, "end": self.end,
            "description": self.description,
            "focus_tokens": list(self.focus_tokens), "weight": self.weight,
        }


@dataclass
class TemporalPromptPlan:
    anchor: str
    deltas: List[TemporalDelta] = field(default_factory=list)
    constraints: Dict[str, str] = field(default_factory=dict)
    alignment: Dict[str, str] = field(default_factory=dict)
    notes: Optional[str] = None
    raw_response: str = ""

    def to_prompt(self) -> str:
        segments = []
        if self.anchor:
            segments.append(f"Anchor scene: {self.anchor}")
        if self.deltas:
            segments.append("; ".join(d.description for d in self.deltas))
        return " ".join(segments)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anchor": self.anchor,
            "deltas": [d.to_dict() for d in self.deltas],
            "constraints": dict(self.constraints),
            "alignment": dict(self.alignment),
            "notes": self.notes,
        }


@dataclass
class FrameSelectionResult:
    frame_id: str
    frame_index: int
    raw_response: str


class FrameSelector:
    """Legacy frame selector — now just picks sharpest frame via Laplacian."""

    def __init__(self, model: str = DEFAULT_VLM_MODEL):
        self.model = model

    def select(
        self, collage: Image.Image, prompt: str, id_to_index: Dict[str, int]
    ) -> FrameSelectionResult:
        # Just return the last frame as default
        if not id_to_index:
            return FrameSelectionResult(frame_id="T00", frame_index=0, raw_response="fallback")
        last_id = max(id_to_index.keys(), key=lambda k: id_to_index[k])
        return FrameSelectionResult(
            frame_id=last_id,
            frame_index=id_to_index[last_id],
            raw_response="laplacian-based selection",
        )


__all__ = [
    "IFEditPromptEnhancer",
    "TemporalCaptionGenerator",
    "FrameSelector",
    "FrameSelectionResult",
    "TemporalDelta",
    "TemporalPromptPlan",
    "build_frame_collage",
]
