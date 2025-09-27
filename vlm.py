"""Vision-language utilities for Frame2Frame.

This module encapsulates the VLM-powered components described in
"Pathways on the Image Manifold: Image Editing via Video Generation".
It provides two primary helpers:

- TemporalCaptionGenerator: builds temporal editing captions c~ from a
  source image and target text prompt using a multimodal LLM (GPT-4o).
- FrameSelector: scores a collage of sampled video frames and returns the
  earliest frame index that satisfies the editing description.

Both helpers are optional. When the OpenAI client cannot be initialized,
they gracefully fall back to simple heuristics so the rest of the pipeline
remains usable without external dependencies.
"""

from __future__ import annotations

import base64
import io
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont

try:  # Optional dependency; fallbacks handle absence.
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - library may be absent in offline envs.
    OpenAI = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openai import OpenAI as OpenAIClient  # type: ignore
else:
    OpenAIClient = Any

DEFAULT_VLM_MODEL = os.environ.get("FRAME2FRAME_VLM_MODEL", "gpt-4o-mini")

ICL_EXAMPLES = [
    (
        "Smiling woman holding a coffee mug",
        "The woman slowly lifts the mug to her lips and takes a gentle sip while the camera remains steady.",
    ),
    (
        "Man standing beside a red car",
        "The man gradually opens the car door, leans inside to collect a jacket, and closes the door with the camera fixed.",
    ),
    (
        "Child holding balloons in a park",
        "The child gently raises the balloons overhead as a breeze sways them while the camera stays static.",
    ),
    (
        "Chef plating a dessert",
        "The chef slowly drizzles chocolate over the dessert, adds one berry, and steps back without camera motion.",
    ),
    (
        "Golden retriever sitting on grass",
        "The dog gradually stands, trots toward the viewer, and wags its tail while the camera stays still.",
    ),
    (
        "Woman painting at an easel",
        "The painter slowly adds bright strokes, tilts her head to inspect them, and keeps the camera steady.",
    ),
    (
        "Family sitting on a sofa",
        "The family gently leans together into a cozy hug and smiles while the camera remains fixed.",
    ),
    (
        "Sunset city skyline",
        "The skyline transitions slowly from sunset glow to twinkling windows as the camera stays static.",
    ),
    (
        "Person wearing a denim jacket",
        "The person gradually unbuttons the jacket, reveals a bright shirt, and straightens their posture with the camera unmoving.",
    ),
]

SYSTEM_PROMPT_TEMPORAL = (
    "You write temporal editing captions for image-to-video edits. "
    "Describe in ONE English sentence (<=25 words) how the scene gradually evolves over time to satisfy the edit. "
    "Use gradual verbs and keep the camera static unless movement is required by the edit. Do not invent new objects beyond the instruction."
)

SYSTEM_PROMPT_SELECTION = (
    "You assist with frame selection for image editing. "
    "Given a collage of numbered frames and the original instruction, identify the earliest frame that fulfils the edit while staying faithful to the source. "
    "If multiple frames qualify, choose the smallest ID. Respond with ONLY the frame ID (e.g., T03)."
)

COLLAGE_BG = (18, 18, 22)
COLLAGE_TEXT_BG = (12, 94, 184)
COLLAGE_TEXT_COLOR = (255, 255, 255)
COLLAGE_GLYPH = (0, 0, 0, 160)


@dataclass
class FrameSelectionResult:
    """Structured response for frame selection."""

    frame_id: str
    frame_index: int
    raw_response: str


def _pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class _OpenAIWrapper:
    """Lazy OpenAI client loader with graceful fallback."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client: Optional[OpenAIClient] = None

    @property
    def available(self) -> bool:
        return OpenAI is not None and bool(self._api_key)

    @property
    def client(self) -> Optional[OpenAIClient]:
        if not self.available:
            return None
        if self._client is None:  # pragma: no cover (requires network + key)
            if OpenAI is None:
                return None
            self._client = OpenAI(api_key=self._api_key)
        return self._client


class TemporalCaptionGenerator:
    """Generates temporal editing captions using a multimodal VLM."""

    def __init__(
        self,
        model: str = DEFAULT_VLM_MODEL,
        api_key: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        self.model = model
        self._client_wrapper = _OpenAIWrapper(api_key=api_key)
        self.enabled = enabled and self._client_wrapper.available

    def _build_prompt(self, target_prompt: str) -> str:
        exemplar_lines = [
            f"Source: {src}\nTemporal caption: {cap}" for src, cap in ICL_EXAMPLES
        ]
        exemplars = "\n\n".join(exemplar_lines)
        instructions = (
            "Given the source image and the target edit description below, write one English sentence (<=25 words) that explains how the scene evolves over time."
            " Use gradual verbs and keep the camera static unless the edit explicitly requires camera motion."
        )
        return f"{instructions}\n\nExamples:\n{exemplars}\n\nTarget edit: {target_prompt.strip()}"

    def generate(self, image: Image.Image, target_prompt: str) -> str:
        if not target_prompt:
            return "The scene stays almost unchanged while the camera remains still."
        if not self.enabled:
            # Fallback: simple deterministic phrasing incorporating target prompt.
            return (
                "The subject slowly adapts so that "
                f"{target_prompt.strip()} while the camera stays fixed."
            )

        client = self._client_wrapper.client
        if client is None:  # Safety fallback
            return (
                "The subject slowly adapts so that "
                f"{target_prompt.strip()} while the camera stays fixed."
            )

        prompt = self._build_prompt(target_prompt)
        image_b64 = _pil_to_base64(image)

        try:  # pragma: no cover - network dependent
            response = client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_TEMPORAL}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image", "image": {"data": image_b64}},
                        ],
                    },
                ],
                max_output_tokens=256,
            )
            text = getattr(response, "output_text", "").strip()
            return text or (
                "Starting from the original scene, the subject gradually transforms so that "
                f"{target_prompt.strip()}."
            )
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"[WARN] Temporal caption VLM 调用失败: {exc}")
            return (
                "The subject slowly adapts so that "
                f"{target_prompt.strip()} while the camera stays fixed."
            )


class FrameSelector:
    """Selects the earliest frame satisfying the edit intent using a VLM."""

    def __init__(
        self,
        model: str = DEFAULT_VLM_MODEL,
        api_key: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        self.model = model
        self._client_wrapper = _OpenAIWrapper(api_key=api_key)
        self.enabled = enabled and self._client_wrapper.available

    def select(self, collage: Image.Image, prompt: str, id_to_index: Dict[str, int]) -> FrameSelectionResult:
        """Return the selected frame ID and index.

        The VLM is instructed to return just the ID (e.g., "T03"). If the VLM is
        unavailable or fails, we fall back to choosing the largest index (latest frame)
        to mimic the baseline behaviour discussed in the paper's ablation.
        """

        fallback_id = max(id_to_index.items(), key=lambda kv: kv[1])[0]
        fallback_result = FrameSelectionResult(
            frame_id=fallback_id,
            frame_index=id_to_index[fallback_id],
            raw_response="fallback",
        )

        if not self.enabled:
            return fallback_result

        client = self._client_wrapper.client
        if client is None:
            return fallback_result

        collage_b64 = _pil_to_base64(collage)
        try:  # pragma: no cover - network dependent
            response = client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_SELECTION}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt.strip()},
                            {"type": "image", "image": {"data": collage_b64}},
                        ],
                    },
                ],
                max_output_tokens=64,
            )
            raw_text = getattr(response, "output_text", "").strip()
        except Exception as exc:  # pragma: no cover - network dependent
            return FrameSelectionResult(
                frame_id=fallback_id,
                frame_index=id_to_index[fallback_id],
                raw_response=f"error: {exc}",
            )

        if not raw_text:
            return FrameSelectionResult(
                frame_id=fallback_id,
                frame_index=id_to_index[fallback_id],
                raw_response="empty",
            )

        lower = raw_text.lower()
        for candidate_id in sorted(id_to_index.keys()):
            if candidate_id.lower() in lower:
                return FrameSelectionResult(
                    frame_id=candidate_id,
                    frame_index=id_to_index[candidate_id],
                    raw_response=raw_text,
                )
        # Try numeric extraction if explicit ID missing.
        digits = re.findall(r"(\d+)", lower)
        if digits:
            num = int(digits[0])
            padded = f"T{num:02d}"
            if padded in id_to_index:
                return FrameSelectionResult(
                    frame_id=padded,
                    frame_index=id_to_index[padded],
                    raw_response=raw_text,
                )

        return FrameSelectionResult(
            frame_id=fallback_id,
            frame_index=id_to_index[fallback_id],
            raw_response=raw_text,
        )


def build_frame_collage(
    source_image: Image.Image,
    frames: Sequence[Image.Image],
    sample_step: int = 4,
    max_tiles: Optional[int] = None,
) -> Tuple[Image.Image, Dict[str, int]]:
    """Create an annotated collage containing the source and sampled frames.

    Returns the collage image along with a mapping from frame IDs (e.g., "T05")
    to their indices within the original `frames` sequence.
    """

    if not frames:
        raise ValueError("frames sequence is empty")

    indices = list(range(0, len(frames), max(1, sample_step)))
    if indices[-1] != len(frames) - 1:
        indices.append(len(frames) - 1)

    if max_tiles is not None and max_tiles > 0:
        indices = indices[: max_tiles - 1]
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

    collage = Image.new("RGB", (collage_w, collage_h), COLLAGE_BG)
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
            fill=COLLAGE_TEXT_BG,
        )
        text = frame_id
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.text(
            (x0 + (tile_width - text_w) / 2, label_y + (label_height - text_h) / 2),
            text,
            font=font,
            fill=COLLAGE_TEXT_COLOR,
        )

    return collage, id_to_index


__all__ = [
    "TemporalCaptionGenerator",
    "FrameSelector",
    "FrameSelectionResult",
    "build_frame_collage",
]
