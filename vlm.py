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

# --- Self-Sufficient Environment Loading ---
# This ensures that whenever this module is imported, it immediately
# loads the .env file, making it independent of app startup order.
from dotenv import load_dotenv
load_dotenv()
# --- End of Self-Sufficient Loading ---

from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI


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
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self._api_key = os.environ.get("OPENAI_API_KEY")
        print(f"[INFO] API Key: {self._api_key}")
        self._base_url = os.environ.get("OPENAI_BASE_URL")
        print(f"[INFO] Base URL: {self._base_url}")
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
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client


class TemporalCaptionGenerator:
    """Generates temporal editing captions using a multimodal VLM."""

    def __init__(
        self,
        model: str = DEFAULT_VLM_MODEL,
    ) -> None:
        self.model = model
        self._client_wrapper = _OpenAIWrapper()
        self.enabled = self._client_wrapper.available

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
            # Revert to a simple, informative error/message if no prompt is given
            raise ValueError("Target edit text cannot be empty.")

        client = self._client_wrapper.client
        if client is None:
            # This should ideally not be reached if dependencies are installed,
            # but serves as a final safeguard.
            raise RuntimeError(
                "OpenAI client could not be initialized. "
                "Please ensure OPENAI_API_KEY is set and the 'openai' library is installed."
            )

        prompt = self._build_prompt(target_prompt)
        image_b64 = _pil_to_base64(image)

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_TEMPORAL},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                            },
                        ],
                    },
                ],
                max_tokens=256,
            )
            text = response.choices[0].message.content.strip()
            return text
        except Exception as exc:
            print(f"[ERROR] Temporal caption VLM call failed: {exc}")
            raise  # Re-raise the exception to make the failure visible


class FrameSelector:
    """Selects the earliest frame satisfying the edit intent using a VLM."""

    def __init__(
        self,
        model: str = DEFAULT_VLM_MODEL,
    ) -> None:
        self.model = model
        self._client_wrapper = _OpenAIWrapper()

    def select(self, collage: Image.Image, prompt: str, id_to_index: Dict[str, int]) -> FrameSelectionResult:
        """Return the selected frame ID and index."""
        client = self._client_wrapper.client
        if client is None:
            raise RuntimeError(
                "OpenAI client could not be initialized. "
                "Please ensure OPENAI_API_KEY is set and the 'openai' library is installed."
            )

        collage_b64 = _pil_to_base64(collage)
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_SELECTION},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt.strip()},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{collage_b64}"},
                            },
                        ],
                    },
                ],
                max_tokens=64,
            )
            raw_text = response.choices[0].message.content.strip()
        except Exception as exc:
            print(f"[ERROR] Frame selector VLM call failed: {exc}")
            raise  # Re-raise the exception

        # Strict parsing of the response
        match = re.search(r"(T\d+)", raw_text, re.IGNORECASE)
        if match:
            selected_id = match.group(1).upper()
            if selected_id in id_to_index:
                return FrameSelectionResult(
                    frame_id=selected_id,
                    frame_index=id_to_index[selected_id],
                    raw_response=raw_text,
                )

        # If parsing fails, raise an error instead of falling back
        raise ValueError(
            f"Could not parse a valid frame ID (e.g., 'T04') from the VLM response. "
            f"Raw response: '{raw_text}'"
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
