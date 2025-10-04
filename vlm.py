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
import json
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

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

SYSTEM_PROMPT_TEMPORAL_STRUCTURED = (
    "You are a prompt optimization planner for text-to-video diffusion editing. "
    "Given a source frame and a textual edit, respond ONLY with JSON following this schema: "
    "{\"anchor\": str, \"deltas\": [{\"label\": str, \"start\": float, \"end\": float, \"description\": str, \"focus_tokens\": [str], \"weight\": float}], "
    "\"constraints\": {str: str}, \"alignment\": {\"L_subj\": str, \"L_prompt\": str, \"L_fid\": str}, \"notes\": str}. "
    "Normalize start/end to [0, 1], produce 2-4 deltas ordered over time, keep camera static unless edit requires movement, "
    "and ensure alignment directives reflect identity, prompt, and fidelity preservation (MagDiff style)."
)

LEGACY_TEMPORAL_STYLE_INSTRUCTIONS = (
    "Maintain the original scene composition, describe motion with gradual verbs, and keep the camera static unless explicitly instructed."
)

STRUCTURED_PLAN_EXAMPLES: List[Dict[str, Any]] = [
    {
        "source": "Wooden door set in a white wall with a brass handle, afternoon light.",
        "edit": "Close the door slowly.",
        "plan": {
            "anchor": "A wooden door mounted in a white painted wall with a brass handle, viewed at eye level with soft afternoon light.",
            "deltas": [
                {
                    "label": "ease_in",
                    "start": 0.0,
                    "end": 0.35,
                    "description": "Door begins to rotate inward up to 20 degrees; keep handle visible and background rigid.",
                    "focus_tokens": ["door", "handle"],
                    "weight": 0.6,
                },
                {
                    "label": "mid_swing",
                    "start": 0.35,
                    "end": 0.7,
                    "description": "Door continues closing toward the frame, reducing the visible gap while hinge side stays fixed.",
                    "focus_tokens": ["door", "hinge"],
                    "weight": 0.9,
                },
                {
                    "label": "settle",
                    "start": 0.7,
                    "end": 1.0,
                    "description": "Door eases into the fully closed position with a slight handle settle; wall and lighting unchanged.",
                    "focus_tokens": ["door", "handle"],
                    "weight": 1.0,
                },
            ],
            "constraints": {
                "camera": "Static tripod, no dolly or zoom.",
                "immutable_regions": "Preserve white wall, floor, and ambient lighting.",
            },
            "alignment": {
                "L_subj": "Keep the same door identity and brass handle texture.",
                "L_prompt": "Align motion with the instruction 'door slowly closes'.",
                "L_fid": "Retain high-frequency wood grain and wall cleanliness.",
            },
            "notes": "Feather the motion so closing speed decelerates near the end.",
        },
    },
    {
        "source": "Desk with a matte black table lamp beside a closed notebook in warm interior lighting.",
        "edit": "Turn the lamp on and brighten the scene slightly.",
        "plan": {
            "anchor": "A matte black table lamp on a wooden desk with a closed notebook and soft warm ambient light, camera locked off.",
            "deltas": [
                {
                    "label": "initial_glow",
                    "start": 0.0,
                    "end": 0.25,
                    "description": "Lamp switch toggles, filament begins to glow with subtle bloom around the shade interior.",
                    "focus_tokens": ["lamp", "switch"],
                    "weight": 0.7,
                },
                {
                    "label": "ramp_up",
                    "start": 0.25,
                    "end": 0.65,
                    "description": "Lamp output increases, casting warmer light onto the desk and notebook without altering other objects.",
                    "focus_tokens": ["lamp", "light", "desk"],
                    "weight": 0.9,
                },
                {
                    "label": "stabilize",
                    "start": 0.65,
                    "end": 1.0,
                    "description": "Scene reaches steady illumination with gentle falloff; maintain shadows consistent with new light source.",
                    "focus_tokens": ["lamp", "glow"],
                    "weight": 1.0,
                },
            ],
            "constraints": {
                "camera": "Do not introduce camera motion or cropping.",
                "immutable_regions": "Keep desk objects and background decor fixed.",
            },
            "alignment": {
                "L_subj": "Lamp geometry and materials must remain unchanged.",
                "L_prompt": "Match the textual intent 'lamp turns on and brightens scene'.",
                "L_fid": "Avoid overexposure; preserve desk grain and notebook detail.",
            },
            "notes": "Diffuse light softly to avoid harsh clipping at the shade rim.",
        },
    },
]


@dataclass
class TemporalDelta:
    label: str
    start: float
    end: float
    description: str
    focus_tokens: Tuple[str, ...] = field(default_factory=tuple)
    weight: float = 1.0

    @staticmethod
    def _coerce_time(value: Any, default: float) -> float:
        if value is None:
            return float(default)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            parsed = value.strip()
            if not parsed:
                return float(default)
            if parsed.endswith("%"):
                try:
                    return float(parsed[:-1]) / 100.0
                except ValueError:
                    return float(default)
            try:
                return float(parsed)
            except ValueError:
                return float(default)
        return float(default)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], fallback_label: str) -> "TemporalDelta":
        label = str(data.get("label") or data.get("name") or fallback_label)
        range_field = data.get("time_range") or data.get("range") or ""
        range_start = None
        range_end = None
        if isinstance(range_field, str) and "-" in range_field:
            parts = [part.strip() for part in range_field.split("-", 1)]
            if len(parts) == 2:
                range_start, range_end = parts
        start = cls._coerce_time(
            data.get("start") or data.get("t_start") or range_start or data.get("time") or data.get("t"),
            0.0,
        )
        end = cls._coerce_time(
            data.get("end") or data.get("t_end") or range_end or data.get("time_end"),
            start if start != 0.0 else 1.0,
        )
        description = str(
            data.get("description")
            or data.get("delta")
            or data.get("prompt")
            or data.get("text")
            or "Describe the temporal change."
        )
        tokens_raw = data.get("focus_tokens") or data.get("tokens") or data.get("keywords") or []
        focus_tokens: Tuple[str, ...]
        if isinstance(tokens_raw, str):
            focus_tokens = tuple(token.strip() for token in re.split(r"[,/]|\s", tokens_raw) if token.strip())
        elif isinstance(tokens_raw, Sequence):  # type: ignore[arg-type]
            focus_tokens = tuple(str(token) for token in tokens_raw if str(token).strip())
        else:
            focus_tokens = tuple()
        weight_raw = data.get("weight") or data.get("alpha") or data.get("emphasis")
        try:
            weight = float(weight_raw) if weight_raw is not None else 1.0
        except (TypeError, ValueError):
            weight = 1.0
        return cls(
            label=label,
            start=float(max(0.0, min(1.0, start))),
            end=float(max(0.0, min(1.0, end))),
            description=description.strip(),
            focus_tokens=focus_tokens,
            weight=weight,
        )

    def to_prompt_segment(self) -> str:
        span = f"t={self.start:.2f}->{self.end:.2f}"
        segment = f"[{self.label}] {span}: {self.description}"
        if self.focus_tokens:
            segment += f" (focus tokens: {', '.join(self.focus_tokens)})"
        if abs(self.weight - 1.0) > 1e-3:
            segment += f" [weight={self.weight:.2f}]"
        return segment

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "description": self.description,
            "focus_tokens": list(self.focus_tokens),
            "weight": self.weight,
        }


@dataclass
class TemporalPromptPlan:
    anchor: str
    deltas: List[TemporalDelta] = field(default_factory=list)
    constraints: Dict[str, str] = field(default_factory=dict)
    alignment: Dict[str, str] = field(default_factory=dict)
    notes: Optional[str] = None
    raw_response: str = ""

    @staticmethod
    def _coerce_mapping(value: Any) -> Dict[str, str]:
        if isinstance(value, dict):
            return {str(k): str(v).strip() for k, v in value.items() if v is not None}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            mapping: Dict[str, str] = {}
            for idx, item in enumerate(value, start=1):
                if isinstance(item, dict):
                    mapping.update({str(k): str(v).strip() for k, v in item.items() if v is not None})
                else:
                    mapping[f"item_{idx}"] = str(item)
            return mapping
        if value is None:
            return {}
        return {"info": str(value)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any], raw_response: str = "") -> "TemporalPromptPlan":
        anchor = str(
            data.get("anchor")
            or data.get("anchor_prompt")
            or data.get("scene")
            or "Maintain the original scene as the static anchor."
        ).strip()
        deltas_raw = data.get("deltas") or data.get("delta") or []
        if isinstance(deltas_raw, dict):
            deltas_iterable = [deltas_raw]
        elif isinstance(deltas_raw, Sequence) and not isinstance(deltas_raw, (str, bytes)):
            deltas_iterable = list(deltas_raw)
        else:
            deltas_iterable = []
        deltas: List[TemporalDelta] = []
        for idx, entry in enumerate(deltas_iterable, start=1):
            if isinstance(entry, dict):
                deltas.append(TemporalDelta.from_dict(entry, f"delta_{idx}"))
        constraints = cls._coerce_mapping(data.get("constraints"))
        alignment = cls._coerce_mapping(data.get("alignment") or data.get("alignments"))
        notes = None
        if data.get("notes"):
            notes = str(data.get("notes")).strip()
        elif data.get("comment"):
            notes = str(data.get("comment")).strip()
        return cls(
            anchor=anchor or "Maintain the original scene as the static anchor.",
            deltas=deltas,
            constraints=constraints,
            alignment=alignment,
            notes=notes,
            raw_response=raw_response,
        )

    def to_prompt(self) -> str:
        segments: List[str] = []
        if self.anchor:
            segments.append(f"Anchor scene: {self.anchor}")
        if self.constraints:
            constraint_text = "; ".join(f"{key}: {value}" for key, value in self.constraints.items())
            segments.append(f"Constraints: {constraint_text}")
        if self.deltas:
            delta_text = "; ".join(delta.to_prompt_segment() for delta in self.deltas)
            segments.append(f"Delta schedule: {delta_text}")
        if self.alignment:
            alignment_text = "; ".join(f"{key}: {value}" for key, value in self.alignment.items())
            segments.append(f"Alignment directives: {alignment_text}")
        if self.notes:
            segments.append(f"Notes: {self.notes}")
        return " ".join(segment for segment in segments if segment)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anchor": self.anchor,
            "deltas": [delta.to_dict() for delta in self.deltas],
            "constraints": dict(self.constraints),
            "alignment": dict(self.alignment),
            "notes": self.notes,
            "raw_response": self.raw_response,
        }


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
        self._base_url = os.environ.get("OPENAI_BASE_URL")
        self._client: Optional[OpenAIClient] = None

    @property
    def available(self) -> bool:
        return OpenAI is not None and bool(self._api_key)

    @property
    def client(self) -> Optional[OpenAIClient]:
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
        except Exception as exc:  # pragma: no cover - best effort logging
            print(f"[OpenAIWrapper] Failed to initialize client: {exc}")
            self._client = None
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
        self.last_plan: Optional[TemporalPromptPlan] = None

    @staticmethod
    def _extract_tokens_from_prompt(prompt: str, top_k: int = 4) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9]+", prompt.lower())
        seen = set()
        ordered: List[str] = []
        for token in tokens:
            if len(token) <= 2:
                continue
            if token in seen:
                continue
            seen.add(token)
            ordered.append(token)
            if len(ordered) >= top_k:
                break
        return ordered

    def _build_structured_prompt(self, target_prompt: str) -> str:
        example_blocks: List[str] = []
        for idx, example in enumerate(STRUCTURED_PLAN_EXAMPLES, start=1):
            plan_json = json.dumps(example["plan"], ensure_ascii=False, indent=2)
            block = textwrap.dedent(
                f"""
                Example {idx}:
                Source summary: {example['source']}
                Target edit: {example['edit']}
                Expected JSON:
                {plan_json}
                """
            ).strip()
            example_blocks.append(block)
        examples_text = "\n\n".join(example_blocks)
        schema_text = textwrap.dedent(
            """
            Analyze the provided source frame and target edit. Produce JSON with fields:
            {
              "anchor": "static scene description rooted in the source frame",
              "deltas": [
                {
                  "label": "short stage name",
                  "start": 0.0,
                  "end": 1.0,
                  "description": "temporal delta phrased with gradual motion",
                  "focus_tokens": ["key tokens controlling ROI"],
                  "weight": 0.0-1.5 emphasis weight
                }
              ],
              "constraints": { "camera": "...", "immutable_regions": "..." },
              "alignment": { "L_subj": "identity lock", "L_prompt": "prompt consistency", "L_fid": "detail fidelity" },
              "notes": "optional execution tips"
            }
            Normalize start/end to [0,1], keep deltas ordered without overlaps, and avoid inventing new objects.
            """
        ).strip()
        return (
            f"{schema_text}\n\n"
            f"Examples:\n{examples_text}\n\n"
            f"Target edit: {target_prompt.strip()}"
        )

    def _parse_plan_response(self, raw_text: str, target_prompt: str) -> Optional[TemporalPromptPlan]:
        candidate = raw_text.strip()
        if not candidate:
            return None
        if "```" in candidate:
            fences = re.findall(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL)
            if fences:
                candidate = fences[0]
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        plan = TemporalPromptPlan.from_dict(parsed, raw_response=raw_text)
        if not plan.deltas:
            tokens = tuple(self._extract_tokens_from_prompt(target_prompt))
            plan.deltas.append(
                TemporalDelta(
                    label="delta_1",
                    start=0.0,
                    end=1.0,
                    description=target_prompt.strip(),
                    focus_tokens=tokens[:3],
                )
            )
        return plan

    def _fallback_plan(self, target_prompt: str, raw_text: str = "") -> TemporalPromptPlan:
        tokens = tuple(self._extract_tokens_from_prompt(target_prompt))
        delta = TemporalDelta(
            label="full_motion",
            start=0.0,
            end=1.0,
            description=target_prompt.strip(),
            focus_tokens=tokens[:3],
            weight=1.0,
        )
        alignment = {
            "L_subj": "Preserve original subject identity and surrounding context.",
            "L_prompt": f"Ensure evolution fulfills: {target_prompt.strip()}",
            "L_fid": "Keep lighting and background fidelity unchanged except for the edited region.",
        }
        constraints = {
            "camera": "Lock camera position; no new parallax.",
            "immutable_regions": "Avoid altering regions outside the described action.",
        }
        return TemporalPromptPlan(
            anchor="Use the source frame appearance as the static anchor scene.",
            deltas=[delta],
            constraints=constraints,
            alignment=alignment,
            notes=LEGACY_TEMPORAL_STYLE_INSTRUCTIONS,
            raw_response=raw_text,
        )

    def generate_plan(self, image: Image.Image, target_prompt: str) -> TemporalPromptPlan:
        if not target_prompt:
            raise ValueError("Target edit text cannot be empty.")
        client = self._client_wrapper.client
        if client is None:
            raise RuntimeError(
                "OpenAI client could not be initialized. "
                "Please ensure OPENAI_API_KEY is set and the 'openai' library is installed."
            )
        prompt = self._build_structured_prompt(target_prompt)
        image_b64 = _pil_to_base64(image)
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_TEMPORAL_STRUCTURED},
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
                max_tokens=384,
            )
            raw_text = response.choices[0].message.content.strip()
            print("[TemporalCaptionGenerator] raw_vlm_response:", raw_text)
        except Exception as exc:
            print(f"[ERROR] Temporal caption VLM call failed: {exc}")
            raise
        plan = self._parse_plan_response(raw_text, target_prompt)
        if plan is None:
            plan = self._fallback_plan(target_prompt, raw_text)
        self.last_plan = plan
        return plan

    def generate(self, image: Image.Image, target_prompt: str) -> str:
        plan = self.generate_plan(image, target_prompt)
        return plan.to_prompt()


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

        # ---- Logging: inputs overview ----
        try:
            sorted_ids = sorted(id_to_index.keys(), key=lambda k: id_to_index[k])
            preview_ids_head = ", ".join(sorted_ids[:6])
            preview_ids_tail = ", ".join(sorted_ids[-3:]) if len(sorted_ids) > 6 else ""
            w, h = collage.size
            print("[FrameSelector] prompt:", prompt)
            print(f"[FrameSelector] collage_size: {w}x{h}")
            print(f"[FrameSelector] candidate_count: {len(sorted_ids)}")
            if preview_ids_tail:
                print(f"[FrameSelector] candidate_ids(sample): {preview_ids_head} ... {preview_ids_tail}")
            else:
                print(f"[FrameSelector] candidate_ids: {preview_ids_head}")
        except Exception as _log_exc:
            print(f"[FrameSelector][warn] failed to log inputs: {_log_exc}")
        
        # Enhanced user prompt with more context
        enhanced_prompt = f"""
Original instruction: {prompt.strip()}

Please analyze the frame sequence and select the frame that shows the MOST COMPLETE realization of the requested action. 
For door closing: choose the frame where the door is most closed.
For light dimming: choose the frame where the light is most dim.
For any gradual change: choose the frame where the change is most advanced.

Consider the progression from SRC (source) through the numbered frames (T00, T01, etc.) and pick the frame with the highest completion level.
"""
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You assist with frame selection for image editing. Given a collage of numbered frames and the original instruction, identify the frame that best represents the COMPLETION of the requested action. For actions like 'closing a door', 'turning off lights', or 'extinguishing flames', choose the frame where the action is MOST COMPLETE, not just beginning. Look for frames where the intended change is fully realized and visually clear. If multiple frames show the action at different completion levels, choose the one with the highest completion. Respond with ONLY the frame ID (e.g., T03)."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": enhanced_prompt},
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
            print("[FrameSelector] raw_vlm_response:", raw_text)
        except Exception as exc:
            print(f"[ERROR] Frame selector VLM call failed: {exc}")
            raise  # Re-raise the exception

        # Strict parsing of the response
        match = re.search(r"(T\d+)", raw_text, re.IGNORECASE)
        if match:
            selected_id = match.group(1).upper()
            print("[FrameSelector] parsed_id:", selected_id)
            if selected_id in id_to_index:
                selected_index = id_to_index[selected_id]
                print(f"[FrameSelector] final_selection: id={selected_id}, index={selected_index}")
                return FrameSelectionResult(
                    frame_id=selected_id,
                    frame_index=selected_index,
                    raw_response=raw_text,
                )
            else:
                print(f"[FrameSelector][warn] parsed id not in candidates: {selected_id}")

        # If parsing fails, raise an error instead of falling back
        print("[FrameSelector][error] unable to parse a valid Txx from response.")
        raise ValueError(
            f"Could not parse a valid frame ID (e.g., 'T04') from the VLM response. Raw response: '{raw_text}'"
        )


def build_frame_collage(
    source_image: Image.Image,
    frames: Sequence[Image.Image],
    sample_step: int = 2,  # Reduced from 4 to 2 for better frame coverage
    max_tiles: Optional[int] = None,
) -> Tuple[Image.Image, Dict[str, int]]:
    """Create an annotated collage containing the source and sampled frames.

    Returns the collage image along with a mapping from frame IDs (e.g., "T05")
    to their indices within the original `frames` sequence.
    """

    if not frames:
        raise ValueError("frames sequence is empty")

    # More aggressive sampling for better action progression visibility
    indices = list(range(0, len(frames), max(1, sample_step)))
    if indices[-1] != len(frames) - 1:
        indices.append(len(frames) - 1)

    # Increase max_tiles to show more frames if not specified
    if max_tiles is None:
        max_tiles = 12  # Increased default to show more frames
    
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
            text,
            font=font,
            fill=(255, 255, 255),
        )

    return collage, id_to_index


__all__ = [
    "TemporalDelta",
    "TemporalPromptPlan",
    "TemporalCaptionGenerator",
    "FrameSelector",
    "FrameSelectionResult",
    "build_frame_collage",
]
