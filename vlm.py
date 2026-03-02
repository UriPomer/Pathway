"""IF-Edit VLM Utilities: CoT Prompt Enhancement for Zero-Shot Image Editing.

Implements the Chain-of-Thought (CoT) temporal prompt enhancement from
"Are Image-to-Video Models Good Zero-Shot Image Editors?" (arXiv: 2511.19435).

The key insight: Instead of directly using the edit instruction as the I2V prompt,
we use a VLM to generate a temporal description of how the scene should evolve
from the source state to the edited state.

Two prompt enhancer backends:
1. IFEditPromptEnhancer — Uses OpenAI-compatible API (GPT-4o / Qwen-VL)
2. Local Qwen-VL (IFEDIT_USE_LOCAL_VLM=1)
"""

from __future__ import annotations

import base64
import io
import os
import textwrap
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

from PIL import Image
from openai import OpenAI
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
        strict: bool = False,
    ):
        self.model = model
        self.use_local_vlm = use_local_vlm or os.environ.get("IFEDIT_USE_LOCAL_VLM") == "1"
        self.strict = strict
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
            if self.strict:
                raise RuntimeError("CoT 生成失败：本地 VLM 不可用或调用失败。")

        # Try OpenAI API
        if self._openai_wrapper.available:
            result = self._enhance_openai(image, edit_instruction)
            if result:
                return result
            if self.strict:
                raise RuntimeError("CoT 生成失败：远程 VLM 调用失败。")

        # Heuristic fallback
        if self.strict:
            raise RuntimeError("CoT 生成失败：未检测到可用 VLM 后端。")
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

        Minimal wrapper so the model sees the user instruction clearly;
        long templates dilute the instruction and can cause confusion.
        """
        instruction = edit_instruction.strip()
        return f"Static camera. {instruction}"


__all__ = ["IFEditPromptEnhancer"]
