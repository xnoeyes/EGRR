from __future__ import annotations

from typing import Any, Dict, List

import torch
from PIL import Image

from .prompts import make_system_text, make_user_text


class QwenVLMDataCollator:
    def __init__(self, processor, max_length: int = 4096):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images: List[Image.Image] = []
        full_texts: List[str] = []
        prompt_texts: List[str] = []

        for ex in batch:
            img = Image.open(ex["image_path"]).convert("RGB")
            evidence_str = ex["evidence_str"]
            target_text = ex["target_text"]

            messages = [
                {"role": "system", "content": make_system_text()},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": make_user_text(evidence_str)},
                    ],
                },
            ]

            prompt_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            full_messages = messages + [{"role": "assistant", "content": target_text}]
            full_text = self.processor.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            images.append(img)
            full_texts.append(full_text)
            prompt_texts.append(prompt_text)

        enc = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # labels = input_ids with prompt part masked (-100)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels = input_ids.clone()

        for i, ptxt in enumerate(prompt_texts):
            p_ids = self.tokenizer(
                ptxt,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
            )["input_ids"]
            prompt_len = min(len(p_ids), labels.size(1))
            labels[i, :prompt_len] = -100

        # also mask pad
        labels[attention_mask == 0] = -100
        enc["labels"] = labels

        return enc
