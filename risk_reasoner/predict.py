from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from .data import Sample
from .prompts import build_infer_messages, parse_sections


def get_model_device(model: torch.nn.Module) -> torch.device:
    return next((p.device for p in model.parameters() if p is not None), torch.device("cpu"))


def _move_to_device(inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


@torch.no_grad()
def save_cot_predictions(
    model,
    processor,
    samples: List[Sample],
    out_dir: Path,
    jsonl_path: Path,
    max_length: int = 4096,
    max_new_tokens: int = 256,
    limit: Optional[int] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    device = get_model_device(model)

    n = len(samples) if limit is None else min(len(samples), int(limit))
    print(f"[PredSave] saving {n}/{len(samples)} samples to {out_dir}")

    with jsonl_path.open("w", encoding="utf-8") as f_jsonl:
        for i in range(n):
            s = samples[i]

            img = Image.open(s.image_path).convert("RGB")
            messages = build_infer_messages(s.evidence_str)

            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = processor(
                text=prompt,
                images=img,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            inputs = _move_to_device(inputs, device)

            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

            in_len = inputs["input_ids"].shape[1]
            raw_text = processor.tokenizer.decode(
                gen_ids[0, in_len:],
                skip_special_tokens=True,
            ).strip()

            parsed = parse_sections(raw_text)

            out = {
                "stem": s.stem,
                "image_path": str(s.image_path),
                "gt_grade": s.grade,
                "gt_risk": str(s.risk_line).strip().splitlines()[0].strip() if s.risk_line else "",
                "raw_text": raw_text,
                **parsed,
            }

            (out_dir / f"{s.stem}.json").write_text(
                json.dumps(out, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            f_jsonl.write(json.dumps(out, ensure_ascii=False) + "\n")

            if (i + 1) % 50 == 0:
                print(f"[PredSave] {i+1}/{n}")

    print(f"[PredSave] done. json_dir={out_dir} jsonl={jsonl_path}")
