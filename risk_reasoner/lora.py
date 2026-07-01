from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model


def apply_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05) -> torch.nn.Module:
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    return get_peft_model(model, lora_cfg)
