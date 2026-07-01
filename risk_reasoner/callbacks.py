from __future__ import annotations

from pathlib import Path
from typing import List

from transformers import TrainerCallback

from . import config as cfg
from .data import Sample
from .predict import save_cot_predictions


class CotSaverCallback(TrainerCallback):
    def __init__(
        self,
        processor,
        samples_subset: List[Sample],
        root_dir: Path,
        max_length: int,
        max_new_tokens: int,
    ):
        self.processor = processor
        self.samples_subset = samples_subset
        self.root_dir = root_dir
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if hasattr(args, "local_rank") and args.local_rank not in (-1, 0):
            return control
        if not cfg.SAVE_COT_DURING_TRAIN or not cfg.COT_SAVE_ON_EVAL:
            return control
        if model is None:
            return control

        step = int(state.global_step)
        out_dir = self.root_dir / f"step_{step:07d}"
        jsonl_path = out_dir / "pred.jsonl"

        save_cot_predictions(
            model=model,
            processor=self.processor,
            samples=self.samples_subset,
            out_dir=out_dir,
            jsonl_path=jsonl_path,
            max_length=self.max_length,
            max_new_tokens=self.max_new_tokens,
            limit=len(self.samples_subset),
        )
        return control
