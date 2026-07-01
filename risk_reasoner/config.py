from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# repo root = riskreasoner/ (this file lives in riskreasoner/risk_reasoner/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default_relative: str) -> Path:
    return Path(os.environ.get(name, str(PROJECT_ROOT / default_relative)))


# Paths (override via environment variables — see .env.example)
LABEL_PATH = _env_path("RISKREASONER_LABEL_PATH", "datasets/______")
JSON_DIR   = _env_path("RISKREASONER_JSON_DIR", "outputs/______")
IMAGE_DIR  = _env_path("RISKREASONER_IMAGE_DIR", "datasets/______")
OUTPUT_DIR = _env_path("RISKREASONER_OUTPUT_DIR", "ckpts/______")

MODEL_NAME = os.environ.get("RISKREASONER_MODEL_NAME", "Qwen/Qwen2-VL-7B-Instruct")
TRUST_REMOTE_CODE = True

# Ablation flags (evidence 필드 on/off)
USE_NUM_OBJECTS = True
USE_DISTANCE = False
USE_BBOX = True

# Training
SEED = 42
VAL_RATIO = 0.05
EPOCHS = 2
LR = 2e-4
BSZ = 1
GAS = 8
WD = 0.0
MAX_LENGTH = 4096
MAX_OBJECTS = 64

# QLoRA / LoRA
BF16 = True
FP16 = False
NUM_WORKERS = 0
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28

# (SAVE) CoT-like predictions DURING training (on_evaluate)
SAVE_COT_DURING_TRAIN = True
COT_SAVE_ON_EVAL = True
COT_SAVE_SUBSET_N = 32
COT_DURING_MAX_NEW_TOKENS = 256
COT_DURING_ROOT = OUTPUT_DIR / "pred_cot_steps"

# (SAVE) After-train prediction saving (full val)
PRED_OUT_DIR = OUTPUT_DIR / "pred_cot_json"
PRED_JSONL   = OUTPUT_DIR / "pred_cot.jsonl"
PRED_MAX_NEW_TOKENS = 256
PRED_LIMIT: Optional[int] = None
