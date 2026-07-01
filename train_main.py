from __future__ import annotations

import random

import torch
from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    set_seed,
    TrainerCallback,
)

try:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForVision2Seq

from peft import prepare_model_for_kbit_training

from risk_reasoner import config as cfg
from risk_reasoner.data import RiskReasonerDataset, build_samples, train_val_split
from risk_reasoner.collator import QwenVLMDataCollator
from risk_reasoner.lora import apply_lora
from risk_reasoner.predict import save_cot_predictions
from risk_reasoner.callbacks import CotSaverCallback


def main():
    assert cfg.LABEL_PATH.exists(), f"Label file not found: {cfg.LABEL_PATH}"
    assert cfg.JSON_DIR.exists(), f"JSON_DIR not found: {cfg.JSON_DIR}"
    assert cfg.IMAGE_DIR.exists(), f"IMAGE_DIR not found: {cfg.IMAGE_DIR}"
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.SEED)

    # Build samples
    samples = build_samples(
        label_path=cfg.LABEL_PATH,
        json_dir=cfg.JSON_DIR,
        image_dir=cfg.IMAGE_DIR,
        max_objects=cfg.MAX_OBJECTS,
        seed=cfg.SEED,
    )
    if len(samples) == 0:
        raise RuntimeError("No training samples built. Check stem matching, paths, and label columns.")

    train_s, val_s = train_val_split(samples, val_ratio=cfg.VAL_RATIO, seed=cfg.SEED)
    print(f"[Split] train={len(train_s)} val={len(val_s)}")

    # Processor / Model (QLoRA)
    processor = AutoProcessor.from_pretrained(
        cfg.MODEL_NAME,
        trust_remote_code=cfg.TRUST_REMOTE_CODE,
        min_pixels=cfg.MIN_PIXELS,
        max_pixels=cfg.MAX_PIXELS,
    )

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if cfg.BF16 else torch.float16,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        cfg.MODEL_NAME,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=cfg.TRUST_REMOTE_CODE,
    )

    # TF32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    model = prepare_model_for_kbit_training(model)
    model = apply_lora(model, r=cfg.LORA_R, alpha=cfg.LORA_ALPHA, dropout=cfg.LORA_DROPOUT)
    model.print_trainable_parameters()

    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Datasets / Collator
    train_ds = RiskReasonerDataset(train_s)
    val_ds = RiskReasonerDataset(val_s)
    collator = QwenVLMDataCollator(processor=processor, max_length=cfg.MAX_LENGTH)

    # Training args
    targs = TrainingArguments(
        output_dir=str(cfg.OUTPUT_DIR),
        num_train_epochs=cfg.EPOCHS,
        learning_rate=cfg.LR,
        weight_decay=cfg.WD,
        per_device_train_batch_size=cfg.BSZ,
        per_device_eval_batch_size=cfg.BSZ,
        gradient_accumulation_steps=cfg.GAS,
        logging_steps=20,

        # transformers 최신: evaluation_strategy -> eval_strategy
        eval_strategy="steps" if len(val_ds) > 0 else "no",
        eval_steps=200,

        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,

        report_to="none",
        remove_unused_columns=False,
        bf16=cfg.BF16,
        fp16=cfg.FP16,
        dataloader_num_workers=cfg.NUM_WORKERS,
        gradient_checkpointing=True,
    )

    rnd = random.Random(cfg.SEED)
    val_subset = val_s[:]
    rnd.shuffle(val_subset)
    val_subset = val_subset[: min(cfg.COT_SAVE_SUBSET_N, len(val_subset))]

    callbacks: list[TrainerCallback] = []
    if cfg.SAVE_COT_DURING_TRAIN and len(val_subset) > 0:
        callbacks.append(
            CotSaverCallback(
                processor=processor,
                samples_subset=val_subset,
                root_dir=cfg.COT_DURING_ROOT,
                max_length=cfg.MAX_LENGTH,
                max_new_tokens=cfg.COT_DURING_MAX_NEW_TOKENS,
            )
        )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds if len(val_ds) > 0 else None,
        data_collator=collator,
        callbacks=callbacks,
    )

    trainer.train()

    # Save final
    final_dir = cfg.OUTPUT_DIR / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"[Done] saved to: {final_dir}")

    # Full-val generation save
    save_cot_predictions(
        model=model,
        processor=processor,
        samples=val_s,
        out_dir=cfg.PRED_OUT_DIR,
        jsonl_path=cfg.PRED_JSONL,
        max_length=cfg.MAX_LENGTH,
        max_new_tokens=cfg.PRED_MAX_NEW_TOKENS,
        limit=cfg.PRED_LIMIT,
    )


if __name__ == "__main__":
    main()
