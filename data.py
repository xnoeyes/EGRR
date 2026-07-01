from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from . import config as cfg
from .prompts import build_target_text


def norm_grade(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().upper()
    return s if s in {"L", "M", "H"} else None


def safe_read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if suf == ".csv":
        for enc in ["utf-8", "utf-8-sig", "cp949"]:
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                pass
        return pd.read_csv(path)
    raise ValueError(f"Unsupported label file: {path}")


def load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_image_path(image_dir: Path, stem: str, image_file_from_json: Optional[str]) -> Optional[Path]:
    cands: List[Path] = []
    if image_file_from_json:
        cands.append(image_dir / image_file_from_json)
    cands += [
        image_dir / f"{stem}.png",
        image_dir / f"{stem}.jpg",
        image_dir / f"{stem}.jpeg",
        image_dir / f"{stem}.webp",
    ]
    for p in cands:
        if p.exists():
            return p
    return None


def compact_objects(
    objects: List[Dict[str, Any]],
    max_objects: int,
    use_distance: bool = True,
    use_bbox: bool = True,
) -> List[Dict[str, Any]]:
    keep = []
    for o in objects[:max_objects]:
        item = {
            "category_name": o.get("category_name"),
        }
        if use_distance:
            item["distance_m"] = o.get("distance_m")
        if use_bbox:
            item["bbox_xywh"] = o.get("bbox_xywh")
        keep.append(item)
    return keep


def build_evidence_str(
    dist_json: Dict[str, Any],
    max_objects: int,
    use_num_objects: bool = True,
    use_distance: bool = True,
    use_bbox: bool = True,
) -> str:
    if max_objects <= 0:
        return json.dumps(dist_json, ensure_ascii=False, indent=2)

    evidence = {
        "stem": dist_json.get("stem"),
        "distance_mode": dist_json.get("distance_mode", "median"),
        "objects": compact_objects(
            dist_json.get("objects", []),
            max_objects=max_objects,
            use_distance=use_distance,
            use_bbox=use_bbox,
        ),
    }

    if use_num_objects:
        evidence["num_objects"] = dist_json.get("num_objects")

    return json.dumps(evidence, ensure_ascii=False, indent=2)


@dataclass
class Sample:
    stem: str
    image_path: Path
    evidence_str: str
    grade: str
    risk_line: str


class RiskReasonerDataset(Dataset):
    def __init__(self, samples: List[Sample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        target_text = build_target_text(grade=s.grade, risk_line=s.risk_line)
        return {
            "image_path": str(s.image_path),
            "evidence_str": s.evidence_str,
            "target_text": target_text,
            "stem": s.stem,
        }


def build_samples(
    label_path: Path,
    json_dir: Path,
    image_dir: Path,
    max_objects: int,
    seed: int,
) -> List[Sample]:
    df = safe_read_table(label_path)

    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    for col in ["stem", "risk", "total_grade"]:
        if col not in df.columns:
            raise ValueError(f"label file must contain column '{col}'. got: {list(df.columns)}")

    rows: List[Tuple[str, str, Any]] = []
    for _, r in df.iterrows():
        stem = str(r["stem"]).strip()
        grade = norm_grade(r.get("total_grade"))
        if grade is None:
            continue
        risk_line = r.get("risk", "")
        rows.append((stem, grade, risk_line))

    random.Random(seed).shuffle(rows)

    samples: List[Sample] = []
    miss_json = 0
    miss_img = 0

    for stem, grade, risk_line in rows:
        jp = json_dir / f"{stem}.json"
        if not jp.exists():
            miss_json += 1
            continue

        dist_json = load_json(jp)
        img_path = find_image_path(image_dir, stem, dist_json.get("image_file"))
        if img_path is None or not img_path.exists():
            miss_img += 1
            continue

        evidence_str = build_evidence_str(
            dist_json,
            max_objects=max_objects,
            use_num_objects=cfg.USE_NUM_OBJECTS,
            use_distance=cfg.USE_DISTANCE,
            use_bbox=cfg.USE_BBOX,
        )

        samples.append(
            Sample(
                stem=stem,
                image_path=img_path,
                evidence_str=evidence_str,
                grade=grade,
                risk_line=str(risk_line) if risk_line is not None else "",
            )
        )

    print(f"[Data] label_rows(valid grade)= {len(rows)}")
    print(f"[Data] samples(built)= {len(samples)} | miss_json={miss_json} miss_img={miss_img}")
    return samples


def train_val_split(samples: List[Sample], val_ratio: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    rnd = random.Random(seed)
    idx = list(range(len(samples)))
    rnd.shuffle(idx)
    n_val = int(len(samples) * val_ratio)
    val_idx = set(idx[:n_val])
    train = [samples[i] for i in range(len(samples)) if i not in val_idx]
    val = [samples[i] for i in range(len(samples)) if i in val_idx]
    return train, val
