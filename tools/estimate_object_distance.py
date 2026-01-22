from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None

# =================================CONFIG==================================
IMAGE_DIR = Path("/home/sen/sen2025/riskreasoner/datasets/image_sampled")
LIDAR_DIR = Path("/home/sen/sen2025/riskreasoner/datasets/lidar_sampled")
YOLO_JSON_DIR = Path("/home/sen/sen2025/riskreasoner/outputs2/yolo_json")

OUT_DIR = Path("/home/sen/sen2025/riskreasoner/outputs/dist_from_yolo")
OUT_IMG_DIR = OUT_DIR / "result_images"
OUT_JSON_DIR = OUT_DIR / "result_json"
OUT_JSONL = OUT_DIR / "dist_results.jsonl"

LIDAR_SCALE = 1.0
MAX_POINTS = 200000
MIN_Z = 0.1
MAX_Z = 200.0

DRAW_PROJECTED_POINTS = True
POINT_STRIDE = 1
USE_OPEN3D = True
DEBUG_FIRST_N = 0  

DIST_MODE = "median"
MIN_PTS_IN_BBOX = 20

# LiDAR ↔ 카메라 좌표 변환용 행렬
DEFAULT_EXTRINSIC = np.array(
    [
        [-0.039504, -0.112135, 0.992907, -0.033354],
        [-0.999218, 0.006204, -0.039054, 0.017872],
        [-0.001780, -0.993674, -0.112293, -0.208281],
        [0.000000, 0.000000, 0.000000, 1.000000],
    ],
    dtype=np.float64,
)

# 카메라 내부 파라미터
DEFAULT_K = np.array(
    [
        [2155.703871, 0.0, 942.229476],
        [0.0, 2165.900519, 574.764416],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# 렌즈 왜곡 계수
DEFAULT_DIST = np.array([-0.159335, 0.057244, 0.0, 0.0, 0.0], dtype=np.float64)

INVERT_EXTRINSIC = True
USE_DISTORTION = True
# ==========================================================================

# utils
def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def find_image_file(stem: str) -> Optional[Path]:
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
    for ext in exts:
        p = IMAGE_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    hits = list(IMAGE_DIR.glob(f"{stem}.*"))
    return hits[0] if hits else None


def extract_bboxes_from_yolo(yolo_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    objs = yolo_obj.get("objects", [])
    if not isinstance(objs, list):
        return out

    for i, o in enumerate(objs):
        if not isinstance(o, dict):
            continue

        bbox = o.get("bbox_xywh", None)
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue

        cx, cy, w, h = map(float, bbox)
        x = cx - w / 2.0
        y = cy - h / 2.0

        out.append(
            {
                "ann_idx": int(o.get("ann_idx", i)),
                "category_name": str(o.get("category_name", "Unknown")),
                "confidence": float(o.get("confidence", 0.0)),
                "bbox_xywh": [x, y, w, h],          # 좌상단 xywh
                "bbox_cxcywh": [cx, cy, w, h],      # 원본 cxcywh
            }
        )
    return out


# point cloud load
def load_pcd_xyz(pcd_path: Path) -> np.ndarray:
    if USE_OPEN3D:
        if o3d is None:
            raise RuntimeError("open3d가 설치되어있지 않습니다.")
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        pts = np.asarray(pcd.points, dtype=np.float64)
    else:
        raise NotImplementedError("USE_OPEN3D=False는 미구현")

    if pts.size == 0:
        return pts

    if LIDAR_SCALE != 1.0:
        pts = pts * LIDAR_SCALE

    if len(pts) > MAX_POINTS:
        idx = np.random.choice(len(pts), MAX_POINTS, replace=False)
        pts = pts[idx]

    return pts


# LiDAR points to image plane
def project_points(
    pts_lidar: np.ndarray, K: np.ndarray, dist: np.ndarray, extr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ones = np.ones((pts_lidar.shape[0], 1), dtype=np.float64)
    pts_h = np.concatenate([pts_lidar, ones], axis=1)

    T = extr.copy()
    if INVERT_EXTRINSIC:
        T = np.linalg.inv(T)

    cam_h = (T @ pts_h.T).T
    X, Y, Z = cam_h[:, 0], cam_h[:, 1], cam_h[:, 2]

    m = (Z > MIN_Z) & (Z < MAX_Z)
    X, Y, Z = X[m], Y[m], Z[m]
    if X.size == 0:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )

    d = np.sqrt(X * X + Y * Y + Z * Z)
    x = X / Z
    y = Y / Z

    if USE_DISTORTION and dist is not None and len(dist) >= 5:
        k1, k2, p1, p2, k3 = dist[:5]
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        x = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        y = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y

    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    uv = np.stack([u, v], axis=1)
    return uv, Z, d


def depth_stat_in_bbox(
    uv: np.ndarray, val: np.ndarray, bbox_xywh: List[float], mode: str = "median"
) -> Tuple[Optional[float], int]:
    x, y, w, h = bbox_xywh
    x2, y2 = x + w, y + h
    m = (uv[:, 0] >= x) & (uv[:, 0] <= x2) & (uv[:, 1] >= y) & (uv[:, 1] <= y2)
    if not np.any(m):
        return None, 0

    vv = val[m]
    n = int(vv.shape[0])

    if mode == "mean":
        out = float(np.mean(vv))
    elif mode == "min":
        out = float(np.min(vv))
    elif mode == "p20":
        out = float(np.percentile(vv, 20))
    else:  # "median"
        out = float(np.median(vv))

    return out, n


def draw_points_by_depth(vis: np.ndarray, uv: np.ndarray, z: np.ndarray) -> None:
    if len(uv) == 0:
        return
    h, w = vis.shape[:2]
    z_vis = z.copy()

    z_min = float(np.percentile(z_vis, 2))
    z_max = float(np.percentile(z_vis, 98))
    if z_max <= z_min:
        z_min, z_max = float(z_vis.min()), float(z_vis.max()) + 1e-6

    invert = True
    zn = (z_vis - z_min) / (z_max - z_min + 1e-12)
    zn = np.clip(zn, 0.0, 1.0)
    if invert:
        zn = 1.0 - zn
    z8 = (zn * 255).astype(np.uint8)

    colors = (
        cv2.applyColorMap(z8.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
    )  # BGR

    for i in range(0, len(uv), POINT_STRIDE):
        u, v = int(round(uv[i, 0])), int(round(uv[i, 1]))
        if 0 <= u < w and 0 <= v < h:
            c = colors[i]
            cv2.circle(vis, (u, v), 2, (int(c[0]), int(c[1]), int(c[2])), -1)


# main
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSONL.write_text("", encoding="utf-8")

    K, dist, extr = DEFAULT_K, DEFAULT_DIST, DEFAULT_EXTRINSIC

    yolo_files = sorted(YOLO_JSON_DIR.glob("*.json"))
    if not yolo_files:
        raise FileNotFoundError(f"YOLO json이 없습니다: {YOLO_JSON_DIR}")

    processed = 0
    for yp in yolo_files:
        stem = yp.stem
        img_path = find_image_file(stem)
        pcd_path = LIDAR_DIR / f"{stem}.pcd"

        if img_path is None:
            print(f"[MISS] image: {stem}")
            continue
        if not pcd_path.exists():
            print(f"[MISS] pcd: {stem}.pcd")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[BAD] image read fail: {img_path.name}")
            continue

        yolo_obj = load_json(yp)
        anns = extract_bboxes_from_yolo(yolo_obj)
        if not anns:
            print(f"[WARN] no yolo bboxes: {yp.name} -> save empty objects")
            anns = []

        pts = load_pcd_xyz(pcd_path)
        uv, z, d = project_points(pts, K, dist, extr)

        vis = img.copy()

        if DRAW_PROJECTED_POINTS and len(uv) > 0:
            draw_points_by_depth(vis, uv, z)

        objects_out: List[Dict[str, Any]] = []
        for ann in anns:
            bbox_xywh = ann["bbox_xywh"] 

            dist_m, n_pts = depth_stat_in_bbox(uv, d, bbox_xywh, mode=DIST_MODE)
            if n_pts < MIN_PTS_IN_BBOX:
                dist_m = None

            x, y, bw, bh = bbox_xywh
            x1, y1, x2, y2 = int(x), int(y), int(x + bw), int(y + bh)

            # bbox 시각화
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cat_name = ann["category_name"]
            conf = ann.get("confidence", None)

            if dist_m is not None:
                if conf is not None:
                    txt = f"{cat_name}({conf:.2f}):{dist_m:.2f}m"
                else:
                    txt = f"{cat_name}:{dist_m:.2f}m"
            else:
                if conf is not None:
                    txt = f"{cat_name}({conf:.2f}):None"
                else:
                    txt = f"{cat_name}:None"

            cv2.putText(
                vis,
                txt,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

            objects_out.append(
                {
                    "ann_idx": ann["ann_idx"],
                    "category_name": cat_name,
                    "confidence": float(conf) if conf is not None else None,
                    "bbox_xywh": bbox_xywh,               # 좌상단 xywh (거리계산에 사용)
                    "bbox_cxcywh": ann["bbox_cxcywh"],    # 원본 YOLO cxcywh
                    "distance_m": dist_m,
                    "distance_mode": DIST_MODE,
                    "num_points_in_bbox": n_pts,
                }
            )

        out_img = OUT_IMG_DIR / f"{stem}.png"
        cv2.imwrite(str(out_img), vis)

        frame_out = {
            "stem": stem,
            "image_file": img_path.name,
            "yolo_file": yp.name,
            "pcd_file": pcd_path.name,
            "num_objects": len(objects_out),
            "distance_mode": DIST_MODE,
            "objects": objects_out,
        }

        out_json = OUT_JSON_DIR / f"{stem}.json"
        write_json(out_json, frame_out)

        with OUT_JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(frame_out, ensure_ascii=False) + "\n")

        print(f"[OK] {stem} -> img:{out_img.name} json:{out_json.name}")

        processed += 1
        if DEBUG_FIRST_N > 0 and processed >= DEBUG_FIRST_N:
            break

    print(
        f"[DONE] processed={processed} images->{OUT_IMG_DIR} json->{OUT_JSON_DIR} jsonl->{OUT_JSONL}"
    )


if __name__ == "__main__":
    main()
