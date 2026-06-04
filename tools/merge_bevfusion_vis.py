"""
BEVFusion Visualization Merger

레이아웃:
  ┌─────────────────────────────────────────────┐
  │ CAM_FRONT_LEFT │ CAM_FRONT │ CAM_FRONT_RIGHT │  ← 전방 행
  ├─────────────────────────────────────────────┤
  │                 LiDAR BEV                   │  ← LiDAR 중앙
  ├─────────────────────────────────────────────┤
  │ CAM_BACK_LEFT  │ CAM_BACK  │ CAM_BACK_RIGHT  │  ← 후방 행
  └─────────────────────────────────────────────┘

- 존재하는 폴더만 자동 감지
- 없는 카메라 슬롯은 빈 타일(검정)로 채워 위치 유지
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── 설정 ──────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path("runs/run-2c7a30f9-7ef5f73e_3cam_pre_20epo/test")
OUTPUT_DIR = Path("runs/run-2c7a30f9-7ef5f73e_3cam_pre_20epo/test/merged")

# 카메라 폴더명 → 레이블 매핑
CAMERA_LABEL_MAP = {
    "camera-0": "CAM_FRONT",
    "camera-1": "CAM_FRONT_RIGHT",
    "camera-2": "CAM_FRONT_LEFT",
    "camera-3": "CAM_BACK",
    "camera-4": "CAM_BACK_LEFT",
    "camera-5": "CAM_BACK_RIGHT",
}

LIDAR_DIR_NAME = "lidar"

# 레이아웃 행 정의 (좌 → 우 순서)
FRONT_ROW = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
BACK_ROW  = ["CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT"]

# 스타일
CANVAS_BG    = (20,  20,  20)
LABEL_BG     = (0,   0,   0)
LABEL_FG     = (255, 255, 255)
BORDER_COLOR = (60,  60,  60)
EMPTY_BG     = (35,  35,  35)
EMPTY_FG     = (80,  80,  80)
BORDER_PX    = 2
LABEL_H      = 30
GAP          = 6


# ── 이미지 유틸 ───────────────────────────────────────────────────────────────

def resize_to_size(img, w, h):
    if img.shape[1] == w and img.shape[0] == h:
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def resize_to_width(img, target_w):
    h, w = img.shape[:2]
    if w == target_w:
        return img
    scale = target_w / w
    return cv2.resize(img, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)


def resize_to_height(img, target_h):
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / h
    return cv2.resize(img, (int(w * scale), target_h), interpolation=cv2.INTER_AREA)


def pad_to_width(img, target_w):
    h, w = img.shape[:2]
    if w >= target_w:
        return img
    pad = np.full((h, target_w - w, 3), CANVAS_BG, dtype=np.uint8)
    return np.hstack([img, pad])


def add_label(img, text, bg=LABEL_BG, fg=LABEL_FG):
    h, w = img.shape[:2]
    bar = np.full((LABEL_H, w, 3), bg, dtype=np.uint8)
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    tx = (w - tw) // 2
    ty = (LABEL_H + th) // 2 - 1
    cv2.putText(bar, text, (tx, ty), font, scale, fg, thick, cv2.LINE_AA)
    return np.vstack([bar, img])


def add_border(img):
    return cv2.copyMakeBorder(
        img, BORDER_PX, BORDER_PX, BORDER_PX, BORDER_PX,
        cv2.BORDER_CONSTANT, value=BORDER_COLOR,
    )


def make_empty_tile(w, h, label):
    """카메라 없는 슬롯을 빈 타일로 채웁니다."""
    tile = np.full((h, w, 3), EMPTY_BG, dtype=np.uint8)
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    text = "[{}]".format(label)
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    tx = (w - tw) // 2
    ty = (h + th) // 2
    cv2.putText(tile, text, (tx, ty), font, scale, EMPTY_FG, thick, cv2.LINE_AA)
    return tile


# ── 행 생성 ───────────────────────────────────────────────────────────────────

def make_camera_row(label_order, cam_map, tile_w, tile_h):
    """
    label_order 순서대로 카메라 타일을 가로 배치.
    없는 카메라는 빈 타일로 대체.
    행 전체에 유효 이미지가 없으면 None 반환.
    """
    if not any(lbl in cam_map for lbl in label_order):
        return None

    tiles = []
    for lbl in label_order:
        if lbl in cam_map:
            img  = resize_to_size(cam_map[lbl], tile_w, tile_h)
            tile = add_label(img, lbl)
        else:
            tile = add_label(make_empty_tile(tile_w, tile_h), lbl,
                             bg=EMPTY_BG, fg=EMPTY_FG)
        tile = add_border(tile)
        tiles.append(tile)

    max_h = max(t.shape[0] for t in tiles)
    tiles = [resize_to_height(t, max_h) for t in tiles]

    spacer = np.full((max_h, GAP, 3), CANVAS_BG, dtype=np.uint8)
    parts = []
    for i, t in enumerate(tiles):
        parts.append(t)
        if i < len(tiles) - 1:
            parts.append(spacer)
    return np.hstack(parts)


def make_lidar_row(lidar_img, target_w):
    img  = resize_to_width(lidar_img, target_w)
    tile = add_label(img, "LiDAR BEV")
    return add_border(tile)


# ── 캔버스 조합 ───────────────────────────────────────────────────────────────

def build_canvas(cam_map, lidar_img):
    """
    전방 카메라 행 → LiDAR → 후방 카메라 행
    각 행은 존재할 때만 포함.
    """
    # 기준 너비 결정
    if lidar_img is not None:
        lidar_w, lidar_h = lidar_img.shape[1], lidar_img.shape[0]
    else:
        lidar_w, lidar_h = 1920, 640

    n_cols = 3
    tile_w = max(240, (lidar_w - GAP * (n_cols - 1)) // n_cols)

    # 카메라 타일 높이: 샘플 이미지 비율 사용
    sample = next(iter(cam_map.values()), None)
    if sample is not None:
        oh, ow = sample.shape[:2]
        tile_h = int(tile_w * oh / ow)
    else:
        tile_h = int(tile_w * 9 / 16)

    front_row = make_camera_row(FRONT_ROW, cam_map, tile_w, tile_h)
    back_row  = make_camera_row(BACK_ROW,  cam_map, tile_w, tile_h)

    # lidar 너비를 카메라 행 너비에 맞춤
    ref_row = front_row if front_row is not None else back_row
    row_w   = ref_row.shape[1] if ref_row is not None else lidar_w

    lidar_row = make_lidar_row(lidar_img, row_w) if lidar_img is not None else None

    rows = [r for r in [front_row, lidar_row, back_row] if r is not None]
    if not rows:
        return None

    max_w = max(r.shape[1] for r in rows)
    rows  = [pad_to_width(r, max_w) for r in rows]

    spacer = np.full((GAP, max_w, 3), CANVAS_BG, dtype=np.uint8)
    parts = []
    for i, row in enumerate(rows):
        parts.append(row)
        if i < len(rows) - 1:
            parts.append(spacer)

    return np.vstack(parts)


# ── 파일 수집 ─────────────────────────────────────────────────────────────────

def get_sorted_image_files(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(f for f in folder.iterdir()
                  if f.is_file() and f.suffix.lower() in exts)


def collect_frame_map(root):
    frame_map = {}
    dirs_to_scan = []
    for dir_name in CAMERA_LABEL_MAP:
        d = root / dir_name
        if d.is_dir():
            dirs_to_scan.append((dir_name, d))
    lidar_dir = root / LIDAR_DIR_NAME
    if lidar_dir.is_dir():
        dirs_to_scan.append((LIDAR_DIR_NAME, lidar_dir))

    for dir_name, folder in dirs_to_scan:
        for img_path in get_sorted_image_files(folder):
            stem = img_path.stem
            if stem not in frame_map:
                frame_map[stem] = {}
            frame_map[stem][dir_name] = img_path
    return frame_map


# ── 메인 ──────────────────────────────────────────────────────────────────────

def merge_frames(root=ROOT_DIR, output=OUTPUT_DIR):
    output.mkdir(parents=True, exist_ok=True)

    available_dirs = [n for n in CAMERA_LABEL_MAP if (root / n).is_dir()]
    has_lidar      = (root / LIDAR_DIR_NAME).is_dir()

    if not available_dirs and not has_lidar:
        print("[ERROR] '{}' 아래에 카메라/lidar 폴더가 없습니다.".format(root))
        return

    available_labels = sorted(CAMERA_LABEL_MAP[d] for d in available_dirs)
    print("[INFO] 감지된 카메라 : {}".format(available_labels))
    print("[INFO] LiDAR 존재   : {}".format(has_lidar))
    print("[INFO] 레이아웃     :")
    print("         [전방] {}".format(" | ".join(FRONT_ROW)))
    print("         [중앙] LiDAR BEV")
    print("         [후방] {}".format(" | ".join(BACK_ROW)))

    frame_map = collect_frame_map(root)
    if not frame_map:
        print("[ERROR] 이미지 파일을 찾을 수 없습니다.")
        return

    total = len(frame_map)
    print("\n[INFO] 총 {}개 프레임 처리 시작...\n".format(total))

    for idx, (stem, sources) in enumerate(sorted(frame_map.items())):
        # 카메라 로드
        cam_map = {}
        for dir_name, label in CAMERA_LABEL_MAP.items():
            if dir_name in sources:
                img = cv2.imread(str(sources[dir_name]))
                if img is not None:
                    cam_map[label] = img
                else:
                    print("  [WARN] 읽기 실패: {}".format(sources[dir_name]))

        # LiDAR 로드
        lidar_img = None
        if has_lidar and LIDAR_DIR_NAME in sources:
            lidar_img = cv2.imread(str(sources[LIDAR_DIR_NAME]))
            if lidar_img is None:
                print("  [WARN] LiDAR 읽기 실패: {}".format(sources[LIDAR_DIR_NAME]))

        if not cam_map and lidar_img is None:
            print("  [SKIP] {}: 유효한 이미지 없음".format(stem))
            continue

        canvas = build_canvas(cam_map, lidar_img)
        if canvas is None:
            continue

        out_path = output / "{}.jpg".format(stem)
        cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])

        if (idx + 1) % 10 == 0 or idx == 0:
            print("  [{:>4}/{}] {}  ({}x{})".format(
                idx + 1, total, out_path.name,
                canvas.shape[1], canvas.shape[0]))

    print("\n[DONE] 저장 위치: {}".format(output.resolve()))


def preview_single(frame_stem, root=ROOT_DIR):
    frame_map = collect_frame_map(root)
    if frame_stem not in frame_map:
        print("[ERROR] '{}' 프레임 없음".format(frame_stem))
        return

    sources = frame_map[frame_stem]
    cam_map = {}
    for dir_name, label in CAMERA_LABEL_MAP.items():
        if dir_name in sources:
            img = cv2.imread(str(sources[dir_name]))
            if img is not None:
                cam_map[label] = img

    lidar_img = None
    lidar_dir = root / LIDAR_DIR_NAME
    if lidar_dir.is_dir() and LIDAR_DIR_NAME in sources:
        lidar_img = cv2.imread(str(sources[LIDAR_DIR_NAME]))

    canvas = build_canvas(cam_map, lidar_img)
    if canvas is not None:
        cv2.imshow("BEVFusion - {}".format(frame_stem), canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    merge_frames()