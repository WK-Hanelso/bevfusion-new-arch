import argparse
import copy
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from mmcv import Config
from nuscenes.utils.data_classes import Box as NuScenesBox
from pyquaternion import Quaternion
from torchpack.utils.config import configs

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.datasets import build_dataset


GT_COLOR_BGR = (0, 0, 255)
PRED_COLOR_BGR = (255, 0, 0)
FOV_COLOR_BGR = (0, 255, 255)
FRONT_AXIS_COLOR_BGR = (0, 0, 255)
POINT_FUSION_COLOR = "#66ffe6"
POINT_LIDAR_ONLY_COLOR = "#777777"
CLASS_COLOR_RGB = {
    "car": (31, 119, 180),
    "truck": (255, 127, 14),
    "construction_vehicle": (148, 103, 189),
    "bus": (44, 160, 44),
    "trailer": (214, 39, 40),
    "barrier": (140, 86, 75),
    "motorcycle": (255, 187, 120),
    "bicycle": (23, 190, 207),
    "pedestrian": (227, 119, 194),
    "traffic_cone": (188, 189, 34),
}
FALLBACK_CLASS_COLOR_RGB = (51, 163, 255)


def class_name_from_label(label: int, classes: Sequence[str]) -> str:
    return classes[label] if 0 <= label < len(classes) else str(label)


def class_color_rgb(label: int, classes: Sequence[str]) -> Tuple[int, int, int]:
    return CLASS_COLOR_RGB.get(
        class_name_from_label(label, classes),
        FALLBACK_CLASS_COLOR_RGB,
    )


def class_color_bgr(label: int, classes: Sequence[str]) -> Tuple[int, int, int]:
    r, g, b = class_color_rgb(label, classes)
    return (b, g, r)


def class_color_mpl(label: int, classes: Sequence[str]) -> Tuple[float, float, float]:
    r, g, b = class_color_rgb(label, classes)
    return (r / 255.0, g / 255.0, b / 255.0)


def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize custom nuScenes FOV split on BEV and camera images."
    )
    parser.add_argument("config", help="BEVFusion config file.")
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to visualize.",
    )
    parser.add_argument(
        "--result-json",
        default=None,
        help="Optional nuScenes results_nusc.json. If omitted, only GT is drawn.",
    )
    parser.add_argument(
        "--prediction-pkl",
        default=None,
        help=(
            "Optional raw prediction pkl saved by tools/test.py --out. "
            "If provided, it is preferred over --result-json for drawing predictions."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="custom_eval_viz",
        help="Directory to save visualizations.",
    )
    parser.add_argument(
        "--samples-per-scene",
        type=int,
        default=1,
        help="Number of evenly spaced samples to draw per scene.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=-1,
        help="Limit number of scenes. -1 means all scenes.",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.1,
        help="Prediction score threshold used only for visualization.",
    )
    parser.add_argument(
        "--camera-names",
        nargs="+",
        default=[
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT",
            "CAM_BACK",
            "CAM_BACK_RIGHT",
        ],
        help="Camera names to render. Missing names are skipped.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=120000,
        help="Subsample lidar points for faster BEV rendering.",
    )
    parser.add_argument(
        "--fov-degree",
        nargs=2,
        type=float,
        default=None,
        help="Override custom_eval.fov_degree. Defaults to config value.",
    )
    parser.add_argument(
        "--lidar-sensor",
        default=None,
        help="Override custom_eval.lidar_sensor. Kept for metadata consistency.",
    )
    parser.add_argument(
        "--fov-front-axis",
        default=None,
        choices=["x", "y"],
        help=(
            "LiDAR axis used as 0 deg/front for FOV drawing. "
            "Defaults to custom_eval.fov_front_axis or y for nuScenes LIDAR_TOP."
        ),
    )
    parser.add_argument(
        "--match-distance-thr",
        type=float,
        default=None,
        help=(
            "GT-prediction matching center-distance threshold for instances.json. "
            "Defaults to evaluation.custom_eval.match_distance_threshold or 2.0m."
        ),
    )
    parser.add_argument(
        "--bev-dpi",
        type=int,
        default=120,
        help="DPI for BEV image output.",
    )
    parser.add_argument(
        "--nuscenes-root",
        default=None,
        help=(
            "nuScenes dataroot used to recover scene names when the info pkl "
            "does not contain scene metadata. Defaults to dataset.dataset_root."
        ),
    )
    parser.add_argument(
        "--nuscenes-version",
        default=None,
        help=(
            "nuScenes version used with --nuscenes-root. Defaults to "
            "dataset.version or v1.0-trainval."
        ),
    )
    args, opts = parser.parse_known_args()
    return args, opts


def load_config(config_path: str, opts: Sequence[str]) -> Config:
    configs.load(config_path, recursive=True)
    configs.update(opts)
    return Config(recursive_eval(configs), filename=config_path)


def load_predictions(result_json: Optional[str]) -> Dict[str, list]:
    if result_json is None:
        return {}
    data = mmcv.load(result_json)
    return data.get("results", data)


def load_prediction_pkl(prediction_pkl: Optional[str]) -> Optional[list]:
    if prediction_pkl is None:
        return None
    return mmcv.load(prediction_pkl)


def get_custom_eval_cfg(cfg: Config, args) -> Dict[str, object]:
    custom_eval = dict(
        enabled=True,
        fov_degree=[-85.0, 85.0],
        lidar_sensor="LIDAR_TOP",
        fov_front_axis="y",
        match_distance_threshold=2.0,
    )
    if cfg.get("evaluation", None) is not None:
        custom_eval.update(dict(cfg.evaluation.get("custom_eval", {})))
    if args.fov_degree is not None:
        custom_eval["fov_degree"] = [float(args.fov_degree[0]), float(args.fov_degree[1])]
    if args.lidar_sensor is not None:
        custom_eval["lidar_sensor"] = args.lidar_sensor
    if args.fov_front_axis is not None:
        custom_eval["fov_front_axis"] = args.fov_front_axis
    if args.match_distance_thr is not None:
        custom_eval["match_distance_threshold"] = args.match_distance_thr
    custom_eval["fov_degree"] = [
        float(custom_eval["fov_degree"][0]),
        float(custom_eval["fov_degree"][1]),
    ]
    custom_eval["fov_front_axis"] = str(custom_eval["fov_front_axis"]).lower()
    if custom_eval["fov_front_axis"] not in ("x", "y"):
        raise ValueError("custom_eval.fov_front_axis must be 'x' or 'y'")
    custom_eval["match_distance_threshold"] = float(
        custom_eval["match_distance_threshold"]
    )
    return custom_eval


def scene_key(info: Dict[str, object], scene_lookup: Optional[Dict[str, str]] = None) -> str:
    for key in ("scene_token", "scene_name", "scene"):
        if key in info and info[key] is not None:
            return str(info[key])
    token = info.get("token")
    if scene_lookup is not None and token in scene_lookup:
        return scene_lookup[token]
    # Fallback for custom infos that do not preserve scene metadata.
    return "scene_unknown"


def build_scene_lookup(data_infos: List[dict], dataset, args) -> Dict[str, str]:
    needs_lookup = any(scene_key(info) == "scene_unknown" for info in data_infos)
    if not needs_lookup:
        return {}

    dataroot = args.nuscenes_root or getattr(dataset, "dataset_root", None)
    version = args.nuscenes_version or getattr(dataset, "version", "v1.0-trainval")
    if dataroot is None:
        print(
            "Scene metadata is missing from the info pkl, and nuScenes dataroot "
            "could not be inferred. Samples will be grouped as scene_unknown."
        )
        return {}

    try:
        from nuscenes import NuScenes

        nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    except Exception as exc:
        print(
            "Failed to load nuScenes DB for scene lookup "
            "(dataroot={}, version={}): {}. Samples will be grouped as "
            "scene_unknown.".format(dataroot, version, exc)
        )
        return {}

    lookup = {}
    for info in data_infos:
        token = info.get("token")
        if token is None:
            continue
        try:
            sample = nusc.get("sample", token)
            scene = nusc.get("scene", sample["scene_token"])
            lookup[token] = scene.get("name", sample["scene_token"])
        except Exception:
            continue
    return lookup


def select_scene_samples(
    data_infos: List[dict],
    samples_per_scene: int,
    max_scenes: int,
    scene_lookup: Optional[Dict[str, str]] = None,
):
    grouped = defaultdict(list)
    for index, info in enumerate(data_infos):
        grouped[scene_key(info, scene_lookup)].append(index)

    selected = []
    for scene_index, (scene, indices) in enumerate(sorted(grouped.items())):
        if max_scenes >= 0 and scene_index >= max_scenes:
            break
        indices = sorted(indices, key=lambda idx: data_infos[idx].get("timestamp", idx))
        if len(indices) <= samples_per_scene:
            picked = indices
        else:
            positions = np.linspace(0, len(indices) - 1, samples_per_scene)
            picked = [indices[int(round(pos))] for pos in positions]
        for order, index in enumerate(picked):
            selected.append((scene, order, index))
    return selected


def load_lidar_points(info: dict) -> np.ndarray:
    path = info["lidar_path"]
    points = np.fromfile(path, dtype=np.float32)
    points = points.reshape(-1, 5)
    return points


def build_lidar2image(camera_info: dict) -> np.ndarray:
    lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
    lidar2camera_t = camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
    lidar2camera_rt = np.eye(4, dtype=np.float32)
    lidar2camera_rt[:3, :3] = lidar2camera_r.T
    lidar2camera_rt[3, :3] = -lidar2camera_t

    camera_intrinsics = np.eye(4, dtype=np.float32)
    camera_intrinsics[:3, :3] = camera_info["cam_intrinsic"]
    return camera_intrinsics @ lidar2camera_rt.T


def get_lidar2image_by_camera(info: dict) -> Dict[str, np.ndarray]:
    return {
        camera_name: build_lidar2image(camera_info)
        for camera_name, camera_info in info.get("cams", {}).items()
    }


def angle_in_fov(angle_degree: np.ndarray, fov_degree: Sequence[float]) -> np.ndarray:
    fov_min, fov_max = fov_degree
    if fov_min <= fov_max:
        return (angle_degree >= fov_min) & (angle_degree <= fov_max)
    return (angle_degree >= fov_min) | (angle_degree <= fov_max)


def lidar_fov_angle_degree(lidar_xyz: np.ndarray, front_axis: str) -> np.ndarray:
    """Return azimuth where 0 deg is the configured LiDAR front axis."""
    if front_axis == "x":
        return np.degrees(np.arctan2(lidar_xyz[..., 1], lidar_xyz[..., 0]))
    return np.degrees(np.arctan2(-lidar_xyz[..., 0], lidar_xyz[..., 1]))


def filter_points_by_range(points: np.ndarray, point_cloud_range: Sequence[float]):
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    mask = (
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
        & (points[:, 2] >= z_min)
        & (points[:, 2] <= z_max)
    )
    return points[mask]


def make_gt_boxes(info: dict, classes: Sequence[str]) -> Tuple[LiDARInstance3DBoxes, np.ndarray]:
    if "valid_flag" in info:
        mask = info["valid_flag"]
    else:
        mask = info["num_lidar_pts"] > 0
    boxes = info["gt_boxes"][mask]
    names = info["gt_names"][mask]

    labels = []
    keep = []
    for index, name in enumerate(names):
        if name in classes:
            labels.append(classes.index(name))
            keep.append(index)
    if keep:
        boxes = boxes[np.asarray(keep)]
    else:
        boxes = np.zeros((0, 9), dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    return LiDARInstance3DBoxes(
        boxes,
        box_dim=boxes.shape[-1] if boxes.size else 9,
        origin=(0.5, 0.5, 0.5),
    ), labels


def global_box_to_lidar_tensor(pred: dict, info: dict) -> np.ndarray:
    box = NuScenesBox(
        center=pred["translation"],
        size=pred["size"],
        orientation=Quaternion(pred["rotation"]),
        score=pred.get("detection_score", 0.0),
    )

    box.translate(-np.asarray(info["ego2global_translation"]))
    box.rotate(Quaternion(info["ego2global_rotation"]).inverse)
    box.translate(-np.asarray(info["lidar2ego_translation"]))
    box.rotate(Quaternion(info["lidar2ego_rotation"]).inverse)

    yaw = -box.orientation.yaw_pitch_roll[0] - np.pi / 2
    bottom_center = np.asarray(box.center, dtype=np.float32)
    bottom_center[2] -= float(box.wlh[2]) / 2.0
    return np.asarray(
        [
            bottom_center[0],
            bottom_center[1],
            bottom_center[2],
            box.wlh[0],
            box.wlh[1],
            box.wlh[2],
            yaw,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )


def make_pred_boxes(
    info: dict,
    predictions: Dict[str, list],
    classes: Sequence[str],
    score_thr: float,
) -> Tuple[Optional[LiDARInstance3DBoxes], Optional[np.ndarray], Optional[np.ndarray]]:
    sample_token = info["token"]
    annos = predictions.get(sample_token, [])
    boxes = []
    labels = []
    scores = []
    for pred in annos:
        score = float(pred.get("detection_score", 0.0))
        name = pred.get("detection_name", "")
        if score < score_thr or name not in classes:
            continue
        boxes.append(global_box_to_lidar_tensor(pred, info))
        labels.append(classes.index(name))
        scores.append(score)

    if not boxes:
        return None, None, None
    boxes = np.stack(boxes, axis=0)
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    return LiDARInstance3DBoxes(boxes, box_dim=9), labels, scores


def to_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def unwrap_detection_output(detection):
    if isinstance(detection, dict) and "pts_bbox" in detection:
        return detection["pts_bbox"]
    return detection


def make_pred_boxes_from_pkl(
    detection,
    score_thr: float,
) -> Tuple[Optional[LiDARInstance3DBoxes], Optional[np.ndarray], Optional[np.ndarray]]:
    detection = unwrap_detection_output(detection)
    if not isinstance(detection, dict) or "boxes_3d" not in detection:
        return None, None, None

    boxes_3d = detection["boxes_3d"]
    scores = to_numpy(detection["scores_3d"])
    labels = to_numpy(detection["labels_3d"]).astype(np.int64)
    keep = scores >= score_thr
    if not np.any(keep):
        return None, None, None

    boxes = to_numpy(boxes_3d.tensor)[keep].copy()
    boxes[:, 2] -= boxes[:, 5] / 2.0
    labels = labels[keep]
    scores = scores[keep].astype(np.float32)
    return LiDARInstance3DBoxes(boxes, box_dim=boxes.shape[-1]), labels, scores


def project_lidar_points(points: np.ndarray, lidar2image: np.ndarray) -> np.ndarray:
    homo = np.concatenate([points[:, :3], np.ones((len(points), 1))], axis=1)
    coords = homo @ lidar2image.T
    valid = coords[:, 2] > 1e-5
    coords = coords[valid]
    coords[:, 0] /= coords[:, 2]
    coords[:, 1] /= coords[:, 2]
    return coords[:, :2]


def fov_rays(
    fov_degree: Sequence[float],
    radius: float,
    z: float = 0.0,
    front_axis: str = "y",
) -> List[np.ndarray]:
    rays = []
    radii = np.linspace(0.0, radius, 120)
    for angle in fov_degree:
        theta = np.radians(angle)
        if front_axis == "x":
            xs = radii * np.cos(theta)
            ys = radii * np.sin(theta)
        else:
            xs = -radii * np.sin(theta)
            ys = radii * np.cos(theta)
        zs = np.full_like(xs, z)
        rays.append(np.stack([xs, ys, zs], axis=1))
    return rays


def front_axis_ray(radius: float, z: float = 0.0, front_axis: str = "y") -> np.ndarray:
    radii = np.linspace(0.0, radius, 120)
    if front_axis == "x":
        xs = radii
        ys = np.zeros_like(radii)
    else:
        xs = np.zeros_like(radii)
        ys = radii
    zs = np.full_like(xs, z)
    return np.stack([xs, ys, zs], axis=1)


def lidar_xy_to_front_up_bev(xy: np.ndarray, front_axis: str) -> np.ndarray:
    """Map LiDAR XY to a BEV image where the configured front axis is up."""
    bev_xy = np.empty_like(xy)
    if front_axis == "x":
        bev_xy[..., 0] = -xy[..., 1]
        bev_xy[..., 1] = xy[..., 0]
    else:
        bev_xy[..., 0] = xy[..., 0]
        bev_xy[..., 1] = xy[..., 1]
    return bev_xy


def draw_lidar_boxes_bev(
    ax,
    boxes: Optional[LiDARInstance3DBoxes],
    color: str,
    linewidth: float,
    center_size: float,
    front_axis: str,
    labels: Optional[np.ndarray] = None,
    classes: Optional[Sequence[str]] = None,
):
    if boxes is None or len(boxes) == 0:
        return
    corners = lidar_xy_to_front_up_bev(
        boxes.corners.numpy()[:, [0, 3, 7, 4, 0], :2],
        front_axis,
    )
    centers = lidar_xy_to_front_up_bev(boxes.gravity_center.numpy()[:, :2], front_axis)
    for index in range(corners.shape[0]):
        box_color = color
        if labels is not None and classes is not None:
            box_color = class_color_mpl(int(labels[index]), classes)
        ax.plot(
            corners[index, :, 0],
            corners[index, :, 1],
            color=box_color,
            linewidth=linewidth,
        )
        ax.scatter(
            centers[index, 0],
            centers[index, 1],
            s=center_size,
            c=[box_color],
        )


def visualize_bev(
    fpath: str,
    points: np.ndarray,
    gt_boxes: Optional[LiDARInstance3DBoxes],
    pred_boxes: Optional[LiDARInstance3DBoxes],
    pred_labels: Optional[np.ndarray],
    classes: Sequence[str],
    fov_degree: Sequence[float],
    front_axis: str,
    point_cloud_range: Sequence[float],
    max_points: int,
    dpi: int,
):
    points = filter_points_by_range(points, point_cloud_range)
    if max_points > 0 and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
        points = points[indices]

    angles = lidar_fov_angle_degree(points[:, :3], front_axis)
    fusion_mask = angle_in_fov(angles, fov_degree)

    bev_points = lidar_xy_to_front_up_bev(points[:, :2], front_axis)
    if front_axis == "x":
        xlim = [-point_cloud_range[4], -point_cloud_range[1]]
        ylim = [point_cloud_range[0], point_cloud_range[3]]
    else:
        xlim = [point_cloud_range[0], point_cloud_range[3]]
        ylim = [point_cloud_range[1], point_cloud_range[4]]
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_axis_off()

    ax.scatter(
        bev_points[~fusion_mask, 0],
        bev_points[~fusion_mask, 1],
        s=0.15,
        c=POINT_LIDAR_ONLY_COLOR,
        alpha=0.55,
        linewidths=0,
    )
    ax.scatter(
        bev_points[fusion_mask, 0],
        bev_points[fusion_mask, 1],
        s=0.2,
        c=POINT_FUSION_COLOR,
        alpha=0.85,
        linewidths=0,
    )

    radius = max(abs(xlim[0]), abs(xlim[1]), abs(ylim[0]), abs(ylim[1]))
    for ray in fov_rays(fov_degree, radius, front_axis=front_axis):
        bev_ray = lidar_xy_to_front_up_bev(ray[:, :2], front_axis)
        ax.plot(bev_ray[:, 0], bev_ray[:, 1], color="yellow", linewidth=2.0)
    front_ray = lidar_xy_to_front_up_bev(
        front_axis_ray(radius, front_axis=front_axis)[:, :2],
        front_axis,
    )
    ax.plot(front_ray[:, 0], front_ray[:, 1], color="red", linewidth=2.4)

    draw_lidar_boxes_bev(
        ax,
        gt_boxes,
        "red",
        linewidth=1.8,
        center_size=10,
        front_axis=front_axis,
    )
    draw_lidar_boxes_bev(
        ax,
        pred_boxes,
        "#33a3ff",
        linewidth=1.2,
        center_size=7,
        front_axis=front_axis,
        labels=pred_labels,
        classes=classes,
    )

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    fig.savefig(fpath, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


BOX_EDGES = [
    (0, 1),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 5),
    (3, 2),
    (3, 7),
    (4, 5),
    (4, 7),
    (2, 6),
    (5, 6),
    (6, 7),
]


def draw_boxes_image(
    canvas: np.ndarray,
    boxes: Optional[LiDARInstance3DBoxes],
    lidar2image: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int,
    labels: Optional[np.ndarray] = None,
    classes: Optional[Sequence[str]] = None,
):
    if boxes is None or len(boxes) == 0:
        return
    corners = boxes.corners.numpy()
    num_boxes = corners.shape[0]
    coords = np.concatenate(
        [corners.reshape(-1, 3), np.ones((num_boxes * 8, 1))], axis=1
    )
    coords = coords @ lidar2image.T
    coords = coords.reshape(num_boxes, 8, 4)
    visible = np.all(coords[..., 2] > 1e-5, axis=1)
    coords = coords[visible]
    visible_labels = None
    if labels is not None:
        visible_labels = np.asarray(labels, dtype=np.int64)[visible]
    if len(coords) == 0:
        return
    coords[..., 0] /= coords[..., 2]
    coords[..., 1] /= coords[..., 2]
    coords = coords[..., :2]
    height, width = canvas.shape[:2]
    for index, box_coords in enumerate(coords):
        if (
            np.all(box_coords[:, 0] < 0)
            or np.all(box_coords[:, 0] >= width)
            or np.all(box_coords[:, 1] < 0)
            or np.all(box_coords[:, 1] >= height)
        ):
            continue
        box_color = color
        if visible_labels is not None and classes is not None:
            box_color = class_color_bgr(int(visible_labels[index]), classes)
        for start, end in BOX_EDGES:
            cv2.line(
                canvas,
                tuple(np.round(box_coords[start]).astype(np.int32)),
                tuple(np.round(box_coords[end]).astype(np.int32)),
                box_color,
                thickness,
                cv2.LINE_AA,
            )


def draw_centers_image(
    canvas: np.ndarray,
    boxes: Optional[LiDARInstance3DBoxes],
    lidar2image: np.ndarray,
    color: Tuple[int, int, int],
    labels: Optional[np.ndarray] = None,
    classes: Optional[Sequence[str]] = None,
):
    if boxes is None or len(boxes) == 0:
        return
    centers = boxes.gravity_center.numpy()
    homo = np.concatenate([centers[:, :3], np.ones((len(centers), 1))], axis=1)
    coords = homo @ lidar2image.T
    valid = coords[:, 2] > 1e-5
    coords = coords[valid]
    visible_labels = None
    if labels is not None:
        visible_labels = np.asarray(labels, dtype=np.int64)[valid]
    coords[:, 0] /= coords[:, 2]
    coords[:, 1] /= coords[:, 2]
    coords = coords[:, :2]
    height, width = canvas.shape[:2]
    for index, point in enumerate(coords):
        x, y = np.round(point).astype(np.int32)
        if 0 <= x < width and 0 <= y < height:
            point_color = color
            if visible_labels is not None and classes is not None:
                point_color = class_color_bgr(int(visible_labels[index]), classes)
            cv2.circle(canvas, (x, y), 4, point_color, -1, cv2.LINE_AA)


def draw_fov_image(
    canvas: np.ndarray,
    lidar2image: np.ndarray,
    fov_degree: Sequence[float],
    radius: float,
    front_axis: str,
):
    for ray in fov_rays(fov_degree, radius, z=0.0, front_axis=front_axis):
        coords = project_lidar_points(ray, lidar2image)
        if len(coords) < 2:
            continue
        height, width = canvas.shape[:2]
        coords = np.round(coords).astype(np.int32)
        for start, end in zip(coords[:-1], coords[1:]):
            if (
                (0 <= start[0] < width and 0 <= start[1] < height)
                or (0 <= end[0] < width and 0 <= end[1] < height)
            ):
                cv2.line(canvas, tuple(start), tuple(end), FOV_COLOR_BGR, 2, cv2.LINE_AA)


def draw_front_axis_image(
    canvas: np.ndarray,
    lidar2image: np.ndarray,
    radius: float,
    front_axis: str,
):
    coords = project_lidar_points(
        front_axis_ray(radius, z=0.0, front_axis=front_axis),
        lidar2image,
    )
    if len(coords) < 2:
        return
    height, width = canvas.shape[:2]
    coords = np.round(coords).astype(np.int32)
    for start, end in zip(coords[:-1], coords[1:]):
        if (
            (0 <= start[0] < width and 0 <= start[1] < height)
            or (0 <= end[0] < width and 0 <= end[1] < height)
        ):
            cv2.line(canvas, tuple(start), tuple(end), FRONT_AXIS_COLOR_BGR, 3, cv2.LINE_AA)


def visualize_camera_image(
    fpath: str,
    image_path: str,
    lidar2image: np.ndarray,
    gt_boxes: Optional[LiDARInstance3DBoxes],
    pred_boxes: Optional[LiDARInstance3DBoxes],
    pred_labels: Optional[np.ndarray],
    classes: Sequence[str],
    fov_degree: Sequence[float],
    front_axis: str,
    radius: float,
):
    canvas = mmcv.imread(image_path)
    draw_fov_image(canvas, lidar2image, fov_degree, radius, front_axis)
    draw_front_axis_image(canvas, lidar2image, radius, front_axis)
    draw_boxes_image(canvas, gt_boxes, lidar2image, GT_COLOR_BGR, thickness=3)
    draw_boxes_image(
        canvas,
        pred_boxes,
        lidar2image,
        PRED_COLOR_BGR,
        thickness=2,
        labels=pred_labels,
        classes=classes,
    )
    draw_centers_image(canvas, gt_boxes, lidar2image, GT_COLOR_BGR)
    draw_centers_image(
        canvas,
        pred_boxes,
        lidar2image,
        PRED_COLOR_BGR,
        labels=pred_labels,
        classes=classes,
    )
    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)


def write_meta(
    fpath: str,
    scene: str,
    scene_order: int,
    index: int,
    info: dict,
    fov_degree: Sequence[float],
    front_axis: str,
    gt_boxes: Optional[LiDARInstance3DBoxes],
    pred_boxes: Optional[LiDARInstance3DBoxes],
):
    meta = {
        "scene": scene,
        "scene_order": scene_order,
        "dataset_index": index,
        "sample_token": info["token"],
        "timestamp": info.get("timestamp"),
        "fov_degree": list(fov_degree),
        "fov_front_axis": front_axis,
        "gt_count": int(len(gt_boxes)) if gt_boxes is not None else 0,
        "pred_count": int(len(pred_boxes)) if pred_boxes is not None else 0,
    }
    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(json_safe(meta), f, indent=2)


def to_float_list(values) -> List[float]:
    return [float(value) for value in np.asarray(values).reshape(-1)]


def json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(val) for val in value]
    return value


def box_instances_to_records(
    source: str,
    boxes: Optional[LiDARInstance3DBoxes],
    labels: Optional[np.ndarray],
    classes: Sequence[str],
    fov_degree: Sequence[float],
    front_axis: str,
    scores: Optional[np.ndarray] = None,
) -> List[dict]:
    if boxes is None or labels is None or len(boxes) == 0:
        return []

    tensor = to_numpy(boxes.tensor)
    centers = to_numpy(boxes.gravity_center)
    labels = np.asarray(labels, dtype=np.int64)
    if scores is not None:
        scores = np.asarray(scores, dtype=np.float32)

    angles = lidar_fov_angle_degree(centers, front_axis)
    fusion_mask = angle_in_fov(angles, fov_degree)
    records = []
    for index in range(len(boxes)):
        label = int(labels[index])
        score = None if scores is None else float(scores[index])
        box_tensor = tensor[index]
        record = {
            "source": source,
            "index": index,
            "class_name": classes[label] if 0 <= label < len(classes) else str(label),
            "label": label,
            "confidence": score,
            "space": "fusion_space" if bool(fusion_mask[index]) else "lidar_only_space",
            "fov_angle_deg": float(angles[index]),
            "center": to_float_list(centers[index]),
            "box": to_float_list(box_tensor),
            "box_size": to_float_list(box_tensor[3:6]),
            "yaw": float(box_tensor[6]) if len(box_tensor) > 6 else None,
            "velocity": to_float_list(box_tensor[7:9])
            if len(box_tensor) >= 9
            else None,
        }
        records.append(record)
    return records


def match_predictions_to_gt(
    gt_boxes: Optional[LiDARInstance3DBoxes],
    gt_labels: Optional[np.ndarray],
    pred_boxes: Optional[LiDARInstance3DBoxes],
    pred_labels: Optional[np.ndarray],
    pred_scores: Optional[np.ndarray],
    match_distance_threshold: float,
) -> Dict[int, Dict[str, object]]:
    if (
        gt_boxes is None
        or gt_labels is None
        or pred_boxes is None
        or pred_labels is None
        or len(gt_boxes) == 0
        or len(pred_boxes) == 0
    ):
        return {}

    gt_centers = to_numpy(gt_boxes.gravity_center)[:, :2]
    pred_centers = to_numpy(pred_boxes.gravity_center)[:, :2]
    gt_labels = np.asarray(gt_labels, dtype=np.int64)
    pred_labels = np.asarray(pred_labels, dtype=np.int64)
    if pred_scores is None:
        pred_scores = np.zeros(len(pred_boxes), dtype=np.float32)
    pred_scores = np.asarray(pred_scores, dtype=np.float32)

    matched_gt = set()
    matches = {}
    pred_order = sorted(
        range(len(pred_boxes)),
        key=lambda pred_index: float(pred_scores[pred_index]),
        reverse=True,
    )

    for pred_index in pred_order:
        candidate_gt = [
            gt_index
            for gt_index in range(len(gt_boxes))
            if gt_index not in matched_gt and gt_labels[gt_index] == pred_labels[pred_index]
        ]
        if not candidate_gt:
            continue

        distances = np.linalg.norm(
            gt_centers[np.asarray(candidate_gt)] - pred_centers[pred_index],
            axis=1,
        )
        best_pos = int(np.argmin(distances))
        best_gt_index = int(candidate_gt[best_pos])
        best_distance = float(distances[best_pos])
        if best_distance < match_distance_threshold:
            matched_gt.add(best_gt_index)
            matches[best_gt_index] = {
                "prediction_index": int(pred_index),
                "center_distance": best_distance,
            }

    return matches


def write_instances_json(
    fpath: str,
    scene: str,
    scene_order: int,
    index: int,
    info: dict,
    classes: Sequence[str],
    fov_degree: Sequence[float],
    front_axis: str,
    score_thr: float,
    match_distance_threshold: float,
    gt_boxes: Optional[LiDARInstance3DBoxes],
    gt_labels: Optional[np.ndarray],
    pred_boxes: Optional[LiDARInstance3DBoxes],
    pred_labels: Optional[np.ndarray],
    pred_scores: Optional[np.ndarray],
):
    cams = {
        name: camera_info.get("data_path")
        for name, camera_info in info.get("cams", {}).items()
    }
    gt_instances = box_instances_to_records(
        "gt",
        gt_boxes,
        gt_labels,
        classes,
        fov_degree,
        front_axis,
    )
    pred_instances = box_instances_to_records(
        "prediction",
        pred_boxes,
        pred_labels,
        classes,
        fov_degree,
        front_axis,
        scores=pred_scores,
    )
    matches = match_predictions_to_gt(
        gt_boxes,
        gt_labels,
        pred_boxes,
        pred_labels,
        pred_scores,
        match_distance_threshold,
    )
    for gt_index, gt_record in enumerate(gt_instances):
        match = matches.get(gt_index)
        if match is None:
            gt_record["matched_prediction"] = None
            continue
        pred_record = copy.deepcopy(pred_instances[match["prediction_index"]])
        pred_record["center_distance"] = match["center_distance"]
        gt_record["matched_prediction"] = pred_record

    payload = {
        "scene": scene,
        "scene_order": scene_order,
        "dataset_index": index,
        "sample_token": info["token"],
        "timestamp": info.get("timestamp"),
        "lidar_path": info.get("lidar_path"),
        "camera_paths": cams,
        "fov_degree": list(fov_degree),
        "fov_front_axis": front_axis,
        "coordinate": "lidar",
        "score_threshold": float(score_thr),
        "match_distance_threshold": float(match_distance_threshold),
        "matching_rule": (
            "same_class, score_desc_prediction_order, nearest_unmatched_gt, "
            "BEV center distance < match_distance_threshold"
        ),
        "gt_count": int(len(gt_boxes)) if gt_boxes is not None else 0,
        "pred_count": int(len(pred_boxes)) if pred_boxes is not None else 0,
        "matched_gt_count": len(matches),
        "missed_gt_count": len(gt_instances) - len(matches),
        "gt_instances": gt_instances,
        "pred_instances": pred_instances,
    }
    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2)


def main():
    args, opts = parse_args()
    cfg = load_config(args.config, opts)
    custom_eval = get_custom_eval_cfg(cfg, args)
    fov_degree = custom_eval["fov_degree"]
    fov_front_axis = custom_eval["fov_front_axis"]
    match_distance_threshold = custom_eval["match_distance_threshold"]
    predictions = load_predictions(args.result_json)
    prediction_outputs = load_prediction_pkl(args.prediction_pkl)

    dataset = build_dataset(cfg.data[args.split])
    classes = list(dataset.CLASSES)
    point_cloud_range = cfg.get("point_cloud_range", [-54, -54, -5, 54, 54, 3])
    if prediction_outputs is not None and len(prediction_outputs) != len(dataset):
        raise ValueError(
            "prediction pkl length must match dataset length: {} != {}".format(
                len(prediction_outputs), len(dataset)
            )
        )

    scene_lookup = build_scene_lookup(dataset.data_infos, dataset, args)
    selected = select_scene_samples(
        dataset.data_infos,
        samples_per_scene=args.samples_per_scene,
        max_scenes=args.max_scenes,
        scene_lookup=scene_lookup,
    )
    print(
        "Selected {} samples from {} scenes.".format(
            len(selected), len({scene for scene, _, _ in selected})
        )
    )

    radius = max(
        abs(point_cloud_range[0]),
        abs(point_cloud_range[1]),
        abs(point_cloud_range[3]),
        abs(point_cloud_range[4]),
    )

    for scene, scene_order, index in mmcv.track_iter_progress(selected):
        info = dataset.data_infos[index]
        sample_token = info["token"]
        sample_dir = os.path.join(args.out_dir, str(scene), f"{scene_order:02d}_{sample_token}")

        points = load_lidar_points(info)
        gt_boxes, gt_labels = make_gt_boxes(info, classes)
        if prediction_outputs is not None:
            pred_boxes, pred_labels, pred_scores = make_pred_boxes_from_pkl(
                prediction_outputs[index], args.score_thr
            )
        else:
            pred_boxes, pred_labels, pred_scores = make_pred_boxes(
                info, predictions, classes, args.score_thr
            )

        visualize_bev(
            os.path.join(sample_dir, "bev.png"),
            points,
            gt_boxes,
            pred_boxes,
            pred_labels,
            classes,
            fov_degree,
            fov_front_axis,
            point_cloud_range,
            args.max_points,
            args.bev_dpi,
        )

        lidar2image_by_camera = get_lidar2image_by_camera(info)
        for camera_name in args.camera_names:
            camera_info = info.get("cams", {}).get(camera_name)
            lidar2image = lidar2image_by_camera.get(camera_name)
            if camera_info is None or lidar2image is None:
                continue
            visualize_camera_image(
                os.path.join(sample_dir, f"{camera_name}.png"),
                camera_info["data_path"],
                lidar2image,
                gt_boxes,
                pred_boxes,
                pred_labels,
                classes,
                fov_degree,
                fov_front_axis,
                radius,
            )

        write_meta(
            os.path.join(sample_dir, "meta.json"),
            scene,
            scene_order,
            index,
            info,
            fov_degree,
            fov_front_axis,
            gt_boxes,
            pred_boxes,
        )

        write_instances_json(
            os.path.join(sample_dir, "instances.json"),
            scene,
            scene_order,
            index,
            info,
            classes,
            fov_degree,
            fov_front_axis,
            args.score_thr,
            match_distance_threshold,
            gt_boxes,
            gt_labels,
            pred_boxes,
            pred_labels,
            pred_scores,
        )


if __name__ == "__main__":
    main()
