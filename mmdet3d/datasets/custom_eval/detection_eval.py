from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from nuscenes.eval.detection.evaluate import DetectionEval
from pyquaternion import Quaternion


class CustomDetectionEval(DetectionEval):
    """nuScenes detection evaluator with LiDAR-FOV space filtering.

    The AP/TP metric implementation stays in the official nuScenes devkit.
    This class only partitions already-loaded GT and prediction boxes by
    center point angle in the nuScenes LiDAR sensor coordinate system.
    For nuScenes LIDAR_TOP, the vehicle front direction is the sensor +Y axis.
    """

    VALID_SPACES = ("fusion_space", "lidar_only_space", "bev360")
    STATIC_OBJECT_CLASSES = {"barrier", "traffic_cone"}

    def __init__(
        self,
        *args,
        space: str = "fusion_space",
        fov_degree: Sequence[float] = (-85.0, 85.0),
        lidar_sensor: str = "LIDAR_TOP",
        fov_front_axis: str = "y",
        match_distance_threshold: float = 2.0,
        **kwargs,
    ) -> None:
        if space not in self.VALID_SPACES:
            raise ValueError(
                "space must be one of {}, but got {}".format(self.VALID_SPACES, space)
            )
        if len(fov_degree) != 2:
            raise ValueError("fov_degree must be a [min_degree, max_degree] pair")
        if fov_front_axis not in ("x", "y"):
            raise ValueError("fov_front_axis must be 'x' or 'y'")

        self.space = space
        self.fov_degree = (float(fov_degree[0]), float(fov_degree[1]))
        self.lidar_sensor = lidar_sensor
        self.fov_front_axis = fov_front_axis
        self.match_distance_threshold = float(match_distance_threshold)
        self._transform_cache = {}
        self.space_stats = {}
        self.match_stats = {}
        self.missed_gt_records = []

        super().__init__(*args, **kwargs)

        self.space_stats = self._build_space_stats()
        if self.space != "bev360":
            self._apply_space_filter()
        self.space_stats.update(
            {
                "gt_after_space_filter": len(self.gt_boxes.all),
                "pred_after_space_filter": len(self.pred_boxes.all),
            }
        )
        self.sample_tokens = self.gt_boxes.sample_tokens
        self.match_stats, self.missed_gt_records = self._build_match_stats()

    def _build_space_stats(self) -> Dict[str, object]:
        gt_counts = self._count_boxes_by_space(self.gt_boxes)
        pred_counts = self._count_boxes_by_space(self.pred_boxes)
        return {
            "space": self.space,
            "fov_degree": list(self.fov_degree),
            "lidar_sensor": self.lidar_sensor,
            "fov_front_axis": self.fov_front_axis,
            "sample_count": len(self.gt_boxes.sample_tokens),
            "gt_before_space_filter": gt_counts["total"],
            "gt_fusion_space": gt_counts["fusion_space"],
            "gt_lidar_only_space": gt_counts["lidar_only_space"],
            "pred_before_space_filter": pred_counts["total"],
            "pred_fusion_space": pred_counts["fusion_space"],
            "pred_lidar_only_space": pred_counts["lidar_only_space"],
        }

    def _count_boxes_by_space(self, eval_boxes) -> Dict[str, int]:
        counts = {"total": 0, "fusion_space": 0, "lidar_only_space": 0}
        for sample_token in eval_boxes.sample_tokens:
            for box in eval_boxes[sample_token]:
                counts["total"] += 1
                if self._is_in_fusion_space(sample_token, box.translation):
                    counts["fusion_space"] += 1
                else:
                    counts["lidar_only_space"] += 1
        return counts

    def _build_match_stats(self) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
        class_names = self._get_class_names()
        per_class = {
            name: self._new_match_counter() for name in class_names
        }
        stats = self._new_match_counter()
        stats.update(
            {
                "space": self.space,
                "match_distance_threshold": self.match_distance_threshold,
                "per_class": per_class,
            }
        )
        missed_records = []

        sample_tokens = sorted(
            set(self.gt_boxes.sample_tokens) | set(self.pred_boxes.sample_tokens)
        )
        for sample_token in sample_tokens:
            gt_by_class = self._group_boxes_by_class(self.gt_boxes[sample_token])
            pred_by_class = self._group_boxes_by_class(self.pred_boxes[sample_token])
            sample_classes = sorted(set(gt_by_class) | set(pred_by_class))

            for class_name in sample_classes:
                if class_name not in per_class:
                    per_class[class_name] = self._new_match_counter()
                gt_boxes = gt_by_class.get(class_name, [])
                pred_boxes = sorted(
                    pred_by_class.get(class_name, []),
                    key=lambda box: getattr(box, "detection_score", 0.0),
                    reverse=True,
                )

                class_counter = per_class[class_name]
                stats["gt_total"] += len(gt_boxes)
                stats["pred_total"] += len(pred_boxes)
                class_counter["gt_total"] += len(gt_boxes)
                class_counter["pred_total"] += len(pred_boxes)

                matched_gt_indices, matched_pred_indices = self._match_class_boxes(
                    gt_boxes, pred_boxes
                )

                matched_gt_count = len(matched_gt_indices)
                matched_pred_count = len(matched_pred_indices)
                missed_gt_count = len(gt_boxes) - matched_gt_count
                unmatched_pred_count = len(pred_boxes) - matched_pred_count

                stats["matched_gt"] += matched_gt_count
                stats["matched_pred"] += matched_pred_count
                stats["missed_gt"] += missed_gt_count
                stats["unmatched_pred"] += unmatched_pred_count
                class_counter["matched_gt"] += matched_gt_count
                class_counter["matched_pred"] += matched_pred_count
                class_counter["missed_gt"] += missed_gt_count
                class_counter["unmatched_pred"] += unmatched_pred_count

                for gt_index, gt_box in enumerate(gt_boxes):
                    if gt_index not in matched_gt_indices:
                        missed_records.append(
                            self._make_missed_gt_record(
                                sample_token,
                                class_name,
                                gt_index,
                                gt_box,
                                pred_boxes,
                            )
                        )

        return stats, missed_records

    def _new_match_counter(self) -> Dict[str, int]:
        return {
            "gt_total": 0,
            "pred_total": 0,
            "matched_gt": 0,
            "missed_gt": 0,
            "matched_pred": 0,
            "unmatched_pred": 0,
        }

    def _get_class_names(self) -> List[str]:
        class_names = getattr(self.cfg, "class_names", None)
        if class_names is None and isinstance(self.cfg, dict):
            class_names = self.cfg.get("class_names", None)
        if class_names is None:
            names = set()
            for box in self.gt_boxes.all + self.pred_boxes.all:
                names.add(box.detection_name)
            class_names = sorted(names)
        return list(class_names)

    def _group_boxes_by_class(self, boxes) -> Dict[str, list]:
        grouped = {}
        for box in boxes:
            grouped.setdefault(box.detection_name, []).append(box)
        return grouped

    def _match_class_boxes(self, gt_boxes, pred_boxes) -> Tuple[set, set]:
        matched_gt_indices = set()
        matched_pred_indices = set()
        for pred_index, pred_box in enumerate(pred_boxes):
            best_gt_index = None
            best_distance = float("inf")
            for gt_index, gt_box in enumerate(gt_boxes):
                if gt_index in matched_gt_indices:
                    continue
                distance = self._center_distance(gt_box, pred_box)
                if distance < best_distance:
                    best_distance = distance
                    best_gt_index = gt_index

            if (
                best_gt_index is not None
                and best_distance < self.match_distance_threshold
            ):
                matched_gt_indices.add(best_gt_index)
                matched_pred_indices.add(pred_index)

        return matched_gt_indices, matched_pred_indices

    def _make_missed_gt_record(
        self,
        sample_token: str,
        class_name: str,
        gt_index: int,
        gt_box,
        pred_boxes,
    ) -> Dict[str, object]:
        nearest_pred, nearest_distance = self._find_nearest_prediction(
            gt_box, pred_boxes
        )
        gt_lidar = self._global_to_lidar(sample_token, gt_box.translation)
        record = {
            "space": self.space,
            "sample_token": sample_token,
            "detection_name": class_name,
            "object_motion_type": self._object_motion_type(class_name),
            "gt_index": gt_index,
            "gt_translation": self._to_float_list(gt_box.translation),
            "gt_translation_lidar": self._to_float_list(gt_lidar),
            "gt_lidar_angle_degree": self._lidar_fov_angle_degree(gt_lidar),
            "match_distance_threshold": self.match_distance_threshold,
        }
        if nearest_pred is not None:
            record.update(
                {
                    "nearest_pred_center_distance": float(nearest_distance),
                    "nearest_pred_score": float(
                        getattr(nearest_pred, "detection_score", 0.0)
                    ),
                    "nearest_pred_translation": self._to_float_list(
                        nearest_pred.translation
                    ),
                }
            )
        else:
            record["nearest_pred_center_distance"] = None
            record["nearest_pred_score"] = None
            record["nearest_pred_translation"] = None
        return record

    def _object_motion_type(self, class_name: str) -> str:
        if class_name in self.STATIC_OBJECT_CLASSES:
            return "static"
        return "dynamic"

    def _find_nearest_prediction(self, gt_box, pred_boxes) -> Tuple[Optional[object], Optional[float]]:
        nearest_pred = None
        nearest_distance = None
        for pred_box in pred_boxes:
            distance = self._center_distance(gt_box, pred_box)
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_pred = pred_box
        return nearest_pred, nearest_distance

    def _center_distance(self, gt_box, pred_box) -> float:
        gt_xy = np.asarray(gt_box.translation[:2], dtype=np.float64)
        pred_xy = np.asarray(pred_box.translation[:2], dtype=np.float64)
        return float(np.linalg.norm(gt_xy - pred_xy))

    def _to_float_list(self, values) -> List[float]:
        return [float(value) for value in values]

    def _apply_space_filter(self) -> None:
        keep_fusion = self.space == "fusion_space"
        self._filter_boxes_by_space(self.gt_boxes, keep_fusion)
        self._filter_boxes_by_space(self.pred_boxes, keep_fusion)

    def _filter_boxes_by_space(self, eval_boxes, keep_fusion: bool) -> None:
        for sample_token in eval_boxes.sample_tokens:
            eval_boxes.boxes[sample_token] = [
                box
                for box in eval_boxes[sample_token]
                if self._is_in_fusion_space(sample_token, box.translation) == keep_fusion
            ]

    def _is_in_fusion_space(
        self, sample_token: str, global_translation: Sequence[float]
    ) -> bool:
        lidar_xyz = self._global_to_lidar(sample_token, global_translation)
        angle_degree = self._lidar_fov_angle_degree(lidar_xyz)
        return self._angle_in_fov(angle_degree)

    def _lidar_fov_angle_degree(self, lidar_xyz: Sequence[float]) -> float:
        """Return azimuth where 0 deg is the configured LiDAR front axis."""
        if self.fov_front_axis == "x":
            angle = np.arctan2(lidar_xyz[1], lidar_xyz[0])
        else:
            # nuScenes LIDAR_TOP convention: +Y is vehicle front. Use +X as right,
            # so positive angle is toward vehicle left.
            angle = np.arctan2(-lidar_xyz[0], lidar_xyz[1])
        return float(np.degrees(angle))

    def _angle_in_fov(self, angle_degree: float) -> bool:
        fov_min, fov_max = self.fov_degree
        if fov_min <= fov_max:
            return fov_min <= angle_degree <= fov_max
        return angle_degree >= fov_min or angle_degree <= fov_max

    def _global_to_lidar(
        self, sample_token: str, global_translation: Sequence[float]
    ) -> np.ndarray:
        ego_pose, lidar_pose = self._get_sample_transforms(sample_token)

        point = np.asarray(global_translation, dtype=np.float64)
        ego_rot = Quaternion(ego_pose["rotation"]).rotation_matrix
        ego_trans = np.asarray(ego_pose["translation"], dtype=np.float64)
        point_ego = ego_rot.T.dot(point - ego_trans)

        lidar_rot = Quaternion(lidar_pose["rotation"]).rotation_matrix
        lidar_trans = np.asarray(lidar_pose["translation"], dtype=np.float64)
        return lidar_rot.T.dot(point_ego - lidar_trans)

    def _get_sample_transforms(
        self, sample_token: str
    ) -> Tuple[Dict[str, Sequence[float]], Dict[str, Sequence[float]]]:
        if sample_token in self._transform_cache:
            return self._transform_cache[sample_token]

        sample = self.nusc.get("sample", sample_token)
        if self.lidar_sensor not in sample["data"]:
            raise KeyError(
                "Sensor {} is not available in sample {}".format(
                    self.lidar_sensor, sample_token
                )
            )
        sample_data = self.nusc.get("sample_data", sample["data"][self.lidar_sensor])
        ego_pose = self.nusc.get("ego_pose", sample_data["ego_pose_token"])

        # Use nuScenes calibrated sensor for LIDAR_TOP; no external calibration override.
        calibrated_sensor = self.nusc.get(
            "calibrated_sensor", sample_data["calibrated_sensor_token"]
        )
        lidar_pose = {
            "rotation": calibrated_sensor["rotation"],
            "translation": calibrated_sensor["translation"],
        }

        self._transform_cache[sample_token] = (ego_pose, lidar_pose)
        return self._transform_cache[sample_token]
