import contextlib
import copy
import io
import time
from os import path as osp
from typing import Any, Dict, Optional

import mmcv
import torch
from mmdet.datasets import DATASETS
from mmdet3d.utils import print_log

from ..nuscenes_dataset import NuScenesDataset
from .detection_eval import CustomDetectionEval


@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    """NuScenesDataset variant that reports official and FOV-split metrics."""

    STATIC_OBJECT_CLASSES = {"barrier", "traffic_cone"}

    DEFAULT_CUSTOM_EVAL = {
        "enabled": True,
        "fov_degree": [-85.0, 85.0],
        "lidar_sensor": "LIDAR_TOP",
        "fov_front_axis": "y",
        "spaces": [
            "nuscenes_official",
            "fusion_space",
            "lidar_only_space",
            "bev360",
        ],
        "output_dir_name": "custom_eval",
        "render_curves": False,
        "plot_examples": 0,
        "verbose": False,
        "suppress_eval_stdout": True,
        "keep_legacy_nuscenes_keys": True,
        "match_distance_threshold": 2.0,
        "dump_missed_gt": True,
        "missed_gt_filename": "missed_gt.json",
        "missed_gt_max_records": -1,
        "log_file": None,
        "log_filename": "custom_eval.log",
    }

    def evaluate(
        self,
        results,
        metric="bbox",
        jsonfile_prefix=None,
        result_names=["pts_bbox"],
        **kwargs,
    ):
        custom_eval = self._build_custom_eval_cfg(kwargs.pop("custom_eval", None))
        if not custom_eval["enabled"]:
            return super().evaluate(
                results,
                metric=metric,
                jsonfile_prefix=jsonfile_prefix,
                result_names=result_names,
                **kwargs,
            )

        logger = kwargs.get("logger", None)
        metrics = {}

        if "masks_bev" in results[0]:
            metrics.update(self.evaluate_map(results))

        if "boxes_3d" in results[0]:
            result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

            if isinstance(result_files, dict):
                for name in result_names:
                    print("Evaluating bboxes of {}".format(name))
                    ret_dict = self._evaluate_single(
                        result_files[name],
                        logger=logger,
                        metric=metric,
                        result_name=name,
                        custom_eval=custom_eval,
                    )
                    metrics.update(ret_dict)
            elif isinstance(result_files, str):
                metrics.update(
                    self._evaluate_single(
                        result_files,
                        logger=logger,
                        metric=metric,
                        custom_eval=custom_eval,
                    )
                )

            if tmp_dir is not None:
                tmp_dir.cleanup()

        return metrics

    def _evaluate_single(
        self,
        result_path,
        logger=None,
        metric="bbox",
        result_name="pts_bbox",
        custom_eval: Optional[Dict[str, Any]] = None,
    ):
        custom_eval = self._build_custom_eval_cfg(custom_eval)
        if not custom_eval["enabled"]:
            return super()._evaluate_single(
                result_path,
                logger=logger,
                metric=metric,
                result_name=result_name,
            )

        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import DetectionEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(version=self.version, dataroot=self.dataset_root, verbose=False)
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
        }
        eval_set = eval_set_map[self.version]

        summaries = {}
        details = {}
        space_stats = {}
        match_stats = {}
        progress_lines = []
        spaces = custom_eval["spaces"]
        if "nuscenes_official" in spaces:
            self._log_custom_eval(
                "[custom_eval] START nuscenes_official", logger, progress_lines
            )
            start_time = time.time()
            official_summary = self._run_detection_eval(
                DetectionEval(
                    nusc,
                    config=self.eval_detection_configs,
                    result_path=result_path,
                    eval_set=eval_set,
                    output_dir=output_dir,
                    verbose=custom_eval["verbose"],
                ),
                custom_eval,
            )
            self._log_custom_eval(
                "[custom_eval] DONE nuscenes_official ({:.1f}s)".format(
                    time.time() - start_time
                ),
                logger,
                progress_lines,
            )
            summaries["nuscenes_official"] = official_summary
            if custom_eval["keep_legacy_nuscenes_keys"]:
                details.update(self._metrics_to_detail(official_summary))
            details.update(
                self._metrics_to_detail(official_summary, prefix="nuscenes_official")
            )

        custom_root = osp.join(output_dir, custom_eval["output_dir_name"])
        for space in spaces:
            if space == "nuscenes_official":
                continue
            space_output_dir = osp.join(custom_root, space)
            self._log_custom_eval(
                "[custom_eval] START {}".format(space), logger, progress_lines
            )
            start_time = time.time()
            evaluator = CustomDetectionEval(
                nusc,
                config=self.eval_detection_configs,
                result_path=result_path,
                eval_set=eval_set,
                output_dir=space_output_dir,
                verbose=custom_eval["verbose"],
                space=space,
                fov_degree=custom_eval["fov_degree"],
                lidar_sensor=custom_eval["lidar_sensor"],
                fov_front_axis=custom_eval["fov_front_axis"],
                match_distance_threshold=custom_eval["match_distance_threshold"],
            )
            summary = self._run_detection_eval(
                evaluator,
                custom_eval,
            )
            space_stats[space] = evaluator.space_stats
            match_stats[space] = evaluator.match_stats
            missed_gt_path = self._dump_missed_gt(evaluator, space_output_dir, custom_eval)
            self._log_custom_eval(
                "[custom_eval] DONE {} ({:.1f}s) | {} | {}{}".format(
                    space,
                    time.time() - start_time,
                    self._format_space_count_log(evaluator.space_stats),
                    self._format_match_count_log(evaluator.match_stats),
                    " | missed_gt: {}".format(missed_gt_path)
                    if missed_gt_path is not None
                    else "",
                ),
                logger,
                progress_lines,
            )
            summaries[space] = summary
            details.update(self._metrics_to_detail(summary, prefix=space))
            details.update(self._match_stats_to_detail(evaluator.match_stats, prefix=space))

        summary_text = self._print_custom_eval_summary(
            summaries, custom_eval, logger, space_stats, match_stats
        )
        self._write_custom_eval_log(
            output_dir, custom_eval, progress_lines, summary_text, logger
        )
        return details

    def _build_custom_eval_cfg(self, custom_eval: Optional[Dict[str, Any]]):
        cfg = copy.deepcopy(self.DEFAULT_CUSTOM_EVAL)
        if custom_eval is not None:
            cfg.update(dict(custom_eval))

        valid_spaces = {"nuscenes_official", *CustomDetectionEval.VALID_SPACES}
        unknown_spaces = set(cfg["spaces"]) - valid_spaces
        if unknown_spaces:
            raise ValueError("Unknown custom_eval spaces: {}".format(unknown_spaces))

        if len(cfg["fov_degree"]) != 2:
            raise ValueError("custom_eval.fov_degree must have two values")
        cfg["fov_degree"] = [float(cfg["fov_degree"][0]), float(cfg["fov_degree"][1])]
        cfg["fov_front_axis"] = str(cfg["fov_front_axis"]).lower()
        if cfg["fov_front_axis"] not in ("x", "y"):
            raise ValueError("custom_eval.fov_front_axis must be 'x' or 'y'")
        cfg["match_distance_threshold"] = float(cfg["match_distance_threshold"])
        cfg["missed_gt_max_records"] = int(cfg["missed_gt_max_records"])
        if cfg["log_file"] is not None:
            cfg["log_file"] = str(cfg["log_file"])
        cfg["log_filename"] = str(cfg["log_filename"])
        return cfg

    def _log_custom_eval(self, message, logger, progress_lines):
        progress_lines.append(message)
        print_log(message, logger=logger)

    def _run_detection_eval(self, evaluator, custom_eval):
        if custom_eval["suppress_eval_stdout"]:
            with contextlib.redirect_stdout(io.StringIO()):
                return evaluator.main(
                    plot_examples=custom_eval["plot_examples"],
                    render_curves=custom_eval["render_curves"],
                )
        return evaluator.main(
            plot_examples=custom_eval["plot_examples"],
            render_curves=custom_eval["render_curves"],
        )

    def _metrics_to_detail(self, metrics, prefix=None):
        detail = dict()
        metric_prefix = "object" if prefix is None else "object/{}".format(prefix)

        for name in self.CLASSES:
            for k, v in metrics["label_aps"][name].items():
                val = float("{:.4f}".format(v))
                detail["{}/{}_ap_dist_{}".format(metric_prefix, name, k)] = val
            for k, v in metrics["label_tp_errors"][name].items():
                val = float("{:.4f}".format(v))
                detail["{}/{}_{}".format(metric_prefix, name, k)] = val

        for k, v in metrics["tp_errors"].items():
            val = float("{:.4f}".format(v))
            detail["{}/{}".format(metric_prefix, self.ErrNameMapping[k])] = val

        detail["{}/nds".format(metric_prefix)] = metrics["nd_score"]
        detail["{}/map".format(metric_prefix)] = metrics["mean_ap"]
        return detail

    def _match_stats_to_detail(self, stats, prefix):
        detail = {}
        metric_prefix = "object/{}".format(prefix)
        for key in (
            "gt_total",
            "pred_total",
            "matched_gt",
            "missed_gt",
            "matched_pred",
            "unmatched_pred",
        ):
            detail["{}/match_{}".format(metric_prefix, key)] = stats[key]

        for class_name, class_stats in stats["per_class"].items():
            for key in (
                "gt_total",
                "pred_total",
                "matched_gt",
                "missed_gt",
                "matched_pred",
                "unmatched_pred",
            ):
                detail[
                    "{}/{}_match_{}".format(metric_prefix, class_name, key)
                ] = class_stats[key]
        return detail

    def _format_metric(self, value, precision=4):
        value = float(value)
        if value != value:
            return "nan"
        return "{:.{}f}".format(value, precision)

    def _mean_metric(self, metric_dict):
        values = []
        for value in metric_dict.values():
            value = float(value)
            if value == value:
                values.append(value)
        if not values:
            return float("nan")
        return sum(values) / len(values)

    def _append_official_style_result(self, lines, name, metrics):
        tp = metrics["tp_errors"]
        lines.append("")
        lines.append("[{}]".format(name))
        lines.append("mAP: {}".format(self._format_metric(metrics["mean_ap"])))
        lines.append("mATE: {}".format(self._format_metric(tp["trans_err"])))
        lines.append("mASE: {}".format(self._format_metric(tp["scale_err"])))
        lines.append("mAOE: {}".format(self._format_metric(tp["orient_err"])))
        lines.append("mAVE: {}".format(self._format_metric(tp["vel_err"])))
        lines.append("mAAE: {}".format(self._format_metric(tp["attr_err"])))
        lines.append("NDS: {}".format(self._format_metric(metrics["nd_score"])))
        if "eval_time" in metrics:
            lines.append("Eval time: {:.1f}s".format(float(metrics["eval_time"])))
        lines.append("")
        lines.append("Per-class results:")
        lines.append(
            "{:<24}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}".format(
                "Object Class", "AP", "ATE", "ASE", "AOE", "AVE", "AAE"
            )
        )

        for class_name in self.CLASSES:
            class_tp = metrics["label_tp_errors"][class_name]
            lines.append(
                "{:<24}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}".format(
                    class_name,
                    self._format_metric(
                        self._mean_metric(metrics["label_aps"][class_name]), 3
                    ),
                    self._format_metric(class_tp["trans_err"], 3),
                    self._format_metric(class_tp["scale_err"], 3),
                    self._format_metric(class_tp["orient_err"], 3),
                    self._format_metric(class_tp["vel_err"], 3),
                    self._format_metric(class_tp["attr_err"], 3),
                )
            )

    def _format_space_count_log(self, stats):
        return "GT kept {}/{} (fusion={}, lidar_only={})".format(
            stats["gt_after_space_filter"],
            stats["gt_before_space_filter"],
            stats["gt_fusion_space"],
            stats["gt_lidar_only_space"],
        )

    def _format_match_count_log(self, stats):
        return "Match GT {}/{} (missed={}) | dist<{:.2f}m".format(
            stats["matched_gt"],
            stats["gt_total"],
            stats["missed_gt"],
            stats["match_distance_threshold"],
        )

    def _dump_missed_gt(self, evaluator, output_dir, custom_eval):
        if not custom_eval["dump_missed_gt"]:
            return None

        max_records = custom_eval["missed_gt_max_records"]
        records = evaluator.missed_gt_records
        if max_records >= 0:
            records = records[:max_records]

        mmcv.mkdir_or_exist(output_dir)
        missed_gt_path = osp.join(output_dir, custom_eval["missed_gt_filename"])
        mmcv.dump(
            {
                "space": evaluator.space,
                "fov_front_axis": evaluator.fov_front_axis,
                "match_distance_threshold": evaluator.match_distance_threshold,
                "missed_gt_count": evaluator.match_stats["missed_gt"],
                "saved_missed_gt_count": len(records),
                "missed_gt": records,
            },
            missed_gt_path,
        )
        return missed_gt_path

    def _append_region_count_summary(self, lines, space_stats):
        if not space_stats:
            return

        lines.append("")
        lines.append("Region counts:")
        lines.append(
            "{:<18}\t{:<14}\t{:<10}\t{:<14}".format(
                "Eval Space",
                "GT kept/total",
                "GT fusion",
                "GT lidar-only",
            )
        )
        for name, stats in space_stats.items():
            lines.append(
                "{:<18}\t{:<14}\t{:<10}\t{:<14}".format(
                    name,
                    "{}/{}".format(
                        stats["gt_after_space_filter"],
                        stats["gt_before_space_filter"],
                    ),
                    stats["gt_fusion_space"],
                    stats["gt_lidar_only_space"],
                )
            )

        if all(k in space_stats for k in ("fusion_space", "lidar_only_space", "bev360")):
            gt_split = (
                space_stats["fusion_space"]["gt_after_space_filter"]
                + space_stats["lidar_only_space"]["gt_after_space_filter"]
            )
            gt_total = space_stats["bev360"]["gt_after_space_filter"]
            lines.append(
                "Split check: GT {} / {} = {}".format(
                    gt_split,
                    gt_total,
                    "OK" if gt_split == gt_total else "MISMATCH",
                )
            )

    def _append_match_count_summary(self, lines, match_stats):
        if not match_stats:
            return

        lines.append("")
        lines.append("Match counts:")
        lines.append(
            "{:<18}\t{:<16}\t{:<10}\t{:<12}".format(
                "Eval Space",
                "GT matched/total",
                "Missed GT",
                "Distance",
            )
        )
        for name, stats in match_stats.items():
            lines.append(
                "{:<18}\t{:<16}\t{:<10}\t{:<12}".format(
                    name,
                    "{}/{}".format(stats["matched_gt"], stats["gt_total"]),
                    stats["missed_gt"],
                    "<{:.2f}m".format(stats["match_distance_threshold"]),
                )
            )

        self._append_classwise_missed_gt_summary(lines, match_stats)

    def _object_motion_type(self, class_name):
        if class_name in self.STATIC_OBJECT_CLASSES:
            return "static"
        return "dynamic"

    def _append_classwise_missed_gt_summary(self, lines, match_stats):
        lines.append("")
        lines.append("Class-wise missed GT:")
        lines.append(
            "{:<18}\t{:<24}\t{:<10}\t{:<12}\t{:<10}".format(
                "Eval Space",
                "Object Class",
                "Type",
                "Missed/GT",
                "Missed %",
            )
        )
        for space_name, stats in match_stats.items():
            per_class = stats.get("per_class", {})
            for class_name in self.CLASSES:
                class_stats = per_class.get(class_name)
                if class_stats is None or class_stats["gt_total"] == 0:
                    continue
                missed = class_stats["missed_gt"]
                gt_total = class_stats["gt_total"]
                missed_rate = 100.0 * missed / gt_total
                lines.append(
                    "{:<18}\t{:<24}\t{:<10}\t{:<12}\t{:<10}".format(
                        space_name,
                        class_name,
                        self._object_motion_type(class_name),
                        "{}/{}".format(missed, gt_total),
                        "{:.2f}".format(missed_rate),
                    )
                )

    def _format_custom_eval_summary(
        self, summaries, custom_eval, logger, space_stats=None, match_stats=None
    ):
        if not summaries:
            return None

        lines = [
            "Custom nuScenes eval results "
            "(LiDAR FOV: [{:.1f}, {:.1f}] deg, sensor: {}, front_axis: +{})".format(
                custom_eval["fov_degree"][0],
                custom_eval["fov_degree"][1],
                custom_eval["lidar_sensor"],
                custom_eval["fov_front_axis"].upper(),
            )
        ]

        self._append_region_count_summary(lines, space_stats or {})
        self._append_match_count_summary(lines, match_stats or {})

        for name, metrics in summaries.items():
            self._append_official_style_result(lines, name, metrics)

        return "\n".join(lines)

    def _print_custom_eval_summary(
        self, summaries, custom_eval, logger, space_stats=None, match_stats=None
    ):
        summary_text = self._format_custom_eval_summary(
            summaries, custom_eval, logger, space_stats, match_stats
        )
        if summary_text is not None:
            print_log(summary_text, logger=logger)
        return summary_text

    def _write_custom_eval_log(
        self, output_dir, custom_eval, progress_lines, summary_text, logger
    ):
        if summary_text is None:
            return None

        if custom_eval["log_file"] is not None:
            log_path = custom_eval["log_file"]
        else:
            log_path = osp.join(
                output_dir,
                custom_eval["output_dir_name"],
                custom_eval["log_filename"],
            )

        log_dir = osp.dirname(log_path)
        if log_dir:
            mmcv.mkdir_or_exist(log_dir)
        log_text = "\n".join(progress_lines + ["", summary_text, ""])
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(log_text)
        print_log("[custom_eval] LOG saved to {}".format(log_path), logger=logger)
        return log_path
