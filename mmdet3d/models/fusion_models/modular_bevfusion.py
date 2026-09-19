"""One BEVFusion model class for legacy and new ablation modules."""

import copy

from mmcv.runner import auto_fp16
from torch import nn

from mmdet3d.models import FUSIONMODELS
from mmdet3d.models.builder import build_backbone

from .bevfusion import BEVFusion
from .bev_layout import canonicalize_bev, inject_head_bev_layout
from .dispatch import select_fuser_call, select_head_call


@FUSIONMODELS.register_module()
class ModularBEVFusion(BEVFusion):
    """Run any supported LiDAR, camera, fuser and object-head combination.

    A raw-point LiDAR encoder is represented by a direct backbone config with
    a top-level ``type`` (currently ``DSVTLidarEncoder``).  Legacy voxel-based
    encoders retain BEVFusion's ``voxelize`` + ``backbone`` config shape.
    """

    def __init__(self, encoders, *args, **kwargs):
        encoders = copy.deepcopy(encoders)
        args = list(args)
        if "heads" in kwargs:
            heads = copy.deepcopy(kwargs["heads"])
            kwargs["heads"] = heads
        elif len(args) >= 3:
            heads = copy.deepcopy(args[2])
            args[2] = heads
        else:
            raise TypeError("ModularBEVFusion requires a heads configuration")
        inject_head_bev_layout(heads)

        lidar_config = encoders.get("lidar")
        self._raw_lidar_encoder = bool(
            lidar_config is not None and lidar_config.get("type") is not None
        )
        if self._raw_lidar_encoder:
            lidar_config = encoders.pop("lidar")
            # Torchpack recursively merges dictionaries.  These legacy-only
            # keys may remain when a leaf selects a direct raw-point encoder.
            lidar_config.pop("voxelize", None)
            lidar_config.pop("backbone", None)
            lidar_config.pop("voxelize_reduce", None)

        super().__init__(encoders, *args, **kwargs)

        if self._raw_lidar_encoder:
            self.encoders["lidar"] = nn.ModuleDict(
                {"backbone": build_backbone(lidar_config)}
            )

    def extract_features(self, points, sensor):
        if sensor == "lidar" and self._raw_lidar_encoder:
            return self.encoders["lidar"]["backbone"](points)
        return super().extract_features(points, sensor)

    @auto_fp16(apply_to=("img", "points", "camera_bev"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths=None,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        camera_bev=None,
        **kwargs,
    ):
        sensor_features = {}
        auxiliary_losses = {}
        sensor_order = (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        )
        for sensor in sensor_order:
            if sensor == "camera":
                feature = camera_bev
                if feature is None:
                    feature = self.extract_camera_features(
                        img,
                        points,
                        radar,
                        camera2ego,
                        lidar2ego,
                        lidar2camera,
                        lidar2image,
                        camera_intrinsics,
                        camera2lidar,
                        img_aug_matrix,
                        lidar_aug_matrix,
                        metas,
                        gt_depths=depths,
                    )
                    if self.use_depth_loss:
                        feature, auxiliary_losses["depth"] = feature[0], feature[-1]
                    feature = canonicalize_bev(
                        feature,
                        self.encoders["camera"]["vtransform"],
                        "camera vtransform",
                    )
                sensor_features["camera"] = feature
            elif sensor == "lidar":
                if points is None:
                    raise ValueError("a configured LiDAR encoder requires points")
                feature = self.extract_features(points, sensor)
                sensor_features["lidar"] = canonicalize_bev(
                    feature,
                    self.encoders["lidar"]["backbone"],
                    "LiDAR encoder",
                )
            else:
                raise ValueError(f"unsupported sensor: {sensor}")

        # Preserve DynamicBEVFusion's external camera-BEV seam even when its
        # camera encoder is intentionally omitted from the config.
        if camera_bev is not None and "camera" not in sensor_features:
            sensor_features["camera"] = camera_bev

        fused = select_fuser_call(self.fuser, sensor_features)
        batch_size = fused.shape[0]
        decoded = self.decoder["backbone"](fused)
        decoded = self.decoder["neck"](decoded)

        if self.training:
            outputs = {}
            for head_type, head in self.heads.items():
                if head_type == "object":
                    predictions = select_head_call(
                        head, decoded, sensor_features, metas
                    )
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, predictions)
                elif head_type == "map":
                    losses = head(decoded, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {head_type}")
                for name, value in losses.items():
                    key = "loss" if value.requires_grad else "stats"
                    scale = self.loss_scale[head_type] if value.requires_grad else 1.0
                    outputs[f"{key}/{head_type}/{name}"] = value * scale
            if self.use_depth_loss:
                if "depth" not in auxiliary_losses:
                    raise ValueError("depth loss is enabled but was not returned")
                outputs["loss/depth"] = auxiliary_losses["depth"]
            return outputs

        outputs = [{} for _ in range(batch_size)]
        for head_type, head in self.heads.items():
            if head_type == "object":
                predictions = select_head_call(head, decoded, sensor_features, metas)
                bboxes = head.get_bboxes(predictions, metas)
                for batch_index, (boxes, scores, labels) in enumerate(bboxes):
                    outputs[batch_index].update(
                        boxes_3d=boxes.to("cpu"),
                        scores_3d=scores.cpu(),
                        labels_3d=labels.cpu(),
                    )
            elif head_type == "map":
                logits = head(decoded)
                for batch_index in range(batch_size):
                    outputs[batch_index].update(
                        masks_bev=logits[batch_index].cpu(),
                        gt_masks_bev=gt_masks_bev[batch_index].cpu(),
                    )
            else:
                raise ValueError(f"unsupported head: {head_type}")
        return outputs


__all__ = ["ModularBEVFusion"]
