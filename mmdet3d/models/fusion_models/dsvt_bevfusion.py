"""BEVFusion variant whose LiDAR encoder consumes raw point batches."""

import copy

from torch import nn

from mmdet3d.models import FUSIONMODELS
from mmdet3d.models.builder import build_backbone

from .bevfusion import BEVFusion


@FUSIONMODELS.register_module()
class DSVTBEVFusion(BEVFusion):
    """Preserve BEVFusion's decoder/heads while replacing only LiDAR encoding."""

    def __init__(self, encoders, *args, **kwargs):
        encoders = copy.deepcopy(encoders)
        lidar_config = encoders.pop("lidar", None)
        super().__init__(encoders=encoders, *args, **kwargs)
        if lidar_config is None:
            raise ValueError("DSVTBEVFusion requires encoders.lidar")
        self.encoders["lidar"] = nn.ModuleDict(
            {"backbone": build_backbone(lidar_config)}
        )

    def extract_features(self, points, sensor):
        if sensor == "lidar":
            return self.encoders["lidar"]["backbone"](points)
        return super().extract_features(points, sensor)


__all__ = ["DSVTBEVFusion"]
