"""MMDetection registry adapter for DSVT-Pillar."""

from mmdet.models import BACKBONES

from .dsvt_core import DSVTLidarEncoder


BACKBONES.register_module()(DSVTLidarEncoder)

__all__ = ["DSVTLidarEncoder"]
