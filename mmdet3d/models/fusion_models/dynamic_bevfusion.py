"""Backward-compatible name for the modular camera/LiDAR fusion model."""

from mmdet3d.models import FUSIONMODELS

from .modular_bevfusion import ModularBEVFusion


@FUSIONMODELS.register_module()
class DynamicBEVFusion(ModularBEVFusion):
    """Compatibility alias preserving existing configs and checkpoint keys."""

    pass


__all__ = ["DynamicBEVFusion"]

