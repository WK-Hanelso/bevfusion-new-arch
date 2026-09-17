"""Pure dispatch helpers for modular BEV fusion.

This module deliberately has no torch/MMCV imports so its call contracts can
be tested in the lightweight config-development environment.
"""

SENSOR_ORDER = ("camera", "lidar")


def select_fuser_call(fuser, sensor_features):
    """Fuse camera/LiDAR features according to the fuser's declared ABI.

    A single configured sensor bypasses fusion.  This keeps lidar-only and
    camera-only models valid without requiring a dummy feature for the absent
    sensor.
    """

    ordered = [sensor_features[name] for name in SENSOR_ORDER if name in sensor_features]
    if not ordered:
        raise ValueError("no camera or LiDAR feature was produced")
    if len(ordered) == 1:
        return ordered[0]
    if fuser is None:
        raise ValueError("multiple sensor features require a fuser")

    input_style = getattr(fuser, "input_style", "list")
    if input_style == "list":
        return fuser(ordered)
    if input_style == "named":
        return fuser(
            camera_bev=sensor_features["camera"],
            lidar_bev=sensor_features["lidar"],
        )
    raise ValueError(f"unsupported fuser input_style: {input_style!r}")


def select_head_call(head, decoded, sensor_features, metas):
    """Call a standard or LiDAR-decoupled object head.

    DAL uses the LiDAR encoder output when available.  For a camera-only model,
    the decoded fused feature is the documented regression-input substitute.
    """

    if getattr(head, "needs_lidar_bev", False):
        lidar_bev = sensor_features.get("lidar", decoded)
        return head(decoded, lidar_bev, metas)
    return head(decoded, metas)


__all__ = ["select_fuser_call", "select_head_call"]
