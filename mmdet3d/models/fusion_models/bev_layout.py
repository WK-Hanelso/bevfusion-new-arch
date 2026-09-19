"""Shared BEV tensor layout contract for modular fusion models."""

CANONICAL_BEV_LAYOUT = "yx"
VALID_BEV_LAYOUTS = frozenset(("xy", "yx"))


def inject_head_bev_layout(heads):
    """Set the canonical contract on a modular model's object head config."""
    if heads.get("object") is not None:
        heads["object"]["bev_layout"] = CANONICAL_BEV_LAYOUT
    return heads


def canonicalize_bev(feature, module, module_name):
    """Return ``feature`` in canonical ``[B, C, Y, X]`` layout."""
    layout = getattr(module, "bev_layout", None)
    if layout not in VALID_BEV_LAYOUTS:
        raise ValueError(
            f"{module_name} must declare bev_layout as 'xy' or 'yx'; got {layout!r}"
        )
    if layout != CANONICAL_BEV_LAYOUT:
        feature = feature.transpose(-1, -2).contiguous()
    return feature


__all__ = [
    "CANONICAL_BEV_LAYOUT",
    "canonicalize_bev",
    "inject_head_bev_layout",
]
