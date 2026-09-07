"""Canonical metric BEV grid shared by camera and LiDAR branches."""

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class BEVGridSpec:
    """Map metric ego coordinates to a canonical ``[B, C, Y, X]`` tensor."""

    xbound: Tuple[float, float, float] = (-54.0, 54.0, 0.6)
    ybound: Tuple[float, float, float] = (-54.0, 54.0, 0.6)

    def __post_init__(self):
        for name, bound in (("xbound", self.xbound), ("ybound", self.ybound)):
            lower, upper, resolution = bound
            cells = (upper - lower) / resolution
            if lower >= upper or resolution <= 0 or abs(cells - round(cells)) > 1e-6:
                raise ValueError(f"invalid {name}: {bound}")

    @property
    def width(self) -> int:
        return round((self.xbound[1] - self.xbound[0]) / self.xbound[2])

    @property
    def height(self) -> int:
        return round((self.ybound[1] - self.ybound[0]) / self.ybound[2])

    def metric_to_grid(self, xy: torch.Tensor) -> torch.Tensor:
        """Return integer ``[..., (row_y, column_x)]`` indices."""
        if xy.shape[-1] != 2:
            raise ValueError(f"expected [...,2] xy coordinates, got {tuple(xy.shape)}")
        column = torch.floor((xy[..., 0] - self.xbound[0]) / self.xbound[2])
        row = torch.floor((xy[..., 1] - self.ybound[0]) / self.ybound[2])
        return torch.stack((row, column), dim=-1).long()

    def centers(self, device=None, dtype=torch.float32):
        """Return metric X/Y center meshes in canonical ``[Y, X]`` layout."""
        x = torch.arange(self.width, device=device, dtype=dtype)
        y = torch.arange(self.height, device=device, dtype=dtype)
        x = self.xbound[0] + (x + 0.5) * self.xbound[2]
        y = self.ybound[0] + (y + 0.5) * self.ybound[2]
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return xx, yy

    def radial_distance(self, device=None, dtype=torch.float32) -> torch.Tensor:
        xx, yy = self.centers(device=device, dtype=dtype)
        return torch.sqrt(xx.square() + yy.square())

    def canonicalize(self, bev: torch.Tensor, layout: str) -> torch.Tensor:
        """Convert ``yx`` or ``xy`` spatial layout to canonical ``yx``."""
        if bev.ndim != 4:
            raise ValueError(f"expected [B,C,H,W] BEV tensor, got {tuple(bev.shape)}")
        if layout == "yx":
            output = bev
        elif layout == "xy":
            output = bev.transpose(-1, -2).contiguous()
        else:
            raise ValueError(f"unsupported BEV layout: {layout}")
        if output.shape[-2:] != (self.height, self.width):
            raise ValueError(
                f"expected canonical BEV {(self.height, self.width)}, "
                f"got {tuple(output.shape[-2:])}"
            )
        return output


__all__ = ["BEVGridSpec"]
