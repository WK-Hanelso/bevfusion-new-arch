"""CPU validation for the official DSVT LiDAR checkpoint layout."""

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[3]
CORE_PATH = ROOT / "mmdet3d" / "models" / "backbones" / "dsvt_core.py"
OFFICIAL_ROOT = Path(
    "/tmp/claude-1000/-home-hanelso-hanelso/"
    "8edcff05-d561-4bd4-9120-e7bcf8ad837d/scratchpad/dsvt_ckpt/official"
)
CONVERTED = ROOT / "pretrained" / "dsvt_nuscenes_official_lidar.pth"
REPORT = ROOT / "pretrained" / "dsvt_nuscenes_official_lidar.report.json"
PREFIX = "encoders.lidar.backbone."

# Frozen from main before adding official_layout. This deliberately detects key
# additions, removals, renames, and shape changes in the default path.
LEGACY_STATE_SNAPSHOT = [
  [
    "vfe.voxel_size",
    [
      3
    ]
  ],
  [
    "vfe.point_cloud_range",
    [
      6
    ]
  ],
  [
    "vfe.grid_size",
    [
      3
    ]
  ],
  [
    "vfe.layers.0.linear.weight",
    [
      64,
      11
    ]
  ],
  [
    "vfe.layers.0.norm.weight",
    [
      64
    ]
  ],
  [
    "vfe.layers.0.norm.bias",
    [
      64
    ]
  ],
  [
    "vfe.layers.0.norm.running_mean",
    [
      64
    ]
  ],
  [
    "vfe.layers.0.norm.running_var",
    [
      64
    ]
  ],
  [
    "vfe.layers.0.norm.num_batches_tracked",
    []
  ],
  [
    "vfe.layers.1.linear.weight",
    [
      128,
      128
    ]
  ],
  [
    "vfe.layers.1.norm.weight",
    [
      128
    ]
  ],
  [
    "vfe.layers.1.norm.bias",
    [
      128
    ]
  ],
  [
    "vfe.layers.1.norm.running_mean",
    [
      128
    ]
  ],
  [
    "vfe.layers.1.norm.running_var",
    [
      128
    ]
  ],
  [
    "vfe.layers.1.norm.num_batches_tracked",
    []
  ],
  [
    "backbone.input_layer.position_layers.0.0.layers.0.weight",
    [
      128,
      2
    ]
  ],
  [
    "backbone.input_layer.position_layers.0.0.layers.0.bias",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.0.0.layers.1.weight",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.0.0.layers.1.bias",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.0.0.layers.1.running_mean",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.0.0.layers.1.running_var",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.0.0.layers.1.num_batches_tracked",
    []
  ],
  [
    "backbone.input_layer.position_layers.0.0.layers.3.weight",
    [
      128,
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.0.0.layers.3.bias",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.0.1.layers.0.weight",
    [
      128,
      2
    ]
  ],
  [
    "backbone.input_layer.position_layers.0.1.layers.0.bias",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.0.1.layers.1.weight",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.0.1.layers.1.bias",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.0.1.layers.1.running_mean",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.0.1.layers.1.running_var",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.0.1.layers.1.num_batches_tracked",
    []
  ],
  [
    "backbone.input_layer.position_layers.0.1.layers.3.weight",
    [
      128,
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.0.1.layers.3.bias",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.1.0.layers.0.weight",
    [
      128,
      2
    ]
  ],
  [
    "backbone.input_layer.position_layers.1.0.layers.0.bias",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.1.0.layers.1.weight",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.1.0.layers.1.bias",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.1.0.layers.1.running_mean",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.1.0.layers.1.running_var",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.1.0.layers.1.num_batches_tracked",
    []
  ],
  [
    "backbone.input_layer.position_layers.1.0.layers.3.weight",
    [
      128,
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.1.0.layers.3.bias",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.1.1.layers.0.weight",
    [
      128,
      2
    ]
  ],
  [
    "backbone.input_layer.position_layers.1.1.layers.0.bias",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.1.1.layers.1.weight",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.1.1.layers.1.bias",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.1.1.layers.1.running_mean",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.1.1.layers.1.running_var",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.1.1.layers.1.num_batches_tracked",
    []
  ],
  [
    "backbone.input_layer.position_layers.1.1.layers.3.weight",
    [
      128,
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.1.1.layers.3.bias",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.2.0.layers.0.weight",
    [
      128,
      2
    ]
  ],
  [
    "backbone.input_layer.position_layers.2.0.layers.0.bias",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.2.0.layers.1.weight",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.2.0.layers.1.bias",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.2.0.layers.1.running_mean",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.2.0.layers.1.running_var",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.2.0.layers.1.num_batches_tracked",
    []
  ],
  [
    "backbone.input_layer.position_layers.2.0.layers.3.weight",
    [
      128,
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.2.0.layers.3.bias",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.2.1.layers.0.weight",
    [
      128,
      2
    ]
  ],
  [
    "backbone.input_layer.position_layers.2.1.layers.0.bias",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.2.1.layers.1.weight",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.2.1.layers.1.bias",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.2.1.layers.1.running_mean",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.2.1.layers.1.running_var",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.2.1.layers.1.num_batches_tracked",
    []
  ],
  [
    "backbone.input_layer.position_layers.2.1.layers.3.weight",
    [
      128,
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.2.1.layers.3.bias",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.3.0.layers.0.weight",
    [
      128,
      2
    ]
  ],
  [
    "backbone.input_layer.position_layers.3.0.layers.0.bias",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.3.0.layers.1.weight",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.3.0.layers.1.bias",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.3.0.layers.1.running_mean",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.3.0.layers.1.running_var",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.3.0.layers.1.num_batches_tracked",
    []
  ],
  [
    "backbone.input_layer.position_layers.3.0.layers.3.weight",
    [
      128,
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.3.0.layers.3.bias",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.3.1.layers.0.weight",
    [
      128,
      2
    ]
  ],
  [
    "backbone.input_layer.position_layers.3.1.layers.0.bias",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.3.1.layers.1.weight",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.3.1.layers.1.bias",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.3.1.layers.1.running_mean",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.3.1.layers.1.running_var",
    [
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.3.1.layers.1.num_batches_tracked",
    []
  ],
  [
    "backbone.input_layer.position_layers.3.1.layers.3.weight",
    [
      128,
      128
    ]
  ],
  [
    "backbone.input_layer.position_layers.3.1.layers.3.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.0.layers.0.attention.in_proj_weight",
    [
      384,
      128
    ]
  ],
  [
    "backbone.blocks.0.layers.0.attention.in_proj_bias",
    [
      384
    ]
  ],
  [
    "backbone.blocks.0.layers.0.attention.out_proj.weight",
    [
      128,
      128
    ]
  ],
  [
    "backbone.blocks.0.layers.0.attention.out_proj.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.0.layers.0.linear1.weight",
    [
      256,
      128
    ]
  ],
  [
    "backbone.blocks.0.layers.0.linear1.bias",
    [
      256
    ]
  ],
  [
    "backbone.blocks.0.layers.0.linear2.weight",
    [
      128,
      256
    ]
  ],
  [
    "backbone.blocks.0.layers.0.linear2.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.0.layers.0.norm1.weight",
    [
      128
    ]
  ],
  [
    "backbone.blocks.0.layers.0.norm1.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.0.layers.0.norm2.weight",
    [
      128
    ]
  ],
  [
    "backbone.blocks.0.layers.0.norm2.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.0.layers.1.attention.in_proj_weight",
    [
      384,
      128
    ]
  ],
  [
    "backbone.blocks.0.layers.1.attention.in_proj_bias",
    [
      384
    ]
  ],
  [
    "backbone.blocks.0.layers.1.attention.out_proj.weight",
    [
      128,
      128
    ]
  ],
  [
    "backbone.blocks.0.layers.1.attention.out_proj.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.0.layers.1.linear1.weight",
    [
      256,
      128
    ]
  ],
  [
    "backbone.blocks.0.layers.1.linear1.bias",
    [
      256
    ]
  ],
  [
    "backbone.blocks.0.layers.1.linear2.weight",
    [
      128,
      256
    ]
  ],
  [
    "backbone.blocks.0.layers.1.linear2.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.0.layers.1.norm1.weight",
    [
      128
    ]
  ],
  [
    "backbone.blocks.0.layers.1.norm1.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.0.layers.1.norm2.weight",
    [
      128
    ]
  ],
  [
    "backbone.blocks.0.layers.1.norm2.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.1.layers.0.attention.in_proj_weight",
    [
      384,
      128
    ]
  ],
  [
    "backbone.blocks.1.layers.0.attention.in_proj_bias",
    [
      384
    ]
  ],
  [
    "backbone.blocks.1.layers.0.attention.out_proj.weight",
    [
      128,
      128
    ]
  ],
  [
    "backbone.blocks.1.layers.0.attention.out_proj.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.1.layers.0.linear1.weight",
    [
      256,
      128
    ]
  ],
  [
    "backbone.blocks.1.layers.0.linear1.bias",
    [
      256
    ]
  ],
  [
    "backbone.blocks.1.layers.0.linear2.weight",
    [
      128,
      256
    ]
  ],
  [
    "backbone.blocks.1.layers.0.linear2.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.1.layers.0.norm1.weight",
    [
      128
    ]
  ],
  [
    "backbone.blocks.1.layers.0.norm1.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.1.layers.0.norm2.weight",
    [
      128
    ]
  ],
  [
    "backbone.blocks.1.layers.0.norm2.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.1.layers.1.attention.in_proj_weight",
    [
      384,
      128
    ]
  ],
  [
    "backbone.blocks.1.layers.1.attention.in_proj_bias",
    [
      384
    ]
  ],
  [
    "backbone.blocks.1.layers.1.attention.out_proj.weight",
    [
      128,
      128
    ]
  ],
  [
    "backbone.blocks.1.layers.1.attention.out_proj.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.1.layers.1.linear1.weight",
    [
      256,
      128
    ]
  ],
  [
    "backbone.blocks.1.layers.1.linear1.bias",
    [
      256
    ]
  ],
  [
    "backbone.blocks.1.layers.1.linear2.weight",
    [
      128,
      256
    ]
  ],
  [
    "backbone.blocks.1.layers.1.linear2.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.1.layers.1.norm1.weight",
    [
      128
    ]
  ],
  [
    "backbone.blocks.1.layers.1.norm1.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.1.layers.1.norm2.weight",
    [
      128
    ]
  ],
  [
    "backbone.blocks.1.layers.1.norm2.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.2.layers.0.attention.in_proj_weight",
    [
      384,
      128
    ]
  ],
  [
    "backbone.blocks.2.layers.0.attention.in_proj_bias",
    [
      384
    ]
  ],
  [
    "backbone.blocks.2.layers.0.attention.out_proj.weight",
    [
      128,
      128
    ]
  ],
  [
    "backbone.blocks.2.layers.0.attention.out_proj.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.2.layers.0.linear1.weight",
    [
      256,
      128
    ]
  ],
  [
    "backbone.blocks.2.layers.0.linear1.bias",
    [
      256
    ]
  ],
  [
    "backbone.blocks.2.layers.0.linear2.weight",
    [
      128,
      256
    ]
  ],
  [
    "backbone.blocks.2.layers.0.linear2.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.2.layers.0.norm1.weight",
    [
      128
    ]
  ],
  [
    "backbone.blocks.2.layers.0.norm1.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.2.layers.0.norm2.weight",
    [
      128
    ]
  ],
  [
    "backbone.blocks.2.layers.0.norm2.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.2.layers.1.attention.in_proj_weight",
    [
      384,
      128
    ]
  ],
  [
    "backbone.blocks.2.layers.1.attention.in_proj_bias",
    [
      384
    ]
  ],
  [
    "backbone.blocks.2.layers.1.attention.out_proj.weight",
    [
      128,
      128
    ]
  ],
  [
    "backbone.blocks.2.layers.1.attention.out_proj.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.2.layers.1.linear1.weight",
    [
      256,
      128
    ]
  ],
  [
    "backbone.blocks.2.layers.1.linear1.bias",
    [
      256
    ]
  ],
  [
    "backbone.blocks.2.layers.1.linear2.weight",
    [
      128,
      256
    ]
  ],
  [
    "backbone.blocks.2.layers.1.linear2.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.2.layers.1.norm1.weight",
    [
      128
    ]
  ],
  [
    "backbone.blocks.2.layers.1.norm1.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.2.layers.1.norm2.weight",
    [
      128
    ]
  ],
  [
    "backbone.blocks.2.layers.1.norm2.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.3.layers.0.attention.in_proj_weight",
    [
      384,
      128
    ]
  ],
  [
    "backbone.blocks.3.layers.0.attention.in_proj_bias",
    [
      384
    ]
  ],
  [
    "backbone.blocks.3.layers.0.attention.out_proj.weight",
    [
      128,
      128
    ]
  ],
  [
    "backbone.blocks.3.layers.0.attention.out_proj.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.3.layers.0.linear1.weight",
    [
      256,
      128
    ]
  ],
  [
    "backbone.blocks.3.layers.0.linear1.bias",
    [
      256
    ]
  ],
  [
    "backbone.blocks.3.layers.0.linear2.weight",
    [
      128,
      256
    ]
  ],
  [
    "backbone.blocks.3.layers.0.linear2.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.3.layers.0.norm1.weight",
    [
      128
    ]
  ],
  [
    "backbone.blocks.3.layers.0.norm1.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.3.layers.0.norm2.weight",
    [
      128
    ]
  ],
  [
    "backbone.blocks.3.layers.0.norm2.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.3.layers.1.attention.in_proj_weight",
    [
      384,
      128
    ]
  ],
  [
    "backbone.blocks.3.layers.1.attention.in_proj_bias",
    [
      384
    ]
  ],
  [
    "backbone.blocks.3.layers.1.attention.out_proj.weight",
    [
      128,
      128
    ]
  ],
  [
    "backbone.blocks.3.layers.1.attention.out_proj.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.3.layers.1.linear1.weight",
    [
      256,
      128
    ]
  ],
  [
    "backbone.blocks.3.layers.1.linear1.bias",
    [
      256
    ]
  ],
  [
    "backbone.blocks.3.layers.1.linear2.weight",
    [
      128,
      256
    ]
  ],
  [
    "backbone.blocks.3.layers.1.linear2.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.3.layers.1.norm1.weight",
    [
      128
    ]
  ],
  [
    "backbone.blocks.3.layers.1.norm1.bias",
    [
      128
    ]
  ],
  [
    "backbone.blocks.3.layers.1.norm2.weight",
    [
      128
    ]
  ],
  [
    "backbone.blocks.3.layers.1.norm2.bias",
    [
      128
    ]
  ],
  [
    "backbone.residual_norms.0.weight",
    [
      128
    ]
  ],
  [
    "backbone.residual_norms.0.bias",
    [
      128
    ]
  ],
  [
    "backbone.residual_norms.1.weight",
    [
      128
    ]
  ],
  [
    "backbone.residual_norms.1.bias",
    [
      128
    ]
  ],
  [
    "backbone.residual_norms.2.weight",
    [
      128
    ]
  ],
  [
    "backbone.residual_norms.2.bias",
    [
      128
    ]
  ],
  [
    "backbone.residual_norms.3.weight",
    [
      128
    ]
  ],
  [
    "backbone.residual_norms.3.bias",
    [
      128
    ]
  ],
  [
    "neck.blocks.0.1.weight",
    [
      128,
      128,
      3,
      3
    ]
  ],
  [
    "neck.blocks.0.2.weight",
    [
      128
    ]
  ],
  [
    "neck.blocks.0.2.bias",
    [
      128
    ]
  ],
  [
    "neck.blocks.0.2.running_mean",
    [
      128
    ]
  ],
  [
    "neck.blocks.0.2.running_var",
    [
      128
    ]
  ],
  [
    "neck.blocks.0.2.num_batches_tracked",
    []
  ],
  [
    "neck.blocks.0.4.weight",
    [
      128,
      128,
      3,
      3
    ]
  ],
  [
    "neck.blocks.0.5.weight",
    [
      128
    ]
  ],
  [
    "neck.blocks.0.5.bias",
    [
      128
    ]
  ],
  [
    "neck.blocks.0.5.running_mean",
    [
      128
    ]
  ],
  [
    "neck.blocks.0.5.running_var",
    [
      128
    ]
  ],
  [
    "neck.blocks.0.5.num_batches_tracked",
    []
  ],
  [
    "neck.blocks.1.1.weight",
    [
      128,
      128,
      3,
      3
    ]
  ],
  [
    "neck.blocks.1.2.weight",
    [
      128
    ]
  ],
  [
    "neck.blocks.1.2.bias",
    [
      128
    ]
  ],
  [
    "neck.blocks.1.2.running_mean",
    [
      128
    ]
  ],
  [
    "neck.blocks.1.2.running_var",
    [
      128
    ]
  ],
  [
    "neck.blocks.1.2.num_batches_tracked",
    []
  ],
  [
    "neck.blocks.1.4.weight",
    [
      128,
      128,
      3,
      3
    ]
  ],
  [
    "neck.blocks.1.5.weight",
    [
      128
    ]
  ],
  [
    "neck.blocks.1.5.bias",
    [
      128
    ]
  ],
  [
    "neck.blocks.1.5.running_mean",
    [
      128
    ]
  ],
  [
    "neck.blocks.1.5.running_var",
    [
      128
    ]
  ],
  [
    "neck.blocks.1.5.num_batches_tracked",
    []
  ],
  [
    "neck.blocks.1.7.weight",
    [
      128,
      128,
      3,
      3
    ]
  ],
  [
    "neck.blocks.1.8.weight",
    [
      128
    ]
  ],
  [
    "neck.blocks.1.8.bias",
    [
      128
    ]
  ],
  [
    "neck.blocks.1.8.running_mean",
    [
      128
    ]
  ],
  [
    "neck.blocks.1.8.running_var",
    [
      128
    ]
  ],
  [
    "neck.blocks.1.8.num_batches_tracked",
    []
  ],
  [
    "neck.blocks.2.1.weight",
    [
      256,
      128,
      3,
      3
    ]
  ],
  [
    "neck.blocks.2.2.weight",
    [
      256
    ]
  ],
  [
    "neck.blocks.2.2.bias",
    [
      256
    ]
  ],
  [
    "neck.blocks.2.2.running_mean",
    [
      256
    ]
  ],
  [
    "neck.blocks.2.2.running_var",
    [
      256
    ]
  ],
  [
    "neck.blocks.2.2.num_batches_tracked",
    []
  ],
  [
    "neck.blocks.2.4.weight",
    [
      256,
      256,
      3,
      3
    ]
  ],
  [
    "neck.blocks.2.5.weight",
    [
      256
    ]
  ],
  [
    "neck.blocks.2.5.bias",
    [
      256
    ]
  ],
  [
    "neck.blocks.2.5.running_mean",
    [
      256
    ]
  ],
  [
    "neck.blocks.2.5.running_var",
    [
      256
    ]
  ],
  [
    "neck.blocks.2.5.num_batches_tracked",
    []
  ],
  [
    "neck.blocks.2.7.weight",
    [
      256,
      256,
      3,
      3
    ]
  ],
  [
    "neck.blocks.2.8.weight",
    [
      256
    ]
  ],
  [
    "neck.blocks.2.8.bias",
    [
      256
    ]
  ],
  [
    "neck.blocks.2.8.running_mean",
    [
      256
    ]
  ],
  [
    "neck.blocks.2.8.running_var",
    [
      256
    ]
  ],
  [
    "neck.blocks.2.8.num_batches_tracked",
    []
  ],
  [
    "neck.deblocks.0.0.weight",
    [
      128,
      128,
      2,
      2
    ]
  ],
  [
    "neck.deblocks.0.1.weight",
    [
      128
    ]
  ],
  [
    "neck.deblocks.0.1.bias",
    [
      128
    ]
  ],
  [
    "neck.deblocks.0.1.running_mean",
    [
      128
    ]
  ],
  [
    "neck.deblocks.0.1.running_var",
    [
      128
    ]
  ],
  [
    "neck.deblocks.0.1.num_batches_tracked",
    []
  ],
  [
    "neck.deblocks.1.0.weight",
    [
      128,
      128,
      1,
      1
    ]
  ],
  [
    "neck.deblocks.1.1.weight",
    [
      128
    ]
  ],
  [
    "neck.deblocks.1.1.bias",
    [
      128
    ]
  ],
  [
    "neck.deblocks.1.1.running_mean",
    [
      128
    ]
  ],
  [
    "neck.deblocks.1.1.running_var",
    [
      128
    ]
  ],
  [
    "neck.deblocks.1.1.num_batches_tracked",
    []
  ],
  [
    "neck.deblocks.2.0.weight",
    [
      256,
      128,
      2,
      2
    ]
  ],
  [
    "neck.deblocks.2.1.weight",
    [
      128
    ]
  ],
  [
    "neck.deblocks.2.1.bias",
    [
      128
    ]
  ],
  [
    "neck.deblocks.2.1.running_mean",
    [
      128
    ]
  ],
  [
    "neck.deblocks.2.1.running_var",
    [
      128
    ]
  ],
  [
    "neck.deblocks.2.1.num_batches_tracked",
    []
  ],
  [
    "neck.adapter.0.weight",
    [
      256,
      384,
      1,
      1
    ]
  ],
  [
    "neck.adapter.1.weight",
    [
      256
    ]
  ],
  [
    "neck.adapter.1.bias",
    [
      256
    ]
  ],
  [
    "neck.adapter.1.running_mean",
    [
      256
    ]
  ],
  [
    "neck.adapter.1.running_var",
    [
      256
    ]
  ],
  [
    "neck.adapter.1.num_batches_tracked",
    []
  ]
]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORE = _load_module("tested_dsvt_core", CORE_PATH)


def _load_main_core():
    source = subprocess.check_output(
        ["git", "show", "main:mmdet3d/models/backbones/dsvt_core.py"],
        cwd=str(ROOT),
    )
    module = types.ModuleType("main_dsvt_core")
    exec(compile(source, str(CORE_PATH), "exec"), module.__dict__)
    return module


def _synthetic_points(count=72):
    generator = torch.Generator().manual_seed(19)
    points = torch.rand((count, 5), generator=generator)
    points[:, 0:2] = points[:, 0:2] * 80.0 - 40.0
    points[:, 2] = points[:, 2] * 6.0 - 4.0
    points[:, 3] *= 0.9
    points[:, 4] *= 31.0
    return [points]


def _install_official_dsvt_stubs():
    names = (
        "_official_dsvt",
        "_official_dsvt.backbones",
        "_official_dsvt.model_utils",
        "_official_dsvt.model_utils.tensorrt_utils",
    )
    for name in names:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module

    input_layer = types.ModuleType("_official_dsvt.backbones.dsvt_input_layer")
    input_layer.DSVTInputLayer = type("DSVTInputLayer", (), {})
    sys.modules[input_layer.__name__] = input_layer

    trt = types.ModuleType(
        "_official_dsvt.model_utils.tensorrt_utils.trtwrapper"
    )
    trt.TRTWrapper = type("TRTWrapper", (nn.Module,), {})
    sys.modules[trt.__name__] = trt


def _load_official_dsvt():
    _install_official_dsvt_stubs()
    return _load_module(
        "_official_dsvt.backbones.dsvt", OFFICIAL_ROOT / "dsvt.py"
    )


class _Config(dict):
    __getattr__ = dict.__getitem__


def _official_bev_config():
    return _Config(
        LAYER_NUMS=[1, 2, 2],
        LAYER_STRIDES=[1, 2, 2],
        NUM_FILTERS=[128, 128, 256],
        UPSAMPLE_STRIDES=[0.5, 1, 2],
        NUM_UPSAMPLE_FILTERS=[128, 128, 128],
    )


def _max_diff(left, right):
    return float((left - right).abs().max())


def test_legacy_state_dict_snapshot():
    encoder = CORE.DSVTLidarEncoder(official_layout=False)
    actual = [[key, list(value.shape)] for key, value in encoder.state_dict().items()]
    assert actual == LEGACY_STATE_SNAPSHOT


def test_legacy_forward_bitwise_equal():
    main_core = _load_main_core()
    torch.manual_seed(23)
    reference = main_core.DSVTLidarEncoder().eval()
    candidate = CORE.DSVTLidarEncoder(official_layout=False).eval()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    points = _synthetic_points()
    with torch.no_grad():
        expected = reference(points)
        actual = candidate(points)
    print(f"legacy_forward_max_abs_diff={_max_diff(expected, actual):.9g}")
    assert torch.equal(actual, expected)


def test_zero_feature_channel_matches_explicit_zero():
    torch.manual_seed(29)
    zeroing = CORE.DynamicPillarVFE(zero_feature_channels=[4]).eval()
    explicit = CORE.DynamicPillarVFE().eval()
    explicit.load_state_dict(zeroing.state_dict(), strict=True)
    points = _synthetic_points(48)
    zeroed_points = [points[0].clone()]
    zeroed_points[0][:, 4] = 0
    with torch.no_grad():
        actual = zeroing(points)
        expected = explicit(zeroed_points)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert actual[2] == expected[2]


def test_converted_checkpoint_loads_with_only_adapter_missing():
    checkpoint = torch.load(str(CONVERTED), map_location="cpu")
    state = {
        key[len(PREFIX) :]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith(PREFIX)
    }
    encoder = CORE.DSVTLidarEncoder(
        official_layout=True, zero_feature_channels=[4]
    )
    incompatible = encoder.load_state_dict(state, strict=False)
    expected_missing = sorted(
        key
        for key in encoder.state_dict()
        if key.startswith("neck.adapter.")
        and not key.endswith("num_batches_tracked")
    )
    assert sorted(incompatible.missing_keys) == expected_missing
    assert incompatible.unexpected_keys == []

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["counts"] == {
        "initialized": 3,
        "matched": 336,
        "missing": 6,
        "unexpected": 113,
    }
    assert report["tensor_elements"]["matched_official"] == 6_232_032
    print(
        "load_missing=5(all_adapter; BN tracker omitted by torch) "
        "load_unexpected=0 report_missing=6 matched=336 "
        "matched_elements=6232032"
    )


def test_official_set_attention_and_encoder_layer_equivalence():
    official = _load_official_dsvt()
    torch.manual_seed(31)
    reference = official.SetAttention(
        8, 2, 0.0, dim_feedforward=16, activation="gelu",
        batch_first=True, mlp_dropout=0,
    ).eval()
    candidate = CORE.SetAttention(8, 2, 16, official_layout=True).eval()
    mapped = {
        key.replace("self_attn.", "attention."): value
        for key, value in reference.state_dict().items()
    }
    candidate.load_state_dict(mapped, strict=True)

    features = torch.randn(4, 8)
    positions = torch.randn(4, 8)
    indices = torch.tensor([[0, 1, 2], [2, 3, 3]])
    mask = torch.tensor([[False, False, False], [False, False, True]])
    with torch.no_grad():
        expected_attention = reference(features, positions, mask, indices)
        actual_attention = candidate(features, indices, mask, positions)
    attention_diff = _max_diff(expected_attention, actual_attention)
    print(f"set_attention_max_abs_diff={attention_diff:.9g}")
    assert attention_diff < 1e-5

    torch.manual_seed(37)
    encoder = official.DSVT_EncoderLayer(
        8, 2, 16, 0.0, "gelu", True, 0
    ).eval()
    candidate_attention = CORE.SetAttention(
        8, 2, 16, official_layout=True
    ).eval()
    candidate_norm = nn.LayerNorm(8).eval()
    attention_state = {
        key[len("win_attn.") :].replace("self_attn.", "attention."): value
        for key, value in encoder.state_dict().items()
        if key.startswith("win_attn.")
    }
    norm_state = {
        key[len("norm.") :]: value
        for key, value in encoder.state_dict().items()
        if key.startswith("norm.")
    }
    candidate_attention.load_state_dict(attention_state, strict=True)
    candidate_norm.load_state_dict(norm_state, strict=True)
    with torch.no_grad():
        expected_encoder = encoder(features, indices, mask, positions)
        actual_encoder = candidate_norm(
            candidate_attention(features, indices, mask, positions) + features
        )
    encoder_diff = _max_diff(expected_encoder, actual_encoder)
    print(f"encoder_layer_max_abs_diff={encoder_diff:.9g}")
    assert encoder_diff < 1e-5


def test_official_basic_block_and_bev_backbone_equivalence():
    official = _load_module(
        "official_base_bev_res_backbone",
        OFFICIAL_ROOT / "base_bev_res_backbone.py",
    )
    torch.manual_seed(41)
    reference_block = official.BasicBlock(4, 8, stride=2, downsample=True).eval()
    candidate_block = CORE.DSVTBEVBasicBlock(
        4, 8, stride=2, downsample=True
    ).eval()
    candidate_block.load_state_dict(reference_block.state_dict(), strict=True)
    block_input = torch.randn(2, 4, 16, 16)
    with torch.no_grad():
        expected_block = reference_block(block_input)
        actual_block = candidate_block(block_input)
    block_diff = _max_diff(expected_block, actual_block)
    print(f"basic_block_max_abs_diff={block_diff:.9g}")
    assert block_diff < 1e-5

    if not hasattr(np, "int"):
        np.int = int
    torch.manual_seed(43)
    reference = official.BaseBEVResBackbone(
        _official_bev_config(), input_channels=128
    ).eval()
    candidate = CORE.DSVTBEVResNeck().eval()
    incompatible = candidate.load_state_dict(reference.state_dict(), strict=False)
    assert sorted(incompatible.missing_keys) == sorted(
        key
        for key in candidate.state_dict()
        if key.startswith("adapter.")
        and not key.endswith("num_batches_tracked")
    )
    assert incompatible.unexpected_keys == []
    features = torch.randn(1, 128, 32, 32)
    with torch.no_grad():
        expected = reference({"spatial_features": features.clone()})[
            "spatial_features_2d"
        ]
        actual = candidate.forward_features(features)
    backbone_diff = _max_diff(expected, actual)
    print(f"bev_backbone_max_abs_diff={backbone_diff:.9g}")
    assert backbone_diff < 1e-5


def test_official_encoder_synthetic_forward():
    checkpoint = torch.load(str(CONVERTED), map_location="cpu")
    state = {
        key[len(PREFIX) :]: value
        for key, value in checkpoint["state_dict"].items()
    }
    encoder = CORE.DSVTLidarEncoder(
        official_layout=True, zero_feature_channels=[4]
    ).eval()
    encoder.load_state_dict(state, strict=False)
    with torch.no_grad():
        output = encoder(_synthetic_points(48))
    assert output.shape == (1, 256, 180, 180)
    assert torch.isfinite(output).all()
    print(f"official_encoder_output_shape={tuple(output.shape)}")
