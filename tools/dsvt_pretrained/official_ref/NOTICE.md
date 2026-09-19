# Vendored DSVT reference

This directory is a CPU-only adaptation of the following files from
Haiyang-W/DSVT:

- `pcdet/models/backbones_3d/vfe/dynamic_pillar_vfe.py`
- `pcdet/models/backbones_3d/dsvt.py`
- `pcdet/models/backbones_3d/dsvt_input_layer.py`
- `pcdet/models/model_utils/dsvt_utils.py`
- `pcdet/models/backbones_2d/map_to_bev/pointpillar3d_scatter.py`
- `pcdet/models/backbones_2d/base_bev_res_backbone.py`

Upstream: https://github.com/Haiyang-W/DSVT

Vendored from the supplied `/tmp/DSVT` checkout at revision
`8cfc2a6f23eed0b10aabcdc4768c60b184357061`.

Copyright 2023 Haiyang-W/DSVT contributors. Licensed under the Apache License,
Version 2.0. The adaptation removes OpenPCDet registry/config dependencies,
fixes the model to the published one-stage nuScenes configuration, and replaces
the `ingroup_inds` CUDA extension with an equivalent pure-PyTorch CPU routine.
