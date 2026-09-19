"""BEV layout contract tests, split between host-safe and Docker checks.

The first group only parses source/YAML (plus the locally available torch for
one tensor assertion) and never imports :mod:`mmdet3d`.  Tests guarded by
``_require_mmdet3d`` are intended for the ``bevfusion-train:cu113`` image.
"""

import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
ABLATIONS = ROOT / "configs/nuscenes/det/ablation"


def _load_source_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


layout_contract = _load_source_module(
    "bev_layout_contract", "mmdet3d/models/fusion_models/bev_layout.py"
)


def _deep_merge(base, override):
    if not isinstance(base, dict) or not isinstance(override, dict):
        return copy.deepcopy(override)
    merged = copy.deepcopy(base)
    for key, value in override.items():
        merged[key] = (
            _deep_merge(merged[key], value)
            if key in merged
            else copy.deepcopy(value)
        )
    return merged


def _class_string_attribute(relative_path, class_name, attribute):
    tree = ast.parse((ROOT / relative_path).read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for statement in node.body:
                if isinstance(statement, ast.Assign):
                    if any(
                        isinstance(target, ast.Name) and target.id == attribute
                        for target in statement.targets
                    ):
                        return ast.literal_eval(statement.value)
    raise AssertionError(f"{class_name}.{attribute} is not declared in {relative_path}")


def _init_default(relative_path, class_name, argument):
    tree = ast.parse((ROOT / relative_path).read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    init = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    names = [arg.arg for arg in init.args.args]
    defaults = [None] * (len(names) - len(init.args.defaults)) + init.args.defaults
    return ast.literal_eval(defaults[names.index(argument)])


def _resolved_ablation_models():
    default = yaml.safe_load((ABLATIONS / "default.yaml").read_text())
    leaves = sorted(ABLATIONS.glob("dsvt[01]_wf[01]_gf[01]_dal[01].yaml"))
    assert len(leaves) == 16
    for leaf in leaves:
        merged = _deep_merge(default, yaml.safe_load(leaf.read_text()))
        if isinstance(merged["model"]["encoders"]["lidar"], str):
            merged["model"]["encoders"]["lidar"] = copy.deepcopy(
                merged["lidar_legacy"]
            )
        yield leaf.name, merged["model"]


def test_all_ablation_modules_have_source_layout_contracts_and_canonical_heads():
    assert layout_contract.CANONICAL_BEV_LAYOUT == "yx"
    declarations = {
        "SparseEncoder": _class_string_attribute(
            "mmdet3d/models/backbones/sparse_encoder.py", "SparseEncoder", "bev_layout"
        ),
        "DSVTLidarEncoder": _class_string_attribute(
            "mmdet3d/models/backbones/dsvt_core.py", "DSVTLidarEncoder", "bev_layout"
        ),
        "PointPillarsScatter": _class_string_attribute(
            "mmdet3d/models/backbones/pillar_encoder.py",
            "PointPillarsScatter",
            "bev_layout",
        ),
        "PointPillarsEncoder": _class_string_attribute(
            "mmdet3d/models/backbones/pillar_encoder.py",
            "PointPillarsEncoder",
            "bev_layout",
        ),
        "DepthLSSTransform": _class_string_attribute(
            "mmdet3d/models/vtransforms/base.py", "BaseTransform", "bev_layout"
        ),
        "WidthFormerTransform": _class_string_attribute(
            "mmdet3d/models/vtransforms/widthformer.py",
            "WidthFormerTransform",
            "bev_layout",
        ),
    }
    assert declarations == {
        "SparseEncoder": "xy",
        "DSVTLidarEncoder": "yx",
        "PointPillarsScatter": "xy",
        "PointPillarsEncoder": "xy",
        "DepthLSSTransform": "xy",
        "WidthFormerTransform": "yx",
    }

    for name, model in _resolved_ablation_models():
        lidar = model["encoders"]["lidar"]
        lidar_type = lidar.get("type") or lidar["backbone"]["type"]
        camera_type = model["encoders"]["camera"]["vtransform"]["type"]
        assert declarations[lidar_type] in ("xy", "yx"), name
        assert declarations[camera_type] in ("xy", "yx"), name

        heads = layout_contract.inject_head_bev_layout(copy.deepcopy(model["heads"]))
        assert heads["object"]["bev_layout"] == "yx", name
        if model["fuser"]["type"] == "DepthGFusion":
            assert model["fuser"]["camera_layout"] == "yx", name
            assert model["fuser"]["lidar_layout"] == "yx", name


def test_legacy_and_dal_head_defaults_are_declared_in_source():
    assert _class_string_attribute(
        "mmdet3d/models/heads/bbox/transfusion.py", "TransFusionHead", "bev_layout"
    ) == "xy"
    assert _class_string_attribute(
        "mmdet3d/models/heads/bbox/dal_decoupled.py", "DALDecoupledHead", "bev_layout"
    ) == "yx"
    assert _init_default(
        "mmdet3d/models/heads/bbox/transfusion.py",
        "TransFusionHead",
        "bev_layout",
    ) == "xy"


def test_canonicalize_bev_transposes_xy_and_rejects_missing_contract():
    xy = torch.arange(6).reshape(1, 1, 2, 3)
    module = type("XYEncoder", (), {"bev_layout": "xy"})()
    actual = layout_contract.canonicalize_bev(xy, module, "test encoder")
    assert actual.shape == (1, 1, 3, 2)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, xy.transpose(-1, -2))

    with pytest.raises(ValueError, match="must declare bev_layout"):
        layout_contract.canonicalize_bev(xy, object(), "missing encoder")


def _load_head_coordinate_methods_without_mmdet3d():
    source = ROOT / "mmdet3d/models/heads/bbox/transfusion.py"
    tree = ast.parse(source.read_text())
    head_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TransFusionHead"
    )
    wanted = {"create_2D_grid", "_dense_heatmap_target"}
    methods = [
        node
        for node in head_class.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]

    def draw_heatmap_gaussian(heatmap, center, radius):
        heatmap[int(center[1]), int(center[0])] = 1.0

    namespace = {
        "torch": torch,
        "gaussian_radius": lambda dimensions, min_overlap: 0,
        "draw_heatmap_gaussian": draw_heatmap_gaussian,
    }
    module = ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    "layout, expected_shape, expected_peak, flat_index",
    [
        ("yx", (1, 6, 8), (2, 3), 2 * 8 + 3),
        ("xy", (1, 8, 6), (3, 2), 3 * 6 + 2),
    ],
)
def test_head_coordinate_source_round_trip_without_mmdet3d(
    layout, expected_shape, expected_peak, flat_index
):
    methods = _load_head_coordinate_methods_without_mmdet3d()
    head = SimpleNamespace(
        bev_layout=layout,
        num_classes=1,
        train_cfg={
            "grid_size": [8, 6, 1],
            "point_cloud_range": [0.0, 0.0, -1.0, 8.0, 6.0, 1.0],
            "voxel_size": [1.0, 1.0, 2.0],
            "out_size_factor": 1,
            "gaussian_overlap": 0.1,
            "min_radius": 0,
        },
    )
    boxes = _SyntheticBoxes(
        torch.tensor([[3.2, 2.2, 0.0, 1.0, 1.0, 1.0, 0.0]])
    )
    target = methods["_dense_heatmap_target"](
        head, boxes, torch.tensor([0]), torch.device("cpu")
    )
    assert target.shape == expected_shape
    peak = tuple(torch.nonzero(target[0] == target[0].max())[0].tolist())
    assert peak == expected_peak

    grid = methods["create_2D_grid"](head, 8, 6)
    torch.testing.assert_close(grid[0, flat_index], torch.tensor([3.5, 2.5]))


def _require_mmdet3d():
    try:
        import mmdet3d.models  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"requires the bevfusion Docker runtime: {exc}")


class _SyntheticBoxes:
    def __init__(self, tensor):
        self.tensor = tensor
        self.gravity_center = tensor[:, :3]


def _bare_head(layout):
    _require_mmdet3d()
    from mmdet3d.models.heads.bbox.transfusion import TransFusionHead

    head = TransFusionHead.__new__(TransFusionHead)
    torch.nn.Module.__init__(head)
    head.bev_layout = layout
    head.num_classes = 1
    head.train_cfg = {
        "grid_size": [8, 6, 1],
        "point_cloud_range": [0.0, 0.0, -1.0, 8.0, 6.0, 1.0],
        "voxel_size": [1.0, 1.0, 2.0],
        "out_size_factor": 1,
        "gaussian_overlap": 0.1,
        "min_radius": 0,
    }
    return head


@pytest.mark.parametrize(
    "layout, expected_peak, flat_index",
    [("yx", (2, 3), 2 * 8 + 3), ("xy", (3, 2), 3 * 6 + 2)],
)
def test_head_grid_and_dense_target_round_trip(layout, expected_peak, flat_index):
    head = _bare_head(layout)
    boxes = _SyntheticBoxes(
        torch.tensor([[3.2, 2.2, 0.0, 1.0, 1.0, 1.0, 0.0]])
    )
    target = head._dense_heatmap_target(boxes, torch.tensor([0]), torch.device("cpu"))
    peak = tuple(torch.nonzero(target[0] == target[0].max())[0].tolist())
    assert peak == expected_peak

    grid = head.create_2D_grid(8, 6)
    torch.testing.assert_close(grid[0, flat_index], torch.tensor([3.5, 2.5]))


def test_modular_forward_transposes_encoder_output_before_decoder():
    _require_mmdet3d()
    from torch import nn
    from mmdet3d.models.fusion_models.modular_bevfusion import ModularBEVFusion

    class XYEncoder(nn.Module):
        bev_layout = "xy"

        def forward(self, points):
            return torch.arange(6).reshape(1, 1, 2, 3).float()

    class Capture(nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None

        def forward(self, value):
            self.seen = value
            return value

    model = ModularBEVFusion.__new__(ModularBEVFusion)
    nn.Module.__init__(model)
    model._raw_lidar_encoder = True
    model.encoders = nn.ModuleDict(
        {"lidar": nn.ModuleDict({"backbone": XYEncoder()})}
    )
    capture = Capture()
    model.decoder = nn.ModuleDict({"backbone": capture, "neck": nn.Identity()})
    model.heads = nn.ModuleDict()
    model.fuser = None
    model.use_depth_loss = False
    model.eval()

    model.forward_single(
        img=None,
        points=[torch.empty(0, 5)],
        camera2ego=None,
        lidar2ego=None,
        lidar2camera=None,
        lidar2image=None,
        camera_intrinsics=None,
        camera2lidar=None,
        img_aug_matrix=None,
        lidar_aug_matrix=None,
        metas=[{}],
    )
    assert capture.seen.shape == (1, 1, 3, 2)
    assert capture.seen.is_contiguous()


def _resolve_config_refs(value, root):
    if isinstance(value, dict):
        return {key: _resolve_config_refs(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_config_refs(item, root) for item in value]
    refs = {
        "${point_cloud_range}": root["point_cloud_range"],
        "${voxel_size}": root["voxel_size"],
        "${voxel_size[:2]}": root["voxel_size"][:2],
        "${point_cloud_range[:2]}": root["point_cloud_range"][:2],
    }
    return copy.deepcopy(refs.get(value, value))


def test_a1_dsvt_transfusion_synthetic_loss_is_finite():
    """Docker-only CPU integration: A1's DSVT -> decoder -> head -> loss."""
    _require_mmdet3d()
    from mmcv import ConfigDict
    from mmdet.core.bbox.assigners import AssignResult
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    from mmdet3d.models.builder import build_backbone, build_head, build_neck

    default = yaml.safe_load((ABLATIONS / "default.yaml").read_text())
    leaf = yaml.safe_load((ABLATIONS / "dsvt1_wf0_gf0_dal0.yaml").read_text())
    merged = _deep_merge(default, leaf)
    model_cfg = _resolve_config_refs(merged["model"], merged)

    encoder = build_backbone(model_cfg["encoders"]["lidar"]).eval()
    decoder = build_backbone(model_cfg["decoder"]["backbone"]).eval()
    neck = build_neck(model_cfg["decoder"]["neck"]).eval()
    head_cfg = model_cfg["heads"]["object"]
    head_cfg["bev_layout"] = "yx"
    head_cfg["train_cfg"] = ConfigDict(head_cfg["train_cfg"])
    head_cfg["test_cfg"] = ConfigDict(head_cfg["test_cfg"])
    head = build_head(head_cfg).eval()

    class CPUAssigner:
        """Exercise target/loss wiring without the CUDA-only 3D IoU op."""

        def assign(self, bboxes, gt_bboxes, gt_labels, cls_pred, train_cfg):
            assigned = bboxes.new_zeros(len(bboxes), dtype=torch.long)
            labels = bboxes.new_full((len(bboxes),), -1, dtype=torch.long)
            overlaps = bboxes.new_zeros(len(bboxes))
            if len(gt_bboxes):
                assigned[0] = 1
                labels[0] = gt_labels[0]
            return AssignResult(len(gt_bboxes), assigned, overlaps, labels=labels)

    head.bbox_assigner = CPUAssigner()

    generator = torch.Generator().manual_seed(17)
    points = torch.rand((256, 5), generator=generator)
    points[:, 0:2] = points[:, 0:2] * 100.0 - 50.0
    points[:, 2] = points[:, 2] * 6.0 - 4.0
    points[:, 3] = 1.0
    points[:, 4] = 0.0

    with torch.no_grad():
        lidar_bev = encoder([points])
        decoded = neck(decoder(lidar_bev))
        predictions = head(decoded, [{}])

    boxes = LiDARInstance3DBoxes(
        torch.tensor([[3.0, 2.0, 0.0, 2.0, 4.0, 1.5, 0.0, 0.0, 0.0]]),
        box_dim=9,
    )
    losses = head.loss([boxes], [torch.tensor([0])], predictions)
    assert losses
    for name, value in losses.items():
        assert torch.isfinite(value).all(), name
