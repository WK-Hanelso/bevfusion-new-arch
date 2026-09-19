from mmcv.cnn import build_conv_layer, build_norm_layer
from torch import nn

from mmdet3d.ops import spconv
from mmdet3d.ops.spconv_compat import (
    get_spconv_module,
    register_legacy_weight_load_hook,
    resolve_spconv_backend,
)
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck


class SparseBottleneck(Bottleneck, spconv.SparseModule):
    """Sparse bottleneck block for PartA^2.

    Bottleneck block implemented with submanifold sparse convolution.

    Args:
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        stride (int): stride of the first block. Default: 1
        downsample (None | Module): down sample module for block.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, conv_cfg=None, norm_cfg=None):

        spconv.SparseModule.__init__(self)
        Bottleneck.__init__(
            self,
            inplanes,
            planes,
            stride=stride,
            downsample=downsample,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
        )

    def forward(self, x):
        identity = x.features

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)
        out.features = self.relu(out.features)

        out = self.conv3(out)
        out.features = self.bn3(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity
        out.features = self.relu(out.features)

        return out


class SparseBasicBlock(BasicBlock, spconv.SparseModule):
    """Sparse basic block for PartA^2.

    Sparse basic block implemented with submanifold sparse convolution.

    Args:
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        stride (int): stride of the first block. Default: 1
        downsample (None | Module): down sample module for block.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, conv_cfg=None, norm_cfg=None):
        spconv.SparseModule.__init__(self)
        BasicBlock.__init__(
            self,
            inplanes,
            planes,
            stride=stride,
            downsample=downsample,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
        )

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, f"x.features.dim()={x.features.dim()}"

        out = self.conv1(x)
        out.features = self.norm1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.norm2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity
        out.features = self.relu(out.features)

        return out


def make_sparse_convmodule(
    in_channels,
    out_channels,
    kernel_size,
    indice_key,
    stride=1,
    padding=0,
    conv_type="SubMConv3d",
    norm_cfg=None,
    order=("conv", "norm", "act"),
    spconv_backend=None,
):
    """Make sparse convolution module.

    Args:
        in_channels (int): the number of input channels
        out_channels (int): the number of out channels
        kernel_size (int|tuple(int)): kernel size of convolution
        indice_key (str): the indice key used for sparse tensor
        stride (int|tuple(int)): the stride of convolution
        padding (int or list[int]): the padding number of input
        conv_type (str): sparse conv type in spconv
        norm_cfg (dict[str]): config of normalization layer
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
        spconv_backend (str, optional): ``legacy`` or ``v2``. If omitted,
            ``BEVFUSION_SPCONV`` is consulted and then defaults to ``legacy``.

    Returns:
        spconv.SparseSequential: sparse convolution module.
    """
    assert isinstance(order, tuple) and len(order) <= 3
    assert set(order) | {"conv", "norm", "act"} == {"conv", "norm", "act"}

    backend = resolve_spconv_backend(spconv_backend)
    sparse_ops = get_spconv_module(backend)
    conv_cfg = dict(type=conv_type, indice_key=indice_key)

    layers = list()
    for layer in order:
        if layer == "conv":
            inverse = conv_type in [
                "SparseInverseConv3d",
                "SparseInverseConv2d",
                "SparseInverseConv1d",
            ]
            if backend == "legacy":
                if not inverse:
                    conv = build_conv_layer(
                        conv_cfg,
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False,
                    )
                else:
                    conv = build_conv_layer(
                        conv_cfg, in_channels, out_channels, kernel_size, bias=False
                    )
            else:
                conv_cls = getattr(sparse_ops, conv_type)
                kwargs = dict(bias=False, indice_key=indice_key)
                if not inverse:
                    kwargs.update(stride=stride, padding=padding)
                conv = register_legacy_weight_load_hook(
                    conv_cls(in_channels, out_channels, kernel_size, **kwargs)
                )
            layers.append(conv)
        elif layer == "norm":
            layers.append(build_norm_layer(norm_cfg, out_channels)[1])
        elif layer == "act":
            layers.append(nn.ReLU(inplace=True))

    layers = sparse_ops.SparseSequential(*layers)
    return layers


def make_sparse_basic_block(
    channels, norm_cfg, conv_cfg=None, spconv_backend=None, downsample=None
):
    """Build a basic residual block for the selected backend.

    The public ``SparseBasicBlock`` above is intentionally untouched for the
    default legacy path.  spconv v2's ``SparseSequential`` requires residual
    blocks to inherit its own ``SparseModule``, so that class is created lazily.
    """

    backend = resolve_spconv_backend(spconv_backend)
    if backend == "legacy":
        return SparseBasicBlock(
            channels,
            channels,
            norm_cfg=norm_cfg,
            conv_cfg=conv_cfg,
            downsample=downsample,
        )

    sparse_ops = get_spconv_module(backend)
    conv_cfg = conv_cfg or dict(type="SubMConv3d")
    conv_type = conv_cfg.get("type", "SubMConv3d")

    class SparseBasicBlockV2(sparse_ops.SparseModule):
        expansion = 1

        def __init__(self):
            super().__init__()
            conv_cls = getattr(sparse_ops, conv_type)
            conv_kwargs = {k: v for k, v in conv_cfg.items() if k != "type"}
            self.conv1 = register_legacy_weight_load_hook(
                conv_cls(
                    channels,
                    channels,
                    3,
                    stride=1,
                    padding=1,
                    bias=False,
                    **conv_kwargs,
                )
            )
            self.norm1_name, norm1 = build_norm_layer(norm_cfg, channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = register_legacy_weight_load_hook(
                conv_cls(
                    channels,
                    channels,
                    3,
                    stride=1,
                    padding=1,
                    bias=False,
                    **conv_kwargs,
                )
            )
            self.norm2_name, norm2 = build_norm_layer(norm_cfg, channels, postfix=2)
            self.add_module(self.norm2_name, norm2)
            self.downsample = downsample

        @staticmethod
        def _replace_features(tensor, features):
            return tensor.replace_feature(features)

        def forward(self, x):
            identity = x.features
            out = self.conv1(x)
            norm1 = getattr(self, self.norm1_name)
            norm2 = getattr(self, self.norm2_name)
            out = self._replace_features(out, self.relu(norm1(out.features)))
            out = self.conv2(out)
            out = self._replace_features(out, norm2(out.features))
            if self.downsample is not None:
                identity = self.downsample(x).features
            return self._replace_features(out, self.relu(out.features + identity))

    return SparseBasicBlockV2()
