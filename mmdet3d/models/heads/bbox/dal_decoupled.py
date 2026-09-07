"""DAL-style decoupled detection head for dynamic BEV fusion."""

import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmdet3d.models.builder import HEADS

from .transfusion import TransFusionHead


@HEADS.register_module()
class DALDecoupledHead(TransFusionHead):
    """Classify fused proposals while regressing from LiDAR-only features.

    Proposal locations and classes come from the fused dense heatmap. At those
    locations, classification consumes the fused decoder feature and box
    regression consumes a separately projected LiDAR feature. The inherited
    TransFusion target assignment, losses and bbox coder remain unchanged.
    """

    def __init__(self, lidar_in_channels=256, hidden_channel=128, **kwargs):
        super().__init__(hidden_channel=hidden_channel, **kwargs)
        if self.num_decoder_layers != 1:
            raise ValueError("DALDecoupledHead currently requires num_decoder_layers=1")
        self.lidar_shared_conv = ConvModule(
            lidar_in_channels,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias="auto",
            conv_cfg=dict(type="Conv2d"),
            norm_cfg=dict(type="BN2d"),
        )
        self.init_bn_momentum()

    def create_2D_grid(self, x_size, y_size):
        """Return flattened ``(x, y)`` positions for a ``[Y, X]`` tensor."""
        y, x = torch.meshgrid(
            torch.arange(y_size, dtype=torch.float32),
            torch.arange(x_size, dtype=torch.float32),
            indexing="ij",
        )
        return torch.stack((x + 0.5, y + 0.5), dim=-1).reshape(1, -1, 2)

    def _select_proposals(self, dense_heatmap):
        batch_size = dense_heatmap.shape[0]
        heatmap = dense_heatmap.detach().sigmoid()
        padding = self.nms_kernel_size // 2
        local_max = F.max_pool2d(
            heatmap,
            kernel_size=self.nms_kernel_size,
            stride=1,
            padding=padding,
        )
        if self.test_cfg["dataset"] == "nuScenes" and heatmap.shape[1] > 8:
            local_max[:, 8:10] = heatmap[:, 8:10]
        elif self.test_cfg["dataset"] == "Waymo" and heatmap.shape[1] > 2:
            local_max[:, 1:3] = heatmap[:, 1:3]
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.reshape(batch_size, heatmap.shape[1], -1)
        # Full argsort exports as TopK(K=C*H*W), exceeding TensorRT's K limit.
        top = torch.topk(
            heatmap.reshape(batch_size, -1), self.num_proposals, dim=-1
        ).indices
        proposal_class = torch.div(top, heatmap.shape[-1], rounding_mode="floor")
        proposal_index = top % heatmap.shape[-1]
        return heatmap, proposal_class, proposal_index

    @staticmethod
    def _gather(feature, proposal_index):
        feature = feature.flatten(2)
        return feature.gather(
            2, proposal_index[:, None].expand(-1, feature.shape[1], -1)
        )

    def forward_single(self, fused_inputs, lidar_bev, metas):
        if fused_inputs.shape[-2:] != lidar_bev.shape[-2:]:
            raise ValueError(
                "fused/LiDAR feature grids differ: "
                f"{tuple(fused_inputs.shape[-2:])} vs {tuple(lidar_bev.shape[-2:])}"
            )
        batch_size = fused_inputs.shape[0]
        fused_feature = self.shared_conv(fused_inputs)
        lidar_feature = self.lidar_shared_conv(lidar_bev)
        dense_heatmap = self.heatmap_head(fused_feature)
        heatmap, proposal_class, proposal_index = self._select_proposals(dense_heatmap)
        self.query_labels = proposal_class

        fused_query = self._gather(fused_feature, proposal_index)
        lidar_query = self._gather(lidar_feature, proposal_index)
        one_hot = F.one_hot(proposal_class, num_classes=self.num_classes).permute(0, 2, 1)
        category = self.class_encoding(one_hot.float())
        fused_query = fused_query + category
        lidar_query = lidar_query + category

        bev_pos = self.bev_pos.to(fused_feature).repeat(batch_size, 1, 1)
        query_pos = bev_pos.gather(
            1, proposal_index[..., None].expand(-1, -1, bev_pos.shape[-1])
        )

        prediction_head = self.prediction_heads[0]
        result = {
            name: getattr(prediction_head, name)(lidar_query)
            for name in prediction_head.heads
            if name != "heatmap"
        }
        result["heatmap"] = prediction_head.heatmap(fused_query)
        result["center"] = result["center"] + query_pos.permute(0, 2, 1)
        result["query_heatmap_score"] = heatmap.gather(
            2, proposal_index[:, None].expand(-1, self.num_classes, -1)
        )
        # The inherited decoder reads this as mutable module state. Expose it
        # explicitly so the class selection crosses a TensorRT boundary.
        result["query_labels"] = proposal_class.to(torch.int32)
        result["dense_heatmap"] = dense_heatmap
        return [result]

    def forward(self, fused_feats, lidar_bev, metas=None):
        if isinstance(fused_feats, (list, tuple)):
            if len(fused_feats) != 1:
                raise ValueError("DALDecoupledHead supports one fused feature level")
            fused_feats = fused_feats[0]
        return (self.forward_single(fused_feats, lidar_bev, metas),)


__all__ = ["DALDecoupledHead"]
