"""Dynamic camera/LiDAR BEV fusion with a decoupled object head."""

from mmcv.runner import auto_fp16

from mmdet3d.models import FUSIONMODELS

from .dsvt_bevfusion import DSVTBEVFusion


@FUSIONMODELS.register_module()
class DynamicBEVFusion(DSVTBEVFusion):
    """Join DSVT, Depth-GFusion and a DAL-style object head.

    ``camera_bev`` is an intentional integration seam: it permits a camera BEV
    encoder (WidthFormer or an existing view transformer) to be tested without
    coupling that experiment to the detector. If it is omitted, the configured
    camera encoder is executed through the existing BEVFusion path.
    """

    @auto_fp16(apply_to=("img", "points", "camera_bev"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths=None,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        camera_bev=None,
        **kwargs,
    ):
        if points is None:
            raise ValueError("DynamicBEVFusion requires LiDAR points")
        lidar_bev = self.extract_features(points, "lidar")
        auxiliary_losses = {}

        if camera_bev is None:
            if "camera" not in self.encoders:
                raise ValueError(
                    "camera_bev was not supplied and no camera encoder is configured"
                )
            camera_bev = self.extract_camera_features(
                img,
                points,
                radar,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                gt_depths=depths,
            )
            if self.use_depth_loss:
                camera_bev, auxiliary_losses["depth"] = camera_bev[0], camera_bev[-1]

        fused_bev = self.fuser(camera_bev, lidar_bev)
        decoded = self.decoder["backbone"](fused_bev)
        decoded = self.decoder["neck"](decoded)

        if self.training:
            outputs = {}
            for head_type, head in self.heads.items():
                if head_type == "object":
                    predictions = head(decoded, lidar_bev, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, predictions)
                elif head_type == "map":
                    losses = head(decoded, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {head_type}")
                for name, value in losses.items():
                    key = "loss" if value.requires_grad else "stats"
                    scale = self.loss_scale[head_type] if value.requires_grad else 1.0
                    outputs[f"{key}/{head_type}/{name}"] = value * scale
            if self.use_depth_loss:
                if "depth" not in auxiliary_losses:
                    raise ValueError("depth loss is enabled but was not returned")
                outputs["loss/depth"] = auxiliary_losses["depth"]
            return outputs

        batch_size = fused_bev.shape[0]
        outputs = [{} for _ in range(batch_size)]
        for head_type, head in self.heads.items():
            if head_type == "object":
                predictions = head(decoded, lidar_bev, metas)
                bboxes = head.get_bboxes(predictions, metas)
                for batch_index, (boxes, scores, labels) in enumerate(bboxes):
                    outputs[batch_index].update(
                        boxes_3d=boxes.to("cpu"),
                        scores_3d=scores.cpu(),
                        labels_3d=labels.cpu(),
                    )
            elif head_type == "map":
                logits = head(decoded)
                for batch_index in range(batch_size):
                    outputs[batch_index].update(
                        masks_bev=logits[batch_index].cpu(),
                        gt_masks_bev=gt_masks_bev[batch_index].cpu(),
                    )
            else:
                raise ValueError(f"unsupported head: {head_type}")
        return outputs


