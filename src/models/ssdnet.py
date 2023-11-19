import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision.ops import boxes as box_ops
from torchvision.models.detection.ssd import SSD, DefaultBoxGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from .mobileone import mobileone
import copy
from typing import Tuple, List
import numpy as np
from collections import OrderedDict

class FPNMobileNetBackbone(nn.Module):
    def __init__(self, backbone: nn.Module, returned_layers: List[str] = None, out_channels = 256):
        super().__init__()
        stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
        num_stages = len(stage_indices)

        if returned_layers is None:
            returned_layers = [num_stages - 2, num_stages - 1]
        
        return_layers = {f'{stage_indices[k]}': str(v) for v, k in enumerate(returned_layers)}
        in_channels_list = [backbone[stage_indices[i]].out_channels for i in returned_layers]

        self.backbone = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)

    def forward(self, x):
        return self.backbone(x)


class SSDNet(nn.Module):
    def __init__(
        self,
        image_size: Tuple[int, int],
        num_classes: int,
        detections_per_img: int,
        backbone: str,
    ):
        super().__init__()

        if backbone == "squeezenet":
            self.backbone = models.squeezenet1_0().features
        elif backbone == "mobileone":
            self.backbone = mobileone(num_classes, variant="s0")
        elif backbone == "mobilenetv3":
            self.backbone = FPNMobileNetBackbone(models.mobilenet_v3_small(weights="DEFAULT").features, out_channels=64)
        else:
            raise ValueError(f"{backbone} is not a valid backbone")

        anchor_generator = DefaultBoxGenerator(aspect_ratios=((0.5, 1.0, 2.0),(0.5, 1.0, 2.0),(0.5, 1.0, 2.0),))
        print(f"image_size is (h*w) {image_size}")
        self.ssd = SSD(
            self.backbone,
            anchor_generator,
            image_size[::-1],  # the image size here has to be w * h
            num_classes,
            detections_per_img=detections_per_img,
        )

    def forward(self, images, labels=None):
        return self.ssd(images, labels)

    def reparameterize(self, image_shape) -> nn.Module:
        return ReparameterizedSSDNet(
            image_shape=image_shape,
            backbone=copy.deepcopy(self.ssd.backbone),
            head=copy.deepcopy(self.ssd.head),
            anchor_generator=copy.deepcopy(self.ssd.anchor_generator),
            box_coder=copy.deepcopy(self.ssd.box_coder),
        )


class ReparameterizedBoxGenerator(nn.Module):
    def __init__(self, image_shape, default_box_generator):
        super().__init__()
        self.image_shape = image_shape
        self.box_generator = default_box_generator

    def forward(self, feature_maps: List[torch.Tensor]) -> torch.Tensor:
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        default_boxes = self.box_generator._grid_default_boxes(
            grid_sizes, self.image_shape, dtype=dtype
        )
        default_boxes = default_boxes.to(device)

        x_y_size = torch.tensor(
            [self.image_shape[1], self.image_shape[0]], device=default_boxes.device
        )

        dboxes_in_image = default_boxes
        dboxes_in_image = torch.cat(
            [
                (dboxes_in_image[:, :2] - 0.5 * dboxes_in_image[:, 2:]) * x_y_size,
                (dboxes_in_image[:, :2] + 0.5 * dboxes_in_image[:, 2:]) * x_y_size,
            ],
            -1,
        )
        return dboxes_in_image


class ReparameterizedSSDNet(nn.Module):
    def __init__(self, image_shape, backbone, head, anchor_generator, box_coder):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.anchor_generator = ReparameterizedBoxGenerator(
            image_shape, anchor_generator
        )
        self.box_coder = box_coder
        self.image_shape = image_shape
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def forward(self, images):
        if not isinstance(images, torch.Tensor):
            images = torch.stack(images)

        batch_size = images.shape[0]
        images = self.normalize(images)
        features = self.backbone(images)
        
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        features = list(features.values())
        
        head_outputs = self.head(features)

        bbox_regression = head_outputs["bbox_regression"]
        pred_scores = F.softmax(head_outputs["cls_logits"], dim=-1)

        image_anchors = self.anchor_generator(features)
        # bbox_regression = self.box_coder.decode(bbox_regression, image_anchors)
        if batch_size == 1:
            print(bbox_regression.shape)
            boxes = self.box_coder.decode_single(bbox_regression[0], image_anchors)
            boxes = box_ops.clip_boxes_to_image(boxes, self.image_shape)
            return [{"boxes": boxes, "scores": pred_scores}]
        
        # return bbox_regression, pred_scores
        results = []

        for boxes, scores in zip(bbox_regression, pred_scores):
            boxes = self.box_coder.decode_single(boxes, image_anchors)
            boxes = box_ops.clip_boxes_to_image(boxes, self.image_shape)

            results.append({
                "boxes": boxes,
                "scores": scores,
            })

        return results
    @classmethod
    def parse_output(cls, bbox_regression, prediction_scores, conf_thresh=0.2):
        assert bbox_regression.ndim == 3
        assert prediction_scores.ndim == 3

        detections = []
        for bboxes, predictions in zip(bbox_regression, prediction_scores):
            scores, labels = torch.max(predictions, dim=-1)
            selector = (labels != 0) * (scores >= conf_thresh)

            detections.append({
                "boxes": bboxes[selector],
                "scores": scores[selector],
                "labels": labels[selector],
            })
        return detections