import timm
import torch
from torch import nn
from typing import Tuple
from torchvision.ops import FeaturePyramidNetwork

from collections import OrderedDict

class RobotDetectionBackbone(nn.Module):
    def __init__(self, model_variant: str, pretrained_weights: bool, feature_mode: str = "last", out_channels: int = 256, image_size: Tuple[int, int] = (480, 640)):
        super().__init__()
        assert feature_mode in ["last", "all", "fpn"]
        self.image_size = image_size
        self.model_variant = model_variant
        self.pretrained_weights = pretrained_weights
        self.feature_mode = feature_mode
        self.fpn_out_channels = out_channels
        self.out_channels = None
        self.backbone = None
        self.fpn = None
        self.num_feature_maps = 1
        self._create_backbone()
        self._compute_out_channels()

    def _compute_out_channels(self):
            """
            Computes the number of output channels for each feature map in the backbone network.
            """
            example_input = torch.randn((1, 3, *self.image_size))
            with torch.no_grad():
                features = self.forward(example_input)
                if isinstance(features, torch.Tensor):
                    features = OrderedDict([("0", features)])

                if isinstance(features, list):
                    features = OrderedDict(list(enumerate(features)))

                self.out_channels = list(f.shape[1] for f in features.values())

    def _create_backbone(self):
        """
        Creates the backbone network and optionally the Feature Pyramid Network (FPN).
        """
        if self.feature_mode == "fpn" or self.feature_mode == "all":
            self.backbone = timm.create_model(self.model_variant, features_only=True, pretrained=self.pretrained_weights, num_classes=0, global_pool='')
        elif self.feature_mode == "last":
            self.backbone = timm.create_model(self.model_variant, pretrained=self.pretrained_weights, num_classes=0, global_pool='')

        if self.feature_mode == "fpn":
            example_input = torch.randn((1, 3, *self.image_size))
            with torch.autograd.detect_anomaly():
                features = self.backbone(example_input)

            self.num_feature_maps = len(features)
            feature_channels = [feature.shape[1] for feature in features]
            self.fpn = FeaturePyramidNetwork(feature_channels, out_channels=self.fpn_out_channels)

    def forward(self, x):
        """
        Forward pass of the backbone network.

        Args:
            x: Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        if self.feature_mode == "all":
            features = self.backbone(x)
            return OrderedDict(list(enumerate(features)))
        elif self.feature_mode == "fpn":
            features = self.backbone(x)
            return self.fpn(OrderedDict(list(enumerate(features))))
        elif self.feature_mode == "last":
            return self.backbone(x)