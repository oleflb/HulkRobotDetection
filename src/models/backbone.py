import timm
import torch
from torch import nn
from typing import Tuple

class RobotDetectionBackbone(nn.Module):
    def __init__(self, model_variant: str, pretrained_weights: bool, use_fpn: bool = False, out_channels: int = 256, image_size: Tuple[int, int] = (480, 640)):
        """
        Initializes the RobotDetectionBackbone class.

        Args:
            model_variant (str): The variant of the model to be used.
            pretrained_weights (bool): Whether to use pretrained weights or not.
            use_fpn (bool, optional): Whether to use Feature Pyramid Network (FPN) or not. Defaults to False.
            out_channels (int, optional): The number of output channels. Defaults to 256.
            image_size (Tuple[int, int], optional): The size of the input image. Defaults to (480, 640).
        """
        super().__init__()
        self.image_size = image_size
        self.model_variant = model_variant
        self.pretrained_weights = pretrained_weights
        self.use_fpn = use_fpn
        self.out_channels = out_channels
        self.backbone = None
        self.fpn = None
        self.num_feature_maps = 1
        self._create_backbone()
    
    def _create_backbone(self):
        """
        Creates the backbone network and optionally the Feature Pyramid Network (FPN).
        """
        self.backbone = timm.create_model(self.model_variant, features_only=self.use_fpn, pretrained=self.pretrained_weights)
        if self.use_fpn:
            example_input = torch.randn((1, 3, *self.image_size))
            with torch.no_grad():
                features = self.backbone(example_input)

            self.num_feature_maps = len(features)
            feature_channels = [feature.shape[1] for feature in features]
            self.fpn = FeaturePyramidNetwork(feature_channels, out_channels=self.out_channels)

    def forward(self, x):
        """
        Forward pass of the backbone network.

        Args:
            x: Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        if self.use_fpn:
            features = self.backbone(x)
            return self.fpn(OrderedDict(list(enumerate(features))))
        else:
            return self.backbone.forward_features(x)