from lightning import LightningModule
import torch
from torch import nn, optim

# from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt
import torchvision.transforms as T

from .ssdnet import SSDNet
from ..visualization.draw_bbox import draw_bboxes_on_axis_from_prediction
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.detection.iou import IntersectionOverUnion
from torchvision.ops import nms

from typing import Tuple


class LightningWrapper(LightningModule):
    def __init__(
        self,
        image_size: Tuple[int, int],
        num_classes: int,
        model: str,
        batch_size: int,
        iou_threshold: float,
        conf_threshold: float,
        detections_per_img: int,
        learning_rate_reduction_factor: float,
        out_channels: int,
        initial_learning_rate: float,
        pretrained_weights: bool,
        use_fpn: bool,
    ):
        super().__init__()
        self.model_variant = model
        self.model = SSDNet(
            image_size,
            num_classes,
            backbone=model,
            detections_per_img=detections_per_img,
            out_channels=out_channels,
            pretrained_weights=pretrained_weights,
            use_fpn=use_fpn,
        )
        self.map_metric = MeanAveragePrecision(iou_thresholds=[0.5, 0.75, 0.9], max_detection_thresholds=[50], backend="faster_coco_eval")
        self.iou_metric = IntersectionOverUnion()

        self.batch_size = batch_size
        self.conf_threshold = conf_threshold
        self.detections_per_img = detections_per_img
        self.image_size = image_size
        self.initial_learning_rate = initial_learning_rate
        self.iou_threshold = iou_threshold
        self.learning_rate_reduction_factor = learning_rate_reduction_factor
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.pretrained_weights = pretrained_weights
        self.use_fpn = use_fpn

        self.save_hyperparameters()

    def forward(self, images):
        return self.model(images)

    def on_train_start(self):
        self.iou_metric.reset()
        self.map_metric.reset()

    def _step(self, images, labels):
        return self.model(images, labels)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        loss = self._step(images, labels)
        bbox_loss = loss["bbox_regression"]
        class_loss = loss["classification"]

        self.log_dict({
            "train/bbox_loss": bbox_loss,
            "train/class_loss": class_loss,
            "train/loss": bbox_loss + class_loss,
        }, batch_size=self.batch_size)
        
        return bbox_loss + class_loss

    def on_validation_start(self):
        self.iou_metric.reset()
        self.map_metric.reset()

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        with torch.no_grad():
            predictions = self.forward(images)
            self.train(True)
            loss = self._step(images, labels)
            self.train(False)

        iou_score = self.iou_metric(predictions, labels)
        map_score = self.map_metric(predictions, labels) 

        self.log_dict({
            "val/iou": iou_score["iou"],
            "val/bbox_loss": loss["bbox_regression"],
            "val/class_loss": loss["classification"],
            "val/loss": loss["bbox_regression"] + loss["classification"],
            "val/map": map_score["map"],
            "val/map_50": map_score["map_50"],
            "val/map_75": map_score["map_75"],
            # "val/map_90": map_score["map_90"],
            "val/mar_1": map_score["mar_1"],
            "val/ma_f1": 2.0 * (map_score["map"] * map_score["mar_1"]) / (map_score["map"] + map_score["mar_1"]),   
        }, batch_size=self.batch_size, rank_zero_only=True, sync_dist=True, on_step=False, on_epoch=True)
            
    def configure_optimizers(self):
        optimizer = optim.RMSprop(self.parameters(), lr=self.initial_learning_rate, weight_decay=5e-4, momentum=0.937, eps=0.0316, alpha=0.9)
#        optimizer = optim.AdamW(self.parameters(), lr=self.initial_learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=self.learning_rate_reduction_factor, patience=15, min_lr=1e-7)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/loss"}

    def forward(self, images):
        return self.model(images)
