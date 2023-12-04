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
        )
        self.map_metric = MeanAveragePrecision()
        self.iou_metric = IntersectionOverUnion()

        self.pretrained_weights = pretrained_weights
        self.out_channels = out_channels
        self.batch_size = batch_size
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.image_size = image_size
        self.num_classes = num_classes
        self.detections_per_img = detections_per_img
        self.learning_rate_reduction_factor = learning_rate_reduction_factor
        self.initial_learning_rate = initial_learning_rate


        self.save_hyperparameters()

    def forward(self, images):
        return self.model(images)

    def get_accuracy(self, predictions, labels):
        def filter_confidence(prediction, indices):
            return indices[prediction["scores"][indices] >= self.conf_threshold]

        box_indices = [
            nms(prediction["boxes"], prediction["scores"], self.iou_threshold)
            for prediction in predictions
        ]

        nms_prediction = [
            {
                "boxes": prediction["boxes"][box_indices],
                "scores": prediction["scores"][box_indices],
                "labels": prediction["labels"][box_indices],
            }
            for prediction, box_indices in zip(predictions, box_indices)
        ]

        return self.map_metric(nms_prediction, labels)

    def _step(self, images, labels):
        loss_dict = self.model(images, labels)
        losses = sum(loss for loss in loss_dict.values())

        return losses

    def training_step(self, batch, batch_idx):
        images, labels = batch
        loss = self._step(images, labels)
        self.log("train_loss", loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        with torch.no_grad():
            predictions = self.forward(images)
            self.train(True)
            loss = self._step(images, labels)
            self.train(False)

        iou = self.iou_metric(predictions, labels)["iou"]
        score = self.map_metric(predictions, labels) # self.get_accuracy(predictions, labels)
        # if score["map"] < 0.0:
        #     print("negative")
        # else:
        #     print("positive")
        # print(iou)
        # print([len(pred["labels"]) for pred in predictions])
        # print("prediction", [pred["labels"] for pred in predictions])
        # print("target", [pred["labels"] for pred in labels])



        self.log_dict({
            "val/map": score["map"],
            "val/map50": score["map_50"],
            "val/map75": score["map_75"],
            "val/mar1": score["mar_1"],
            "val/mar10": score["mar_10"],
            "val/mar100": score["mar_100"],
            "val/iou": iou,
            "val/loss": loss,
        }, sync_dist=True, batch_size=self.batch_size)

    def configure_optimizers(self):
        optimizer = optim.RMSprop(self.parameters(), lr=self.initial_learning_rate, weight_decay=1e-4, momentum=0.9, eps=0.0316, alpha=0.9)
#        optimizer = optim.AdamW(self.parameters(), lr=self.initial_learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=self.learning_rate_reduction_factor, patience=30, min_lr=1e-7)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/loss"}

    def forward(self, images):
        return self.model(images)
