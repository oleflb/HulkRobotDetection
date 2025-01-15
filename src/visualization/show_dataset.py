from ..dataloader.lightningdataset import DataModule
from ..models.lightning import LightningWrapper
from ..models.ssdnet import SSDNet, ReparameterizedSSDNet
from .draw_bbox import (
    draw_bboxes_on_axis_from_prediction,
    draw_bboxes_on_axis_from_truth,
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import torch.nn.functional as F
import matplotlib as mpl
mpl.use("QtAgg")

import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import torchvision.transforms as T
import argparse

def load_torch_model(args):
    image_size = (480, 640)
    if args.ckpt:
        model = LightningWrapper.load_from_checkpoint(
            args.ckpt, map_location=torch.device("cpu"), feature_mode="last"
        )
        image_size = model.image_size
        model.eval()
    if args.reparameterize:
        model = model.model.reparameterize(image_size)
        model.eval()
    if args.image_size:
        image_size = eval(args.image_size)
    if args.ckpt:
        return model, image_size
    # use default image size
    return None, image_size

def load_yolo_model(args):
    if args.yolo:
        return YOLO(args.yolo)
    return None

def main(args):
    torch_model, image_size = load_torch_model(args)
    yolo_model = load_yolo_model(args)

    dataloader = DataModule(image_size, num_workers=8, batch_size=int(args.batchsize))
    dataloader.setup("real")
    dataloader.setup("fit")
    dataloader.setup("test")
    dataset = dataloader.val_dataloader()

    for images, labels in dataset:
        detections = {
            "truth": labels,
        }
        if torch_model:
            torch_detections = torch_model(images)
            detections["torch"] = torch_detections

        if isinstance(torch_model, ReparameterizedSSDNet):
            torch_detections = ReparameterizedSSDNet.parse_output(
                bbox_regression=torch.stack([detection["boxes"] for detection in torch_detections]),
                prediction_scores=torch.stack([detection["scores"] for detection in torch_detections])
            )
            detections["torch"] = torch_detections

        if yolo_model:
            yolo_detections = yolo_model([T.ToPILImage()(image) for image in images])
            detections["yolo"] = yolo_detections

        for idx, image in enumerate(images):
            image = T.ToPILImage()(image)
            fig, ax = plt.subplots()
            ax.imshow(image)
            ax.set_title(labels[idx]["image_id"])
            
            draw_bboxes_on_axis_from_truth(ax, labels[idx])
            if "torch" in detections:
                draw_bboxes_on_axis_from_prediction(
                    ax, detections["torch"][idx], image.height, image.width
                )

            if "yolo" in detections:
                yolo_detection = detections["yolo"][idx]
                boxes = yolo_detection.boxes
                try:
                    prediction = {
                        "boxes": torch.vstack([box.xyxy[0] for box in boxes]),
                        "scores": torch.tensor([box.conf.item() for box in boxes]),
                        "labels": torch.tensor([int(box.cls + 1) for box in boxes]),
                    }
                    draw_bboxes_on_axis_from_prediction(
                        ax, prediction, image.height, image.width, color="yellow"
                    )
                except RuntimeError:
                    # no detections in this image
                    pass

            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo", help="a path to a yolo .pt file", default=None)
    parser.add_argument("--ckpt", help="a path to a .ckpt file that can be loaded with a LightningWrapper", default=None)
    parser.add_argument("--reparameterize", help="whether to reparameterize the lightning model", default=False)
    parser.add_argument("--batchsize", help="the batchsize", default=16)
    parser.add_argument("--image_size", help="the image size to use", default=None)
    main(parser.parse_args())
