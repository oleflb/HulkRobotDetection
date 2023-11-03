from ..dataloader.lightningdataset import DataModule
from ..models.lightning import LightningWrapper
from ..models.ssdnet import SSDNet, ReparameterizedSSDNet
from .draw_bbox import draw_bboxes_on_axis_from_prediction, draw_bboxes_on_axis_from_truth
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import torchvision.transforms as T

def main():
    # image_size = (120, 160)
    model = LightningWrapper.load_from_checkpoint(sys.argv[1], map_location=torch.device("cpu"))
    image_size = model.image_size
    model = model.model
    model.eval()
    reparam_model = model.reparameterize(image_size)
    reparam_model.eval()
    
    yolo_model = YOLO(sys.argv[2])
    dataloader = DataModule(image_size, num_workers=8)
    dataloader.setup("fit")
    dataset = dataloader.val_dataloader()


    # for (image, label) in zip(images, labels):
    #     image = T.ToPILImage()(image)
    #     fig, ax = plt.subplots()
    #     ax.imshow(image)
    #     draw_bboxes_on_axis_from_truth(ax, label, image.height, image.width)
    #     plt.show()


    for images, labels in dataset:
        predictions = reparam_model(images)

        if isinstance(reparam_model, ReparameterizedSSDNet):
            predictions = [
                {
                    "labels": torch.argmax(prediction["scores"], dim=-1),
                    "scores": torch.max(prediction["scores"], dim=-1).values,
                    "boxes": prediction["boxes"]
                }
                for prediction in predictions
            ]
            predictions = [
                {
                    "labels": prediction["labels"][prediction["labels"] != 0],
                    "scores": prediction["scores"][prediction["labels"] != 0],
                    "boxes": prediction["boxes"][prediction["labels"] != 0]
                }
                for prediction in predictions
            ]
            

        for (image, reparam_prediction, prediction, label) in zip(images, predictions, model(images), labels):
            image = T.ToPILImage()(image)
            fig, (ax, ax2) = plt.subplots(1, 2)
            ax.imshow(image)
            draw_bboxes_on_axis_from_prediction(ax, reparam_prediction, image.height, image.width)
            # draw_bboxes_on_axis_from_prediction(ax, prediction, image.height, image.width, color="blue")
            draw_bboxes_on_axis_from_truth(ax, label, image.height, image.width)

            results = yolo_model(image)[0]
            annotator = Annotator(np.array(image))
            ax2.imshow(image)
            draw_bboxes_on_axis_from_truth(ax, label, image.height, image.width)
            try:
                boxes = results.boxes
                prediction = {
                    "boxes": torch.vstack([box.xyxy[0] for box in boxes]),
                    "scores": torch.tensor([box.conf.item() for box in boxes]),
                    "labels": torch.tensor([int(box.cls + 1) for box in boxes])
                }
                draw_bboxes_on_axis_from_prediction(ax2, prediction, image.height, image.width, color="yellow")
            except RuntimeError:
                pass
            plt.show()





if __name__ == "__main__":
    main()