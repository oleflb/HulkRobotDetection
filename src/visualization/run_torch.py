import openvino as ov
from ..dataloader.lightningdataset import DataModule
from .draw_bbox import (
    draw_bboxes_on_axis_from_prediction,
    draw_bboxes_on_axis_from_truth,
)
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
from albumentations.pytorch.transforms import ToTensorV2

from PIL import Image
from os import path
from ..models.lightning import LightningWrapper

def infer(image: Image, model):
    image = ToTensorV2()(image=np.array(image).astype(np.float32))["image"]
    image = image / 255.0
    print(torch.min(image), torch.max(image))
    # image = np.transpose(image, [2,0,1])
    print(image.shape)
    
    boxes, scores = model(image[None, :, :, :])
    # boxes = torch.from_numpy(boxes)
    # scores = torch.from_numpy(scores)

    print(boxes.shape)
    print(scores.shape)

    detection = {
        "labels": torch.argmax(scores, dim=-1),
        "scores": torch.max(scores, dim=-1).values,
        "boxes": boxes,
    }
    # print(detection)
    print(len(detection["labels"]))
    detection = {
        "labels": detection["labels"][detection["labels"] != 0],
        "scores": detection["scores"][detection["labels"] != 0],
        "boxes": detection["boxes"][detection["labels"] != 0],
    }
    print(len(detection["labels"]))

    detection = {
        "labels": detection["labels"][detection["scores"] > 0.2],
        "scores": detection["scores"][detection["scores"] > 0.2],
        "boxes": detection["boxes"][detection["scores"] > 0.2],
    }
    return detection

def main(args):
    model = LightningWrapper.load_from_checkpoint(
        args.model, map_location=torch.device("cpu")
    )
    image_size = model.image_size
    print(image_size)
    model = model.model.reparameterize(image_size)
    model.eval()

    image = Image.open(args.image).convert("RGB").resize(image_size[::-1], resample=Image.Resampling.NEAREST)
    # image = ToTensorV2()(image=np.array(image).astype(np.float32))["image"]
    # image = image / 255.0

    # dataloader = DataModule(image_size, num_workers=8)
    # dataloader.setup("test")
    # dataloader.setup("fit")
    # dataloader.setup("real")
    # dataset = dataloader.real_dataloader()

    # images, _ = next(iter(dataset))
    # image_loader = torch.tensor(images[0])
    detection = infer(image, model)

    # print(torch.max(image - image_loader))


    image = T.ToPILImage()(image)
    fig, ax = plt.subplots()

    ax.imshow(image)
    draw_bboxes_on_axis_from_prediction(
        ax, detection, image.height, image.width
    )

    fig.savefig(f"output.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="the torch model path")
    parser.add_argument("--image", help="the image path")
    main(parser.parse_args())
