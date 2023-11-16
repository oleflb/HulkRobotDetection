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
from PIL import Image
from os import path

def infer(image: Image, compiled_model):
    image = np.array(image).astype(np.float32) / 255.0
    image = np.transpose(image, [2,0,1])
    print(image.shape)
    print(np.min(image), np.max(image))
    
    predictions = compiled_model(image[None, :, :, :])
    boxes = torch.from_numpy(predictions["boxes"])
    # print(predictions.keys())
    scores = torch.from_numpy(predictions["scores"])
    print(boxes.shape)
    print(scores.shape)

    detection = {
        "labels": torch.argmax(scores, dim=-1),
        "scores": torch.max(scores, dim=-1).values,
        "boxes": boxes,
    }
    print(detection)
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
    core = ov.Core()
    ov_model = core.read_model(args.model)
    compiled_model = core.compile_model(model=ov_model, device_name="CPU")
    output_layer = compiled_model.output(0)
    image_size = (120, 160) #np.array(compiled_model.input(0).shape)[-2:]

    image = Image.open(args.image).resize(image_size[::-1]).convert("RGB")
    detection = infer(image, compiled_model)

    fig, ax = plt.subplots()

    ax.imshow(image)
    draw_bboxes_on_axis_from_prediction(
        ax, detection, image.height, image.width
    )

    fig.savefig(f"output.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="the model path")
    parser.add_argument("--image", help="the image path")
    main(parser.parse_args())
