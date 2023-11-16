import openvino as ov
from ..dataloader.lightningdataset import DataModule
from .draw_bbox import (
    draw_bboxes_on_axis_from_prediction,
    draw_bboxes_on_axis_from_truth,
)
from ..models.lightning import LightningWrapper
from ..models.ssdnet import ReparameterizedSSDNet

import argparse
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
from os import path
import onnxruntime as ort
from albumentations.pytorch.transforms import ToTensorV2

class Model:
    def __init__(self, runnable_model):
        self.model = runnable_model
    
    @staticmethod
    def create(runtime: str, model_path: str, reparameterize=False):
        print(reparameterize)
        model_extension = path.splitext(model_path)[1]
        if runtime == "ort":
            assert model_extension == ".onnx", "ort only supports onnx format"
            return Model(ort.InferenceSession(model_path))

        elif runtime == "torch":
            assert model_extension == ".ckpt", "torch only supports .ckpt format"
            model = LightningWrapper.load_from_checkpoint(
                model_path, map_location=torch.device("cpu")
            )
            model.eval()
            if reparameterize:
                return Model(model.model.reparameterize(model.image_size))
            else:
                return Model(model)

        elif runtime == "openvino":
            assert model_extension in [".onnx", ".xml"]
            core = ov.Core()
            ov_model = core.read_model(model_path)
            return Model(core.compile_model(model=ov_model, device_name="CPU"))

        raise ValueError(f"{runtime} is not a valid runtime")
    
    def infer(self, image):
        print(type(self.model))
        if isinstance(self.model, nn.Module):
            boxes, scores = self.infer_torch(image)
        elif isinstance(self.model, ov.runtime.ie_api.CompiledModel):
            boxes, scores = self.infer_openvino(image)
        elif isinstance(self.model, ort.capi.onnxruntime_inference_collection.InferenceSession):
            boxes, scores = self.infer_ort(image)

        print(boxes.shape, scores.shape)

    def infer_torch(self, image):
        if isinstance(self.model, ReparameterizedSSDNet):
            detection = self.model(torch.from_numpy(image))
        else:
            detection = self.model(torch.from_numpy(image))

        boxes = np.array([prediction["boxes"].detach().numpy() for prediction in detection])
        scores = np.array([prediction["scores"].detach().numpy() for prediction in detection])
        return boxes, scores

    def infer_openvino(self, image):
        predictions = self.model(image)
        return predictions["boxes"], predictions["scores"]

    def infer_ort(self, image):
        return self.model.run(["boxes", "scores"], {"data": image})

    


def infer(image: Image, compiled_model):
    image = ToTensorV2()(image=np.array(image).astype(np.float32))["image"]
    image = image / 255.0

    print(image.shape)
    print(torch.min(image), torch.max(image))
    
    boxes, scores = compiled_model.run(["boxes", "scores"], {"data": np.array(image[None, :, :, :])})
    # print(predictions)
    # predictions = compiled_model(image[None, :, :, :])
    boxes = torch.from_numpy(boxes)
    # print(predictions.keys())
    scores = torch.from_numpy(scores)
    print(boxes.shape)
    print(scores.shape)

    detection = {
        "labels": torch.argmax(scores, dim=-1),
        "scores": torch.max(scores, dim=-1).values,
        "boxes": boxes,
    }
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
    # if path.splitext(args.model)[1] == ".onnx":
    #     ov_model = ov.convert_model(args.model)
    # else:
    #     ov_model = args.model

    # core = ov.Core()
    # compiled_model = core.compile_model(model=ov_model, device_name="CPU")
    # output_layer = compiled_model.output(0)
    # image_size = np.array(compiled_model.input(0).shape)[-2:]

    model = ort.InferenceSession(args.model)
    input_layer = model.get_inputs()[0]
    image_size = input_layer.shape[-2:]   
    output_layer = model.get_outputs()[0]

    image = Image.open(args.image).convert("RGB").resize(image_size[::-1], resample=Image.Resampling.NEAREST)
    detection = infer(image, model)

    fig, ax = plt.subplots()

    ax.imshow(image)
    draw_bboxes_on_axis_from_prediction(
        ax, detection, image.height, image.width
    )

    fig.savefig(f"output.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="the model path")
    parser.add_argument("--runtime", help="the runtime to use for inference", choices=["torch", "openvino", "ort"])
    parser.add_argument("--reparameterize", help="whether to reparameterize the lightning model", default=False)
    parser.add_argument("--image", help="the image path", default=None)
    args = parser.parse_args()
    model = Model.create(args.runtime, args.model, reparameterize=args.reparameterize)
    
    image = np.random.randn(1, 3, 120, 160).astype(np.float32)
    model.infer(image)
    # main(parser.parse_args())
