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
from typing import Tuple

class Model:
    def __init__(self, runnable_model):
        self.model = runnable_model
    
    @staticmethod
    def create(runtime: str, model_path: str, reparameterize=False):
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
        if isinstance(self.model, nn.Module):
            boxes, scores = self.infer_torch(image)
        elif isinstance(self.model, ov.runtime.ie_api.CompiledModel):
            boxes, scores = self.infer_openvino(image)
        elif isinstance(self.model, ort.capi.onnxruntime_inference_collection.InferenceSession):
            boxes, scores = self.infer_ort(image)
        else:
            raise TypeError(f"invalid model type {type(self.model)}")
        
        return torch.from_numpy(boxes), torch.from_numpy(np.squeeze(scores))

    def infer_torch(self, image):
        if isinstance(self.model, ReparameterizedSSDNet):
            detection = self.model(torch.from_numpy(image))
        else:
            detection = self.model(torch.from_numpy(image))
        boxes = np.array([prediction["boxes"].detach().numpy() for prediction in detection])[0]
        scores = np.array([prediction["scores"].detach().numpy() for prediction in detection])[0]

        return boxes, scores

    def infer_openvino(self, image):
        predictions = self.model(image)
        return predictions["boxes"], predictions["scores"]

    def infer_ort(self, image):
        return self.model.run(["boxes", "scores"], {"data": image})

    
def postprocess_predictions(box_predictions, score_predictions):
    detections = ReparameterizedSSDNet.parse_output(box_predictions[None, :, :], score_predictions[None, :, :], conf_thresh=0.001)
    return detections

def prepare_image_for_inference(image):
    image = ToTensorV2()(image=np.array(image).astype(np.float32))["image"]
    image = image / 255.0

    return np.array(image)[None, :, :, :]

def main(args):
    image_size = (120, 160)
    image = Image.open(args.image).convert("RGB").resize(image_size[::-1], resample=Image.Resampling.NEAREST)
    
    model = Model.create(args.runtime, args.model, reparameterize=args.reparameterize)
    
    inference_image = prepare_image_for_inference(image)
    boxes, scores = model.infer(inference_image)
    detections = postprocess_predictions(boxes, scores)
    fig, ax = plt.subplots()

    ax.imshow(image)
    draw_bboxes_on_axis_from_prediction(
        ax, detections[0], image.height, image.width, confidence_threshold=0.2
    )
    plt.show()
    # fig.savefig(f"output.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="the model path")
    parser.add_argument("--runtime", help="the runtime to use for inference", choices=["torch", "openvino", "ort"])
    parser.add_argument("--reparameterize", help="whether to reparameterize the lightning model", default=False)
    parser.add_argument("--image", help="the image path", default=None)
    
    main(parser.parse_args())
