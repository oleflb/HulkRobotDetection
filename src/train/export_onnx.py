from ..models.ssdnet import SSDNet
from ..models.lightning import LightningWrapper
import sys
import torch

model = LightningWrapper.load_from_checkpoint(sys.argv[1], map_location=torch.device("cpu"))

image_size = model.image_size
input_image = torch.randn((1,3,*image_size))
name = f"{model.model_variant}_{model.image_size[0]}_{model.image_size[1]}_{sys.argv[2]}"
model = model.model.reparameterize(image_size)
torch.onnx.export(model, input_image, name, input_names=["data"], output_names=["output"])
