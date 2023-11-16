from ..models.ssdnet import SSDNet
from ..models.lightning import LightningWrapper
import sys
import torch
import openvino as ov

model = LightningWrapper.load_from_checkpoint(
    sys.argv[1], map_location=torch.device("cpu")
)

image_size = model.image_size
input_image = torch.randn((1, 3, *image_size))
name = (
    f"{model.model_variant}_{model.image_size[0]}_{model.image_size[1]}.xml"
)
model = model.model.reparameterize(image_size)
model.eval()
ov_model = ov.convert_model(model, example_input=input_image)
ov.save_model(ov_model, name)
