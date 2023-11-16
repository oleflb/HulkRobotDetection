from ..models.ssdnet import SSDNet
from ..models.lightning import LightningWrapper
import sys
import torch
from onnxsim import simplify
import onnx

model = LightningWrapper.load_from_checkpoint(
    sys.argv[1], map_location=torch.device("cpu")
)

image_size = model.image_size
input_image = torch.randn((1, 3, *image_size))
name = (
    f"{model.model_variant}_{model.image_size[0]}_{model.image_size[1]}_{sys.argv[2]}"
)
model = model.model.reparameterize(image_size)
model.train(False)
torch.onnx.export(
    model, input_image, name, input_names=["data"], output_names=["boxes", "scores"]
)

onnx_model = onnx.load(name)
model_simp, check = simplify(onnx_model)
assert check, "could not simplify model"
onnx.save(model_simp, f"simplified-{name}")
