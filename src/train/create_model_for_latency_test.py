from ..models.lightning import LightningWrapper
from onnxsim import simplify
import onnx
import torch
import sys

BATCH_SIZE = 1
image_size = (480 // 2, 640 // 2)
num_classes = 1 + 4
model = LightningWrapper(
    image_size,
    num_classes,
    model="mobilenetv3",
    batch_size=BATCH_SIZE,
    iou_threshold=0.5,
    conf_threshold=0.2,
    detections_per_img=50,
    learning_rate_reduction_factor=0.8,
    out_channels=16,
    initial_learning_rate=2e-3,
    pretrained_weights=False,
    pixelunshuffle=2,
)

image_size = model.image_size
input_image = torch.randn((1, 3, *image_size))
name = (
    f"{model.model_variant}_{model.image_size[0]}_{model.image_size[1]}_{sys.argv[1]}"
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
