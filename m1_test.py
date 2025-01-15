from src.models.mobileone import mobileone
from torchvision.models import mobilenet_v3_small
import torch

images = torch.randn(8, 3, 120, 160)

mobilenet = mobilenet_v3_small().features
print(mobilenet(images).shape)

m1 = mobileone(5, variant="s0")
print(m1(images).shape)