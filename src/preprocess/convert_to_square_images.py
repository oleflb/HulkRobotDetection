import sys
import albumentations as A
from os import path
import os
from PIL import Image
from ..dataloader import torchdataset
from ..visualization.draw_bbox import draw_bboxes_on_axis_from_truth
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.ops import box_convert
import numpy as np
import torch

dataset = sys.argv[1]
new_dataset = f"{path.basename(dataset)}-square"

images = os.listdir(path.join(dataset, "images"))
with open("tmp.txt", "w") as f:
    f.write("\n".join(path.abspath(path.join(dataset, "images", image)) for image in images))


augmenter = A.Compose([
    A.CenterCrop(480, 480)
], bbox_params=A.BboxParams(
    format="albumentations", label_fields=["class_labels"],
),)

def transform(image, class_labels, bboxes):
    transformed = augmenter(
        image=image, bboxes=bboxes, class_labels=class_labels
    )
    return transformed["image"], transformed["class_labels"], transformed["bboxes"]

dataset = torchdataset.BBoxDataset("tmp.txt", augmenter=transform)

os.mkdir(path.join("datasets", new_dataset))
os.mkdir(path.join("datasets", new_dataset, "images"))
os.mkdir(path.join("datasets", new_dataset, "labels"))

for (image, label) in tqdm(dataset):
    image_name = path.basename(label["image_id"])
    txt_name = path.splitext(image_name)[0] + ".txt"

    image = Image.fromarray((image * 255.0).numpy().astype(np.uint8), 'RGB')
    image.save(path.join("datasets", new_dataset, "images", image_name))
    
    with open(path.join("datasets", new_dataset, "labels", txt_name), "w") as f:
        for bbox, label in zip(label["boxes"], label["labels"]):
            bbox = box_convert(bbox / 480., "xyxy", "cxcywh").numpy()
            bbox = bbox.astype(str)
            bbox = " ".join(bbox)
            f.write(f"{label - 1} {bbox}\n")
