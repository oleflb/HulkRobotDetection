import torch
from os import path
import numpy as np
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.ops import box_convert
from typing import Tuple

def to_label_path(image_path: str) -> str:
    dir_name = path.dirname(image_path)
    label_path = "labels".join(dir_name.rsplit("images"))
    label_path = path.join(label_path, path.basename(image_path))
    return path.splitext(label_path)[0] + ".txt"

class BBoxDataset(torch.utils.data.Dataset):
    def __init__(self, image_size: Tuple[int, int], txt_file: str, augmenter=None):
        self.augmenter = augmenter
        self.data_paths = list((image_path.strip(), to_label_path(image_path.strip())) for image_path in open(txt_file).readlines())
        self.image_size = image_size

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        image_path, label_path = self.data_paths[idx]
        target_classes, target_bboxes = self.__get_labels(label_path)
        image  = Image.open(image_path)
        height, width = self.image_size
        image = np.array(image, dtype=np.uint8)

        # assert np.all(target_bboxes >= 0.0)
        # assert np.all(target_bboxes < 1.0)
        if self.augmenter:
            image, target_classes, target_bboxes = self.augmenter(image, target_classes, target_bboxes)
            target_classes = np.array(target_classes).astype(np.float32)
            target_bboxes = np.array(target_bboxes).reshape(-1, 4).astype(np.float32)

        # assert np.all(target_bboxes >= 0.0)
        # assert np.all(target_bboxes <= 1.0)
        
        # bbox format is cxcywh
        area = torch.from_numpy(target_bboxes[:, 2] * target_bboxes[:, 3])
        image = tv_tensors.Image(image / 255.0, dtype=torch.float32)

        target_classes = torch.from_numpy(target_classes).to(torch.long)
        # print("TODO: use absolute coordinates?")
        target_bboxes  = torch.from_numpy(target_bboxes).to(torch.float32) * torch.tensor([width, height, width, height])

        # convert bboxes to xyxy format
        target_bboxes = box_convert(target_bboxes, "cxcywh", "xyxy")
        
        return image, {
            "labels": target_classes,
            # "boxes": target_bboxes, 
            "boxes": tv_tensors.BoundingBoxes(target_bboxes, format="XYXY", canvas_size=self.image_size),
            "image_id": image_path,
            "area": area,
            "iscrowd": torch.zeros_like(target_classes, dtype=torch.int64)
        }

    def __get_labels(self, label_path: str):
        label_strings = []
        
        if path.exists(label_path):
            label_strings = open(label_path).readlines()
        target_classes = []
        target_bboxes  = []

        for label in label_strings:
            numbers = label.split()
            # the numbers in the dataset start at 0,
            # but torchvision interprets class 0 as background
            bbox_class = int(numbers[0]) + 1
            bbox = np.array([float(coord) for coord in numbers[1:]])

            target_classes.append(bbox_class)
            target_bboxes.append(bbox)

        return np.array(target_classes), np.array(target_bboxes)
