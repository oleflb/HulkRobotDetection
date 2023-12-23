import torch
from os import path
import numpy as np
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.ops import box_convert
from typing import Tuple
import json

JSON_LABELS = ["background", "ball", "robot", "goal_post", "pen_spot"]
CLASS_MAP = {
    0: 0, # background
    1: 1, # ball
    2: 2, # robot
    3: 3, # goal_post
    4: 4, # pen_spot
}

def convert_image_path_to_label_path(image_path):
    # image_path = .../datasets/NAME/images/123.png
    dir_name = path.dirname(image_path)
    # dir_name = .../datasets/NAME/images
    labels_path = "labels".join(dir_name.rsplit("images", 1))
    # labels_path = .../datasets/NAME/labels
    image_name = path.splitext(path.basename(image_path))[0]
    # image_name = 123
    label_path_without_ext = path.join(labels_path, image_name)
    # label_path_without_ext = .../datasets/NAME/labels/123
    return label_path_without_ext + ".txt"

def load_labels(labels_path):
    if not path.exists(labels_path):
        return np.array([]), np.array([])
    
    label_strings = open(labels_path).readlines()
    target_classes = []
    target_bboxes = []

    for label in label_strings:
        numbers = label.split()
        # the numbers in the dataset start at 0,
        # but torchvision interprets class 0 as background
        bbox_class = int(numbers[0]) + 1
        if bbox_class not in CLASS_MAP.keys():
            continue
        bbox = torch.tensor([float(coord) for coord in numbers[1:]])
        bbox = box_convert(bbox, "cxcywh", "xyxy")
        target_classes.append(CLASS_MAP[bbox_class])
        target_bboxes.append(np.array(bbox))

    return np.array(target_classes), np.array(target_bboxes)

class BBoxDataset(torch.utils.data.Dataset):
    def __init__(self, image_size: Tuple[int, int], txt_file: str, augmenter=None):
        self.augmenter = augmenter
        self.data_paths = list(
            (image_path.strip(), load_labels(convert_image_path_to_label_path(image_path.strip())))
            for image_path in open(txt_file).readlines()
        )
        self.image_size = image_size

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        image_path, (target_classes, target_bboxes) = self.data_paths[idx]
        # target_bboxes are in normalized xyxy format
        # target_classes, target_bboxes = label_path.load_labels()

        # albumentations only allows the bbox interval (0, 1], therefore we clip the bbox
        target_bboxes = np.clip(target_bboxes, np.finfo(np.float32).eps, 1.0)
        assert np.all(target_bboxes > 0.0), target_bboxes
        assert np.all(target_bboxes <= 1.0), target_bboxes

        height, width = self.image_size
        image = Image.open(image_path).resize((width, height), resample=Image.Resampling.NEAREST).convert("RGB")
        image = np.array(image, dtype=np.uint8)

        if self.augmenter:
            try:
                image, target_classes, target_bboxes = self.augmenter(
                    image, target_classes, target_bboxes
                )
            except ValueError as e:
                print(f"Error in bboxes: {label_path.path}")
                raise e
            
            target_classes = np.array(target_classes)
            target_bboxes = np.array(target_bboxes).reshape(-1, 4)

        # bbox format is normalized xyxy
        image = tv_tensors.Image(image / 255.0, dtype=torch.float32)

        target_classes = torch.from_numpy(target_classes).to(torch.long)
        # convert to absolute xyxy
        target_bboxes = torch.from_numpy(target_bboxes).to(
            torch.float32
        ) * torch.tensor([width, height, width, height])

        return image, {
            "labels": target_classes,
            "boxes": torch.reshape(target_bboxes, (-1, 4)),
            # "boxes": tv_tensors.BoundingBoxes(
            #     target_bboxes, format="XYXY", canvas_size=self.image_size
            # ),
            "image_id": image_path,
            # "area": area,
            "iscrowd": torch.zeros_like(target_classes, dtype=torch.int64),
        }

    def __get_labels(self, label_path: str):
        label_strings = []

        if path.exists(label_path):
            label_strings = open(label_path).readlines()
        target_classes = []
        target_bboxes = []

        for label in label_strings:
            numbers = label.split()
            # the numbers in the dataset start at 0,
            # but torchvision interprets class 0 as background
            bbox_class = int(numbers[0]) + 1
            bbox = np.array([float(coord) for coord in numbers[1:]])

            target_classes.append(bbox_class)
            target_bboxes.append(bbox)

            assert(np.all(bbox >= 0.0) and np.all(bbox <= 1.0))

        return np.array(target_classes), np.array(target_bboxes)
