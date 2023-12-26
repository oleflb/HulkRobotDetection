import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import cv2


class Augmenter:
    def __init__(self, height, width):
        self.height = height
        self.width = width

        self.__transform = A.Compose(
            [
                A.OneOf([
                    A.Resize(height, width, interpolation=0),
                    A.Resize(height, width, interpolation=1),
                    A.Resize(height, width, interpolation=2),
                    # A.RandomSizedCrop(min_max_height=[self.height // 2, self.height], height=self.height, width=self.width, w2h_ratio=self.width/self.height, interpolation=0),
                    # A.RandomSizedCrop(min_max_height=[self.height // 2, self.height], height=self.height, width=self.width, w2h_ratio=self.width/self.height, interpolation=1),
                    # A.RandomSizedCrop(min_max_height=[self.height // 2, self.height], height=self.height, width=self.width, w2h_ratio=self.width/self.height, interpolation=2),
                ], p=1),
                A.ToGray(p=0.05),
                A.SomeOf([
                    A.RandomGamma(p=0.1),
                    A.CLAHE(p=0.2),
                    A.RandomBrightnessContrast(),
                    A.HueSaturationValue(),
                    A.RGBShift(),
                    A.PixelDropout(dropout_prob=0.05, per_channel=True),
                    A.ColorJitter(),
                    A.RandomSunFlare(src_radius=100),
                    # A.Blur(blur_limit=[5, 5]),
                    # A.ChannelShuffle(),
                ], n=3, p=0.6),
                # A.OneOf([ # not implented for bbox
                #     A.CoarseDropout(max_holes=16, max_height=height // 20, max_width=width // 20, fill_value=0.0),
                #     A.CoarseDropout(max_holes=16, max_height=height // 20, max_width=width // 20, fill_value=255),
                # ], p=1),
                A.HorizontalFlip(p=0.5),
                # A.Lambda(image=self.add_bright_spots, p=0.5),
                A.RandomShadow(num_shadows_lower=3, num_shadows_upper=9, p=0.5),
                A.GaussNoise(p=0.5, var_limit=50),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="albumentations", label_fields=["class_labels"], min_visibility=0.2, min_height=4.0, min_width=4.0,
            ),
        )

        self.__test_transform = A.Compose(
            [
                A.Rotate(limit=[-20, 20], p=1, crop_border=True),
                A.Resize(self.height, self.width, interpolation=0),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=1),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="albumentations", label_fields=["class_labels"], min_visibility=0.2
            ),
        )

        self.__real_transform = A.Compose(
            [
                A.Resize(self.height, self.width, interpolation=0), 
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="albumentations", label_fields=["class_labels"], min_visibility=0.2
            ),
        )

    def transform(self, image, class_labels, bboxes):
        transformed = self.__transform(
            image=image, bboxes=bboxes, class_labels=class_labels
        )
        return transformed["image"], transformed["class_labels"], transformed["bboxes"]

    def test_transform(self, image, class_labels, bboxes):
        transformed = self.__test_transform(
            image=image, bboxes=bboxes, class_labels=class_labels
        )
        return transformed["image"], transformed["class_labels"], transformed["bboxes"]

    def real_transform(self, image, class_labels, bboxes):
        transformed = self.__real_transform(
            image=image, bboxes=bboxes, class_labels=class_labels
        )
        return transformed["image"], transformed["class_labels"], transformed["bboxes"]

    def add_bright_spots(self, image, **kwargs):
        max_number_bright_spots = 3
        number_bright_spots = np.random.randint(3, max_number_bright_spots + 1)

        for _ in range(number_bright_spots):
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            vertex = []
            for _ in range(4):
                vertex.append(
                    (
                        np.random.randint(0, self.width),
                        np.random.randint(0, self.height),
                    )
                )

            vertices = np.array([vertex], dtype=np.int32)

            cv2.fillPoly(mask, vertices, 255)

        image[:, :, 0][mask == 255] = 255
        image[:, :, 1][mask == 255] = 255
        image[:, :, 2][mask == 255] = 255
        assert not np.any(np.isnan(image))

        return np.clip(image, 0, 255)
