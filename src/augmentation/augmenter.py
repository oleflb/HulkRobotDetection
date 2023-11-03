import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import cv2

class Augmenter:
    def __init__(self, height, width):
        self.height = height
        self.width  = width 
        print(self.width, self.height)
        self.__transform = A.Compose([
            A.RandomGamma(p=0.5),
            A.Rotate(limit=[-5,5], p=0.2, crop_border=True),
            A.Resize(self.height, self.width, interpolation=0),
            # A.OneOrOther(transforms=[
            #     A.RandomSizedCrop(min_max_height=[self.height // 2, self.height], height=self.height, width=self.width, w2h_ratio=self.width/self.height),
            # ], p=1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            # # A.ChannelShuffle(p=0.1),

            A.Blur(blur_limit=[1, 3]),
            A.RandomSunFlare(src_radius=100),
            # A.Lambda(image=self.add_bright_spots, p=0.5),
            A.RandomShadow(num_shadows_lower=3, num_shadows_upper=9, p=0.5),
            A.RGBShift(p=0.5),
            A.GaussNoise(p=0.5, var_limit=50),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=["class_labels"], min_visibility=0.2))
        
        self.__test_transform = A.Compose([
            A.Rotate(limit=[-20,20], p=1, crop_border=True),
            A.Resize(self.height, self.width, interpolation=0),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=1),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=["class_labels"], min_visibility=0.2))

        self.__real_transform = A.Compose([
            A.Resize(self.height, self.width, interpolation=0),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=["class_labels"], min_visibility=0.2))

    def transform(self, image, class_labels, bboxes):
        transformed = self.__transform(image=image, bboxes=bboxes, class_labels=class_labels)
        return transformed["image"], transformed["class_labels"], transformed["bboxes"]

    def test_transform(self, image, class_labels, bboxes):
        transformed = self.__test_transform(image=image, bboxes=bboxes, class_labels=class_labels)
        return transformed["image"], transformed["class_labels"], transformed["bboxes"]

    def real_transform(self, image, class_labels, bboxes):
        transformed = self.__real_transform(image=image, bboxes=bboxes, class_labels=class_labels)
        return transformed["image"], transformed["class_labels"], transformed["bboxes"]

    def add_bright_spots(self, image, **kwargs):
        max_number_bright_spots = 7
        number_bright_spots = np.random.randint(3, max_number_bright_spots + 1)


        for _ in range(number_bright_spots):
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            vertex = []
            for _ in range(4):
                vertex.append((np.random.randint(0, self.width), np.random.randint(0, self.height)))

            vertices = np.array([vertex], dtype=np.int32)
            
            cv2.fillPoly(mask, vertices, 255)

        image[:,:,0][mask == 255] = 255
        image[:,:,1][mask == 255] = 255
        image[:,:,2][mask == 255] = 255
        assert not np.any(np.isnan(image))

        return np.clip(image, 0, 255)
        