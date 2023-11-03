from enum import Enum
import glob
from os import path
from typing import List
from torch.utils.data import random_split

class Type(Enum):
    Train = 0
    Validation = 1
    Test = 2
    TrainAndValidation = 3
    All = 4

class Split:
    def __init__(self, train: float, val: float, test: float) -> None:
        self.train = train
        self.val   = val
        self.test  = test
    
    def get_split(self, sequence, type: Type):
        if type == Type.TrainAndValidation:
            p_train = self.train / (self.train + self.val)
            train_len = int(len(sequence) * p_train)
            val_len   = len(sequence) - train_len
            return random_split(sequence, [train_len, val_len])
        if type == Type.All:
            train_len = int(len(sequence) * self.train)
            val_len   = int(len(sequence) * self.val)
            test_len  = len(sequence) - (train_len + val_len)
            return random_split(sequence, [train_len, val_len, test_len])

        

class DataList:
    def __init__(self, train_image_paths=None, validation_image_paths=None, test_image_paths=None):
        self.train_image_paths = list(path.realpath(image) for image in train_image_paths or [])
        self.validation_image_paths = list(path.realpath(image) for image in validation_image_paths or [])
        self.test_image_paths = list(path.realpath(image) for image in test_image_paths or [])

    def merge(self, other):
        self.train_image_paths += other.train_image_paths
        self.validation_image_paths += other.validation_image_paths
        self.test_image_paths += other.test_image_paths

    def is_negative(self, image_path: str) -> bool:
        (image_path, image_name) = path.split(image_path)
        label_name = path.splitext(image_name)[0] + ".txt"
        label_path = path.join(image_path, "..", "labels", label_name)
            
        return not path.exists(label_path)

    # f / (f+t) <= r
    # => f <= r * f + r * t
    # => f * (1-r) <= r * t
    # => f >= r * t / (1-r)
    def drop_negatives(self, rate: float):
        for current_list in [self.train_image_paths, self.validation_image_paths, self.test_image_paths]:
            is_negative = list(idx for (idx, image_path) in enumerate(current_list) if self.is_negative(image_path))
            negative_samples = len(is_negative)
            
            if negative_samples / len(current_list) <= rate:
                continue
            print(f"Rate: {negative_samples / len(current_list)}")
            positive_samples = len(current_list) - negative_samples
            target_negative_count = rate * positive_samples / ( 1 - rate )
            drop_amount = int(negative_samples - target_negative_count)
            print(f"negative samples: {negative_samples}, positive_samples: {positive_samples}, to drop: {drop_amount}")
            for drop_index in reversed(is_negative[:drop_amount]):
                assert self.is_negative(current_list[drop_index])
                current_list.pop(drop_index)
                negative_samples -= 1

            print(f"Rate: {negative_samples / len(current_list)}")
            is_negative = list(idx for (idx, image_path) in enumerate(current_list) if self.is_negative(image_path))
            negative_samples = len(is_negative)
            print(f"new negative: {negative_samples}, total: {len(current_list)}, ratio: {negative_samples / len(current_list)}")
            
    def dump_to_file(self, at: str):
        with open(path.join(at, "train.txt"), "w") as file:
            file.writelines(image + '\n' for image in self.train_image_paths)
        with open(path.join(at, "val.txt"), "w") as file:
            file.writelines(image + '\n' for image in self.validation_image_paths)
        with open(path.join(at, "test.txt"), "w") as file:
            file.writelines(image + '\n' for image in self.test_image_paths)

def add_data(image_path: str, type: Type, split: Split) -> DataList:
    if type == Type.Train:
        return DataList(train_image_paths=glob.glob(image_path))
    if type == Type.Validation:
        return DataList(validation_image_paths=glob.glob(image_path))
    if type == Type.Test:
        return DataList(test_image_paths=glob.glob(image_path))
    
    if type == Type.TrainAndValidation:
        [train_paths, val_paths] = split.get_split(glob.glob(image_path), type)
        return DataList(train_image_paths=train_paths, validation_image_paths=val_paths)
    
    if type == Type.All:
        [train_paths, val_paths, test_paths] = split.get_split(glob.glob(image_path), type)
        return DataList(train_image_paths=train_paths, validation_image_paths=val_paths, test_image_paths=test_paths)

def merge_datalists(datalists: List[DataList]) -> DataList:
    first = datalists[0]
    for datalist in datalists[1:]:
        first.merge(datalist)
    return first

if __name__ == "__main__":
    default_split = Split(0.75, 0.2, 0.05)
    target_negative_example_rate = 0.1
    datafolders = [
        ("./datasets/SPLObjDetectDatasetV2/trainval/images/*.png", Type.TrainAndValidation, default_split),
        ("./datasets/SPLObjDetectDatasetV2/test/images/*.png", Type.Test, default_split),
    #    ("./datasets/coco/images/*.jpg", Type.All, Split(1/3, 1/3, 1/3)),
    ]
    datalists = list(add_data(path, datatype, split) for (path, datatype, split) in datafolders)
    datalist = merge_datalists(datalists)
    datalist.drop_negatives(target_negative_example_rate)
    datalist.dump_to_file("./datasets")
