from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from torch.utils.data import DataLoader
from os import path

from .torchdataset import BBoxDataset
from ..augmentation.augmenter import Augmenter


def collate_fn(batch):
    images, labels = tuple(zip(*batch))
    return torch.stack(images), labels


class DataModule(LightningDataModule):
    def __init__(
        self,
        image_size,
        data_directory: str = "datasets",
        test_file: str = "test.txt",
        train_file: str = "train.txt",
        val_file: str = "val.txt",
        real_file: str = "real.txt",
        batch_size=16,
        num_workers=24,
        collate_function=collate_fn,
    ):
        super().__init__()
        self.train_data_directory = path.join(data_directory, train_file)
        self.test_data_directory = path.join(data_directory, test_file)
        self.val_data_directory = path.join(data_directory, val_file)
        self.real_data_directory = path.join(data_directory, real_file)
        self.image_height, self.image_width = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_function = collate_fn

    def setup(self, stage: str):
        if stage in ["fit", "validate"]:
            augmenter = Augmenter(width=self.image_width,
                                  height=self.image_height)
            self.train_dataset = BBoxDataset(
                self.train_data_directory,
                augmenter.transform,
            )
            self.val_dataset = BBoxDataset(
                self.val_data_directory,
                augmenter.test_transform,
            )

        if stage in ["test"]:
            augmenter = Augmenter(width=self.image_width,
                                  height=self.image_height)
            self.test_dataset = BBoxDataset(
                self.test_data_directory,
                augmenter.test_transform,
            )

        if stage in ["real"]:
            augmenter = Augmenter(width=self.image_width,
                                  height=self.image_height)
            self.real_dataset = BBoxDataset(
                self.real_data_directory,
                augmenter.real_transform,
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.collate_function,
            drop_last=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.collate_function,
            drop_last=False,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.collate_function,
            drop_last=False,
        )

    def real_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.real_dataset,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.collate_function,
            drop_last=False,
        )
