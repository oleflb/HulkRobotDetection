from lightning import Trainer
from lightning.pytorch.callbacks import StochasticWeightAveraging, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from ..models.lightning import LightningWrapper
from ..dataloader.lightningdataset import DataModule
import torch


def main():
    BATCH_SIZE = 64
    # image_size = (480, 640)
    image_size = (480 // 4, 640 // 4)
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
        out_channels=128,
        initial_learning_rate=2e-3,
        pretrained_weights=False
    )

    dataloader = DataModule(image_size, num_workers=12, batch_size=BATCH_SIZE)

    logger = WandbLogger(project="detection")
    trainer = Trainer(
        logger=logger,
        gradient_clip_val=10.0,
        max_epochs=800,
        # precision="16-mixed",
        callbacks=[
            ModelCheckpoint(save_top_k=1, monitor="val/map", mode="max"),
            StochasticWeightAveraging(swa_lrs=1e-4, annealing_epochs=50),
            LearningRateMonitor(logging_interval='step')
        ],
        fast_dev_run=False)

    # Does not work in DDP
    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(model, "binsearch")

    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
