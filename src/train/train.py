from lightning import Trainer
from lightning.pytorch.callbacks import StochasticWeightAveraging, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from ..models.lightning import LightningWrapper
from ..dataloader.lightningdataset import DataModule
import torch
from tqdm import tqdm

def main():
    BATCH_SIZE = 128
    image_size = (480 // 4, 640 // 4)
    num_classes = 1 + 4

    # example models:
    # - 'repvit_m0_9'
    # - 'mobilenetv3_small_050', 'mobilenetv3_small_075', 'mobilenetv3_small_100'
    # - 'mobileone_s0', 'mobileone_s1'
    # - 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnetv2_s'
    # - 'convnext_nano', 'convnextv2_nano'

    model = LightningWrapper(
        image_size,
        num_classes,
        model="efficientnet_b0",
        batch_size=BATCH_SIZE,
        iou_threshold=0.5,
        conf_threshold=0.2,
        detections_per_img=500,
        learning_rate_reduction_factor=0.8,
        out_channels=1024,
        initial_learning_rate=2e-3,
        feature_mode="last",
        pretrained_weights=True,
    )
    model = torch.compile(model)
    
    dataloader = DataModule(image_size, num_workers=8, batch_size=BATCH_SIZE)

    logger = WandbLogger(project="detection")
    trainer = Trainer(
        logger=logger,
        gradient_clip_val=100.0,
        max_epochs=400,
        precision="16-mixed",
        # strategy="deepspeed_stage_2",
        callbacks=[
            ModelCheckpoint(save_top_k=1, monitor="val/ma_f1", mode="max", save_last=True),
            # StochasticWeightAveraging(swa_lrs=1e-4, annealing_epochs=50),
            LearningRateMonitor(logging_interval='step')
        ],
        fast_dev_run=False)

    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
