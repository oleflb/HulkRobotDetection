import os
from datetime import timedelta
import sys

from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, StochasticWeightAveraging, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import optuna

from ..callbacks.Pruning import PyTorchLightningPruningCallback
from ..models.lightning import LightningWrapper
from ..dataloader.lightningdataset import DataModule

def compose_hyperparameters(trial: optuna.trial.Trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-8, 1e-1, log=True),
        # "swa_lrs": trial.suggest_float("swa_lrs", 1e-4, 1e-1),
        "out_channels": trial.suggest_categorical("num_out_channels", [16, 32, 64, 128, 256, 512, 1024]),
        "iou_threshold": trial.suggest_float("iou_threshold", 0.1, 0.9),
        "conf_threshold": trial.suggest_float("conf_threshold", 0.1, 0.9),
    }

GLOBAL_GPU_INDEX = 0

def objective(trial: optuna.trial.Trial):
    study_name = trial.study.study_name
    hyperparameters = compose_hyperparameters(trial)
        
    BATCH_SIZE = 32
    # image_size = (480, 640)
    image_size = (480 // 4, 640 // 4)
    num_classes = 1 + 4
    model = LightningWrapper(
        image_size,
        num_classes,
        model="mobilenetv3",
        batch_size=BATCH_SIZE,
        iou_threshold=hyperparameters["iou_threshold"],
        conf_threshold=hyperparameters["conf_threshold"],
        detections_per_img=50,
        learning_rate_reduction_factor=0.8,
        out_channels=hyperparameters["out_channels"]
    )

    dataloader = DataModule(image_size, num_workers=28, batch_size=BATCH_SIZE)

    logger = WandbLogger(project="detection")
    trainer = Trainer(
        logger=logger,
        max_epochs=400,
        callbacks=[
            ModelCheckpoint(save_top_k=1, monitor="val/iou", mode="max"),
            PyTorchLightningPruningCallback(
                   trial,
                   monitor="val/iou"
            ),
            # StochasticWeightAveraging(swa_lrs=1e-2),
            LearningRateMonitor(logging_interval='step')
        ],
        fast_dev_run=False)

    # Does not work in DDP
    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(model, "binsearch")

    trainer.fit(model, dataloader)

    iou = trainer.callback_metrics["val/iou"].item()
    
    return iou


if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.HyperbandPruner()
    study_name = sys.argv[1]
    storage = sys.argv[2]
    gpu_idx = sys.argv[3]
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=f"sqlite:///{storage}.db",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    timeout = timedelta(weeks=4)
    GLOBAL_GPU_INDEX = int(gpu_idx)
    study.optimize(objective, timeout=timeout.total_seconds())
