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
        "pretrained_weights": trial.suggest_categorical("pretrained", [True, False]),
        "feature_mode": "last",
        "model_variant": trial.suggest_categorical("model_variant", ["mobileone_s0", "mobileone_s1", "mobilenetv3_small_100", "efficientnet_b0", "efficientnet_b1", "convnext_nano", "convnextv2_nano"]),
        "image_width": trial.suggest_categorical("image_width", [160, 320, 640]),
        "image_height": trial.suggest_categorical("image_height", [120, 240, 480]),
        # "swa_lrs": trial.suggest_float("swa_lrs", 1e-4, 1e-1),
        # "out_channels": trial.suggest_categorical("num_out_channels", [16, 32, 64, 128, 256, 512, 1024]),
        # "feature_mode": trial.suggest_categorical("feature_mode", ["last"]),
        "initial_learning_rate": trial.suggest_float("initial_learning_rate", 1e-4, 1e-1, log=True),
    }

GLOBAL_GPU_INDEX = 0

def objective(trial: optuna.trial.Trial):
    study_name = trial.study.study_name
    hyperparameters = compose_hyperparameters(trial)
        
    BATCH_SIZE = 64
    image_size = (hyperparameters["image_height"], hyperparameters["image_width"])
    num_classes = 1 + 4
    model = LightningWrapper(
        image_size,
        num_classes,
        model=hyperparameters["model_variant"],
        batch_size=BATCH_SIZE,
        iou_threshold=0.5,
        conf_threshold=0.2,
        detections_per_img=500,
        learning_rate_reduction_factor=0.8,
        out_channels=1024,
        initial_learning_rate=hyperparameters["initial_learning_rate"],
        feature_mode="last", #hyperparameters["feature_mode"],
        pretrained_weights=hyperparameters["pretrained_weights"],
    )

    dataloader = DataModule(image_size, num_workers=8, batch_size=BATCH_SIZE)
    model = torch.compile(model)
    
    logger = WandbLogger(project=f"detection-{study_name}")
    trainer = Trainer(
        logger=logger,
        devices=[GLOBAL_GPU_INDEX],
        max_epochs=400,
        callbacks=[
            ModelCheckpoint(save_top_k=1, monitor="val/ma_f1", mode="max"),
            PyTorchLightningPruningCallback(
                   trial,
                   monitor="val/ma_f1"
            ),
            EarlyStopping(monitor="val/ma_f1", patience=30, mode="max", min_delta=0.01),
            # StochasticWeightAveraging(swa_lrs=1e-2),
            LearningRateMonitor(logging_interval='epoch')
        ],
        fast_dev_run=False)

    trainer.fit(model, dataloader)

    f1 = trainer.callback_metrics["val/ma_f1"].item()
    return f1


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
