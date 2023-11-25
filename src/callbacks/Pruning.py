import warnings

from packaging import version

import optuna
from optuna.storages._cached_storage import _CachedStorage
from optuna.storages._rdb.storage import RDBStorage

# Define key names of `Trial.system_attrs`.
_EPOCH_KEY = "ddp_pl:epoch"
_INTERMEDIATE_VALUE = "ddp_pl:intermediate_value"
_PRUNED_KEY = "ddp_pl:pruned"

from lightning import Callback, LightningModule, Trainer

class PyTorchLightningPruningCallback(Callback):
    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        super().__init__()

        self._trial = trial
        self.monitor = monitor
        self.is_ddp_backend = False

    def on_fit_start(self, trainer: Trainer, pl_module: "LightningModule") -> None:
        self.is_ddp_backend = trainer._accelerator_connector.is_distributed
        if not (
            isinstance(self._trial.study._storage, _CachedStorage)
            and isinstance(self._trial.study._storage._backend, RDBStorage)
        ):
            raise ValueError(
                "optuna.integration.PyTorchLightningPruningCallback"
                " supports only optuna.storages.RDBStorage in DDP."
            )
        # It is necessary to store intermediate values directly in the backend storage because
        # they are not properly propagated to main process due to cached storage.
        # TODO(Shinichi) Remove intermediate_values from system_attr after PR #4431 is merged.
        if trainer.is_global_zero:
            self._trial.storage.set_trial_system_attr(
                self._trial._trial_id,
                _INTERMEDIATE_VALUE,
                dict(),
            )

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Trainer calls `on_validation_end` for sanity check. Therefore, it is necessary to avoid
        # calling `trial.report` multiple times at epoch 0. For more details, see
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1391.
        if trainer.sanity_checking:
            return

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = (
                f"The metric '{self.monitor}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name."
            )
            warnings.warn(message)
            return

        # Workaround for multiple validation phases per epoch
        epoch = pl_module.global_step
        should_stop = False

        # Determine if the trial should be terminated in a single process.
        if not self.is_ddp_backend:
            self._trial.report(current_score.item(), step=epoch)
            if not self._trial.should_prune():
                return
            print("Trial pruned in non DDP")
            raise optuna.TrialPruned(f"Trial was pruned at epoch {epoch}.")

        # Determine if the trial should be terminated in a DDP.
        if trainer.is_global_zero:
            self._trial.report(current_score.item(), step=epoch)
            should_stop = self._trial.should_prune()

            # Update intermediate value in the storage.
            _trial_id = self._trial._trial_id
            _study = self._trial.study
            _trial_system_attrs = _study._storage.get_trial_system_attrs(_trial_id)
            intermediate_values = _trial_system_attrs.get(_INTERMEDIATE_VALUE)
            intermediate_values[epoch] = current_score.item()  # type: ignore[index]
            self._trial.storage.set_trial_system_attr(
                self._trial._trial_id, _INTERMEDIATE_VALUE, intermediate_values
            )

        # Terminate every process if any world process decides to stop.
        should_stop = trainer.strategy.broadcast(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if not should_stop:
            return

        print("Pruned in DDP")
        if trainer.is_global_zero:
            # Update system_attr from global zero process.
            self._trial.storage.set_trial_system_attr(self._trial._trial_id, _PRUNED_KEY, True)
            self._trial.storage.set_trial_system_attr(self._trial._trial_id, _EPOCH_KEY, epoch)

    def check_pruned(self) -> None:
        """Raise :class:`optuna.TrialPruned` manually if pruned.

        Currently, ``intermediate_values`` are not properly propagated between processes due to
        storage cache. Therefore, necessary information is kept in trial_system_attrs when the
        trial runs in a distributed situation. Please call this method right after calling
        ``pytorch_lightning.Trainer.fit()``.
        If a callback doesn't have any backend storage for DDP, this method does nothing.
        """

        _trial_id = self._trial._trial_id
        _study = self._trial.study
        # Confirm if storage is not InMemory in case this method is called in a non-distributed
        # situation by mistake.
        if not isinstance(_study._storage, _CachedStorage):
            return

        _trial_system_attrs = _study._storage._backend.get_trial_system_attrs(_trial_id)
        is_pruned = _trial_system_attrs.get(_PRUNED_KEY)
        intermediate_values = _trial_system_attrs.get(_INTERMEDIATE_VALUE)

        # Confirm if DDP backend is used in case this method is called from a non-DDP situation by
        # mistake.
        if intermediate_values is None:
            return
        for epoch, score in intermediate_values.items():
            self._trial.report(score, step=int(epoch))
        if is_pruned:
            epoch = _trial_system_attrs.get(_EPOCH_KEY)
            raise optuna.TrialPruned(f"Trial was pruned at epoch {epoch}.")