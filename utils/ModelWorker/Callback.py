import os
import warnings
import torch


class Callback:
    """
    回调类
    """

    def set_model(self, model):
        self.model = model

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def on_epoch_end(self, epoch, logs=None):
        pass


class EarlyStopping(Callback):
    def __init__(self, patience=5, min_delta=0, monitor="val_loss", mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.best_metric = None
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_metric = logs.get(self.monitor)
        if current_metric is None:
            warnings.warn(f"EarlyStopping monitor '{self.monitor}' not found in logs")
            return False

        if self.best_metric is None:
            self.best_metric = current_metric
            self.wait = 0
        else:
            if self.mode == "min":
                improved = current_metric < self.best_metric - self.min_delta
            else:
                improved = current_metric > self.best_metric + self.min_delta

            if improved:
                self.best_metric = current_metric
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    return True
        return False


class Checkpoint(Callback):
    def __init__(
        self, filepath, monitor="val_loss", mode="min", save_best_only=False, period=1
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.period = period
        self.epochs_since_last_save = 0
        self.best_metric = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save < self.period:
            return False

        self.epochs_since_last_save = 0
        current_metric = logs.get(self.monitor)
        if current_metric is None:
            warnings.warn(f"Checkpoint monitor '{self.monitor}' not found in logs")
            return False

        if self.save_best_only:
            if self.best_metric is None:
                self.best_metric = current_metric
                self._save_model(epoch, current_metric)
            else:
                if (self.mode == "min" and current_metric < self.best_metric) or (
                    self.mode == "max" and current_metric > self.best_metric
                ):
                    self.best_metric = current_metric
                    self._save_model(epoch, current_metric)
        else:
            self._save_model(epoch, current_metric)
        return False

    def _save_model(self, epoch, metric):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        filepath = self.filepath.format(epoch=epoch, metric=metric)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metric": metric,
            },
            filepath,
        )
        print(f"Checkpoint saved to {filepath}")
