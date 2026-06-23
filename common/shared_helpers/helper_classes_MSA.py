"""
Neural network classes for NARX model training.

Contents:
    pytorch_lightning_standard_network  - Base PyTorch Lightning module (train/val/test loop)
    GeLU                               - GELU activation function
    MLP                                - Feed-forward network; primary model for SO-NARX
    StagnationEarlyStopping            - Stops training when neither val nor train loss improves
"""

import torch
from torch import nn
import pytorch_lightning as pl


class pytorch_lightning_standard_network(pl.LightningModule):
    """Base Lightning module with standard train/val/test steps and configurable optimizer."""

    def __init__(self, loss_function, optimizer_class, optimizer_kwargs,
                 scheduler_class=None, scheduler_kwargs=None):
        super().__init__()
        self.loss_function = loss_function
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs

    def forward(self, x):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_function(self(x), y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        loss = self.loss_function(self(x), y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        loss = self.loss_function(self(x), y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)
        if self.scheduler_class is not None:
            scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
            }
        return optimizer

    def predict(self, x):
        return self(x)


class GeLU(nn.Module):
    """Tanh-based GELU approximation."""

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


class MLP(pytorch_lightning_standard_network):
    """
    Feed-forward MLP for MSA-NARX.

    network_hyperparameters keys:
        input_dim   : int
        hidden_dims : list[int]
        output_dim  : int
        activation  : nn.Module instance (e.g. nn.ReLU(), GeLU())

    training_hyperparameters keys (passed to base class):
        loss_function, optimizer_class, optimizer_kwargs,
        scheduler_class (optional), scheduler_kwargs (optional)
    """

    def __init__(self, network_hyperparameters, training_hyperparameters):
        super().__init__(**training_hyperparameters)
        self.save_hyperparameters()

        input_dim = network_hyperparameters['input_dim']
        hidden_dims = network_hyperparameters['hidden_dims']
        output_dim = network_hyperparameters['output_dim']
        activation = network_hyperparameters['activation']
        self.noise_sigma = network_hyperparameters.get('noise_sigma', 0.0)

        layers = []
        in_features = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(activation)
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.noise_sigma > 0.0:
            x = x + torch.randn_like(x) * self.noise_sigma
        loss = self.loss_function(self(x), y)
        self.log('train_loss', loss)
        return loss


class StagnationEarlyStopping(pl.callbacks.Callback):
    """
    Stops training when neither val_loss nor train_loss improves by at least
    `min_delta` (fractional) for `patience` consecutive validation checks.
    """

    def __init__(self, monitor='val_loss', monitor_train='train_loss',
                 patience=5, min_delta=0.01, mode='min', verbose=False):
        super().__init__()
        self.monitor = monitor
        self.monitor_train = monitor_train
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.wait_count = 0
        self.stopped_epoch = 0
        self.best_val_loss = None
        self.best_train_loss = None

    def on_validation_end(self, trainer, pl_module):
        current_val = trainer.callback_metrics.get(self.monitor)
        current_train = trainer.callback_metrics.get(self.monitor_train)
        if current_val is None or current_train is None:
            return

        current_val = float(current_val)
        current_train = float(current_train)

        if self.best_val_loss is None:
            self.best_val_loss = current_val
            self.best_train_loss = current_train
            return

        val_imp = (self.best_val_loss - current_val) / abs(self.best_val_loss) if self.best_val_loss else 0
        train_imp = (self.best_train_loss - current_train) / abs(self.best_train_loss) if self.best_train_loss else 0

        if self.mode == 'max':
            val_imp, train_imp = -val_imp, -train_imp

        val_improved = val_imp > self.min_delta
        train_improved = train_imp > self.min_delta

        if val_improved:
            self.best_val_loss = current_val
        if train_improved:
            self.best_train_loss = current_train

        if not val_improved and not train_improved:
            self.wait_count += 1
            if self.verbose:
                print(f"EarlyStopping patience: {self.wait_count}/{self.patience} "
                      f"(val_imp={val_imp:.4f}, train_imp={train_imp:.4f})")
        else:
            self.wait_count = 0

        if self.wait_count >= self.patience:
            self.stopped_epoch = trainer.current_epoch
            trainer.should_stop = True
            if self.verbose:
                print(f"Stopping at epoch {trainer.current_epoch}: no improvement in either loss")

    def on_train_end(self, trainer, pl_module):
        if self.stopped_epoch > 0 and self.verbose:
            print(f"Training stopped early at epoch {self.stopped_epoch}")


class MLP_MSA(pytorch_lightning_standard_network):
    """
    Feed-forward MLP for MSA-NARX.
    """

    def __init__(self, network_hyperparameters, training_hyperparameters):
        super().__init__(**training_hyperparameters)
        self.save_hyperparameters()

        input_dim = network_hyperparameters['input_dim']
        hidden_dims = network_hyperparameters['hidden_dims']
        self.base_output_dim = network_hyperparameters['base_output_dim']
        self.M = network_hyperparameters['M'] # Number of time steps to predict
        output_dim = network_hyperparameters['output_dim']  # typically base_output_dim * M
        activation = network_hyperparameters['activation']
        self.noise_sigma = network_hyperparameters.get('noise_sigma', 0.0)

        layers = []
        in_features = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(activation)
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.noise_sigma > 0.0:
            x = x + torch.randn_like(x) * self.noise_sigma
        loss = self.loss_function(self(x), y)
        self.log('train_loss', loss)
        return loss

    def predict_horizons(self, x):
        """Returns predictions reshaped to (batch, M, base_output_dim)"""
        pred = self(x)
        return pred.view(pred.shape[0], self.M, self.base_output_dim)

