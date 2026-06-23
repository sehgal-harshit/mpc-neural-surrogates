"""Per-output variance head for SS-AE-NARX (ver_12) uncertainty quantification.

The ver_12 NARX_AE produces only a mean prediction. Here we add a parallel
log-variance head on the latent z (same shape as pred_head) trained with Gaussian
NLL while encoder/decoder/pred_head stay FROZEN -> the mean is bit-identical to
ver_12; only the variance head learns. Outputs are in scaled label space.
"""

import torch
from torch import nn


class VarianceHead(nn.Module):
    """Per-output log-variance head on the latent z (parallel to pred_head)."""

    def __init__(self, latent_dim=25, hidden=(64,), output_dim=26,
                 logvar_clamp=(-10.0, 5.0)):
        super().__init__()
        dims = [latent_dim] + list(hidden) + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)
        self.clamp = logvar_clamp

    def forward(self, z):
        return torch.clamp(self.net(z), *self.clamp)   # log-variance, scaled label space


class AE_NARX_UQ(nn.Module):
    """Frozen ver_12 base + trainable variance head. forward(x) -> (mu, logvar).

    `base` is a NARX_AE: base(x) -> (z, x_recon, y_pred). mu == y_pred (frozen).
    """

    def __init__(self, base, var_head):
        super().__init__()
        self.base = base.eval()
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.var_head = var_head

    def forward(self, x):
        with torch.no_grad():
            z, _, mu = self.base(x)          # mu frozen == ver_12
        return mu, self.var_head(z)

    @torch.no_grad()
    def latent(self, x):
        z, _, _ = self.base(x)
        return z


def gaussian_nll(mu, logvar, y):
    """Heteroscedastic Gaussian negative log-likelihood (scaled label space).

    NLL = 0.5 * mean[ exp(-logvar) * (y - mu)^2 + logvar ]   (constant dropped).
    mu is frozen; gradients flow only into logvar (the variance head).
    """
    return 0.5 * (torch.exp(-logvar) * (y - mu) ** 2 + logvar).mean()
