"""Variational autoencoder for MNIST (Exercise 4 - VAE, ANN-DL 2026.2).

Everything VAE-specific is written by hand in this file: the encoder/decoder
module, the reparameterization trick, both loss terms (BCE summed over the
784 pixels and the closed-form Gaussian KL) and the train / evaluate loops.
Autograd computes the gradients; no library VAE class and no
torch.distributions are used.
"""
import torch
import torch.nn.functional as F
from torch import nn


# --8<-- [start:model]
class VAE(nn.Module):
    """MLP VAE: 784 -> 512 -> 256 -> (mu, log sigma^2) in R^d -> 256 -> 512 -> 784."""

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        # Encoder trunk: 784 -> 512 -> 256, ReLU after each layer.
        self.encoder = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        # Two separate linear heads 256 -> d read the same features: one gives
        # the mean mu of q(z|x), the other its log-variance log sigma^2.
        # Predicting log sigma^2 (any real number) keeps sigma^2 = exp(.) > 0.
        self.mu_head = nn.Linear(256, d)
        self.logvar_head = nn.Linear(256, d)
        # Decoder: d -> 256 -> 512 -> 784, ReLU on the hidden layers and a
        # sigmoid at the output, so each of the 784 outputs is a Bernoulli
        # mean: the probability that the pixel is "on".
        self.decoder = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid(),
        )

    def encode(self, x):
        """x: (B, 784) in [0, 1]  ->  mu, logvar: (B, d) each."""
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def decode(self, z):
        """z: (B, d)  ->  x_hat: (B, 784), per-pixel Bernoulli means in (0, 1)."""
        return self.decoder(z)

    def forward(self, x):
        """One stochastic pass: encode, draw z by reparameterization, decode."""
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
# --8<-- [end:model]


# --8<-- [start:reparam]
def reparameterize(mu, logvar):
    """Draw z ~ N(mu, diag(sigma^2)) as a differentiable function of (mu, logvar).

    z = mu + sigma * eps with eps ~ N(0, I). The randomness lives in eps, which
    does not depend on any parameter, so dz/dmu = 1 and dz/dsigma = eps: the
    reconstruction gradient can flow back through z into the encoder.
    """
    sigma = torch.exp(0.5 * logvar)   # sigma = exp(log(sigma^2) / 2)
    eps = torch.randn_like(sigma)     # standard normal noise, shape (B, d)
    return mu + sigma * eps
# --8<-- [end:reparam]


# --8<-- [start:loss]
def loss_terms(x, x_hat, mu, logvar):
    """Return (recon, kl), each in nats per image.

    recon = -sum_j [x_j log x_hat_j + (1 - x_j) log(1 - x_hat_j)]   over the 784 pixels
    kl    = 1/2 sum_i (mu_i^2 + sigma_i^2 - 1 - log sigma_i^2)      over the d latent dims
    Both are summed per image and then averaged over the B images of the batch.
    """
    batch_size = x.shape[0]
    # Bernoulli negative log-likelihood. reduction="sum" adds the BCE over all
    # B x 784 pixels; dividing by B leaves nats per image. reduction="mean"
    # would also divide by 784, i.e. secretly set beta ~ 784.
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / batch_size
    # Closed-form KL( N(mu, diag sigma^2) || N(0, I) ), written out term by
    # term (sigma^2 = exp(logvar)), summed over dims and batch, divided by B.
    kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1.0 - logvar) / batch_size
    return recon, kl
# --8<-- [end:loss]


# --8<-- [start:train]
def run_epoch(model, loader, beta=1.0, optimizer=None):
    """One pass over `loader`; returns the mean (recon, kl) in nats per image.

    With an optimizer it trains; without one it only evaluates (no gradients).
    Evaluation uses the same single-sample reparameterized pass as training,
    so validation and test numbers are one-sample estimates of the same loss.
    """
    training = optimizer is not None
    model.train(training)
    total_recon, total_kl, n_images = 0.0, 0.0, 0
    with torch.set_grad_enabled(training):
        for x, _ in loader:                    # labels are never used by the VAE
            x_hat, mu, logvar = model(x)       # encode -> sample z -> decode
            recon, kl = loss_terms(x, x_hat, mu, logvar)
            if training:
                loss = recon + beta * kl       # negative ELBO per image (beta = 1)
                optimizer.zero_grad()
                loss.backward()                # autograd: decoder -> z -> encoder
                optimizer.step()
            b = x.shape[0]                     # the last batch is smaller, so
            total_recon += recon.item() * b    # weight each batch mean by its size
            total_kl += kl.item() * b
            n_images += b
    return total_recon / n_images, total_kl / n_images


def evaluate(model, loader, beta=1.0):
    """(recon, kl) of a fixed model on `loader`, in nats per image."""
    return run_epoch(model, loader, beta)


def train(model, train_loader, val_loader, epochs=20, lr=1e-3, beta=1.0):
    """Adam on the negative ELBO; logs both terms on train and validation every epoch."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_recon": [], "train_kl": [], "val_recon": [], "val_kl": []}
    for epoch in range(1, epochs + 1):
        # Training terms: averaged over the epoch's mini-batches while the weights move.
        tr_recon, tr_kl = run_epoch(model, train_loader, beta, optimizer)
        # Validation terms: the model as it stands at the end of the epoch.
        va_recon, va_kl = evaluate(model, val_loader, beta)
        for key, value in zip(history, (tr_recon, tr_kl, va_recon, va_kl)):
            history[key].append(value)
        print(f"  epoch {epoch:2d} | train recon {tr_recon:7.2f}  KL {tr_kl:6.2f}"
              f" | val recon {va_recon:7.2f}  KL {va_kl:6.2f}")
    return history
# --8<-- [end:train]
