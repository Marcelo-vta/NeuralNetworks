"""Exercise 4 - VAE (ANN-DL 2026.2): single entry point.

Run from the repository root:

    python docs/exercises/vae/code/main.py

MNIST is downloaded to ./data/ (git-ignored). Figures 1-6 are written to
docs/exercises/vae/figures/. Every number quoted in the report is printed
with a label. The d=16 model (Exercise 1) and the d=2 model (Exercise 2) are
trained in this same process, so Figure 6 reuses the d=16 model without any
checkpoint on disk.
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # render to files, no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from scipy.stats import norm
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vae import VAE, evaluate, loss_terms, reparameterize, train  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Fixed by the statement: Adam, lr 1e-3, batch 128, 20 epochs, beta = 1.
EPOCHS, BATCH_SIZE, LR, BETA = 20, 128, 1e-3, 1.0
# Exercise 2D: two test images of different digits (indices into the test set).
IDX_A, IDX_B = 1, 3
DIGIT_COLORS = plt.cm.tab10.colors         # one fixed colour per digit


def save(fig, name):
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figures/{name}")


# --8<-- [start:data]
def load_split():
    """The statement's split, verbatim: 50 000 train / 10 000 validation from
    the official training set (generator seeded with 42); the official 10 000
    test images are the test set."""
    train_full = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
    train, val = random_split(train_full, [50_000, 10_000],
                              generator=torch.Generator().manual_seed(42))
    return train, val, test


def flatten(dataset):
    """Run every image through the dataset's own ToTensor transform once
    (pixels in [0, 1]) and stack them into an (N, 784) tensor + (N,) labels.
    Doing it once instead of every epoch only saves time; values are identical."""
    xs, ys = zip(*(dataset[i] for i in range(len(dataset))))
    return torch.stack(xs).view(len(xs), 784), torch.tensor(ys)


def make_loaders(train_xy, val_xy):
    """Mini-batches of 128. Only the training loader shuffles; its order is
    drawn from torch's global RNG, which was just seeded with 42."""
    train_loader = DataLoader(TensorDataset(*train_xy), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(*val_xy), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)
    return train_loader, val_loader
# --8<-- [end:data]


# --8<-- [start:fit]
def fit_vae(d, train_xy, val_xy):
    """Seed -> build -> loaders -> train, identical for d=16 and d=2."""
    torch.manual_seed(42)                       # the statement's seed, before building
    model = VAE(d)
    print(f"[1B/2A] VAE(d={d}): {sum(p.numel() for p in model.parameters()):,} parameters")
    train_loader, val_loader = make_loaders(train_xy, val_xy)
    init_recon, init_kl = evaluate(model, val_loader)   # "epoch 0": untrained model
    print(f"  epoch  0 | (untrained)                      | val recon {init_recon:7.2f}"
          f"  KL {init_kl:6.2f}")
    t0 = time.perf_counter()
    history = train(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, beta=BETA)
    history["seconds"] = time.perf_counter() - t0
    history["init_val_recon"], history["init_val_kl"] = init_recon, init_kl
    return model, history


def test_numbers(model, test_xy, label):
    """Final numbers only: called once per model, after its training is over."""
    test_loader = DataLoader(TensorDataset(*test_xy), batch_size=BATCH_SIZE, shuffle=False)
    recon, kl = evaluate(model, test_loader)
    print(f"[{label}] test recon      = {recon:.2f} nats/image")
    print(f"[{label}] test KL         = {kl:.2f} nats/image")
    print(f"[{label}] test neg. ELBO  = {recon + kl:.2f} nats/image")
    return recon, kl
# --8<-- [end:fit]


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
# --8<-- [start:tile]
def tile(images, nrows, ncols, pad=2):
    """Lay (nrows*ncols, 784) images out row-major in one array, with white
    `pad`-pixel separators. Returns the canvas and the tile-centre coordinates."""
    imgs = np.asarray(images).reshape(nrows, ncols, 28, 28)
    step = 28 + pad
    canvas = np.ones((nrows * step + pad, ncols * step + pad))
    for r in range(nrows):
        for c in range(ncols):
            canvas[pad + r * step: pad + r * step + 28, pad + c * step: pad + c * step + 28] = imgs[r, c]
    centres_x = pad + np.arange(ncols) * step + 13.5
    centres_y = pad + np.arange(nrows) * step + 13.5
    return canvas, centres_x, centres_y
# --8<-- [end:tile]


def gray_fraction(p):
    """Fraction of the 784 pixels that are neither clearly off nor clearly on."""
    return ((p > 0.1) & (p < 0.9)).float().mean(dim=1)


def bernoulli_entropy(p):
    """Entropy of the decoder's 784 independent Bernoulli pixels, nats per image."""
    p = p.clamp(1e-6, 1 - 1e-6)
    return -(p * p.log() + (1 - p) * (1 - p).log()).sum(dim=1)


def log_q_aggregate(z, mu, logvar, chunk=128):
    """log q(z) of the aggregate posterior q(z) = 1/N sum_n N(z; mu_n, diag sigma_n^2)."""
    out, var = [], logvar.exp()
    for zc in z.split(chunk):
        sq = (zc[:, None, :] - mu[None]) ** 2 / var[None]               # (m, N, d)
        logp = -0.5 * (np.log(2 * np.pi) + logvar[None] + sq).sum(-1)   # (m, N)
        out.append(torch.logsumexp(logp, dim=1) - np.log(mu.shape[0]))
    return torch.cat(out)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
# --8<-- [start:fig1]
def figure1(h):
    epochs = np.arange(1, EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, term, name in zip(axes, ("recon", "kl"), ("Reconstruction (BCE)", "KL")):
        ax.plot(epochs, h[f"train_{term}"], "o-", color="tab:blue", label="train (epoch average)")
        ax.plot(epochs, h[f"val_{term}"], "s-", color="tab:orange", label="validation (end of epoch)")
        ax.set_title(f"{name} term per epoch (d = 16, beta = 1)")
        ax.set_xlabel("epoch")
        ax.set_ylabel(f"{name} [nats / image]")
        ax.set_xticks(epochs[::2] if EPOCHS > 10 else epochs)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Figure 1 - the two terms of the VAE loss, logged separately", y=1.02)
    save(fig, "fig1.png")
# --8<-- [end:fig1]


# --8<-- [start:fig2]
def figure2(model, x_test, y_test):
    # First test image of each digit 0-9.
    idx = [int(torch.nonzero(y_test == c)[0]) for c in range(10)]
    x = x_test[idx]
    with torch.no_grad():
        mu, _ = model.encode(x)
        x_rec = model.decode(mu)          # decode the mean: no sampling of z
    fig, axes = plt.subplots(2, 10, figsize=(13, 3.3))
    for j, c in enumerate(range(10)):
        for i, img in enumerate((x[j], x_rec[j])):
            ax = axes[i, j]
            ax.imshow(img.view(28, 28), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[0, j].set_title(f"digit {c}\ntest #{idx[j]}", fontsize=9)
    axes[0, 0].set_ylabel("original x", fontsize=10)
    axes[1, 0].set_ylabel(r"decode($\mu$)", fontsize=10)
    fig.suptitle("Figure 2 - first test image of each digit (top) and its d = 16 "
                 r"reconstruction from $\mu$ (bottom)", y=1.03)
    save(fig, "fig2.png")
    return idx, x, x_rec
# --8<-- [end:fig2]


# --8<-- [start:fig3]
def figure3(model, d):
    torch.manual_seed(0)                  # the statement's seed for the samples
    z = torch.randn(64, d)                # 64 draws from the prior N(0, I)
    with torch.no_grad():
        samples = model.decode(z)         # Bernoulli means, shown as grey levels
    canvas, cx, cy = tile(samples, 8, 8)
    fig, ax = plt.subplots(figsize=(6.4, 6.6))
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
    ax.set_xticks(cx, [str(k) for k in range(1, 9)])
    ax.set_yticks(cy, [str(k) for k in range(1, 9)])
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_title("Figure 3 - 8x8 samples decode(z), z ~ N(0, I), d = 16\n"
                 "(drawn right after torch.manual_seed(0))")
    save(fig, "fig3.png")
    return z, samples
# --8<-- [end:fig3]


# --8<-- [start:fig4]
def figure4(mu2, y_test, window):
    fig, ax = plt.subplots(figsize=(8, 7))
    labels = y_test.numpy()
    # Draw the points in a fixed random order, so no digit is systematically
    # painted over another (drawing class by class would bury 4 under 9).
    order = np.random.default_rng(0).permutation(len(labels))
    colours = np.array(DIGIT_COLORS)[labels[order]]
    ax.scatter(mu2[order, 0], mu2[order, 1], s=3, alpha=0.5, c=colours)
    for c in range(10):                       # legend entries, one per digit
        ax.scatter([], [], s=12, color=DIGIT_COLORS[c], label=str(c))
    # The square covered by Figure 5's grid points, for reference.
    ax.add_patch(Rectangle((-window, -window), 2 * window, 2 * window, fill=False,
                           ls="--", lw=1.2, color="black", label="Fig. 5 grid"))
    ax.set_xlabel(r"$\mu_1$")
    ax.set_ylabel(r"$\mu_2$")
    ax.set_title(r"Figure 4 - encoder means $\mu$ of all 10 000 test images (d = 2)")
    ax.legend(title="digit", markerscale=4, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    save(fig, "fig4.png")
# --8<-- [end:fig4]


# --8<-- [start:fig5]
def figure5(model2):
    q = np.linspace(0.05, 0.95, 15)               # 15 evenly spaced prior quantiles
    g = norm.ppf(q)                               # z = Phi^{-1}(q), the same on both axes
    # Row r (top to bottom) has z2 = g[14 - r]; column c has z1 = g[c],
    # so the tiles keep the orientation of the scatter in Figure 4.
    z1, z2 = np.meshgrid(g, g[::-1])
    z = torch.tensor(np.stack([z1.ravel(), z2.ravel()], axis=1), dtype=torch.float32)
    with torch.no_grad():
        imgs = model2.decode(z)
    canvas, cx, cy = tile(imgs, 15, 15, pad=1)
    fmt = lambda v: f"{v:.2f}".replace("-0.00", "0.00")
    fig, ax = plt.subplots(figsize=(10, 10.5))
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
    ax.set_xticks(cx, [fmt(v) for v in g], rotation=90, fontsize=8)
    ax.set_yticks(cy, [fmt(v) for v in g[::-1]], fontsize=8)
    ax.set_xlabel(r"$z_1 = \Phi^{-1}(q_1)$,  $q_1$ = 0.05 ... 0.95 in 15 even steps")
    ax.set_ylabel(r"$z_2 = \Phi^{-1}(q_2)$")
    # Column / row numbers on the opposite sides, so cells can be cited by position.
    top = ax.secondary_xaxis("top")
    top.set_xticks(cx, [str(k) for k in range(1, 16)], fontsize=8)
    top.set_xlabel("column", fontsize=9)
    right = ax.secondary_yaxis("right")
    right.set_yticks(cy, [str(k) for k in range(1, 16)], fontsize=8)
    right.set_ylabel("row", fontsize=9)
    ax.set_title(r"Figure 5 - d = 2 decoder on the 15x15 grid $z = \Phi^{-1}(q)$", pad=12)
    save(fig, "fig5.png")
    return g, imgs
# --8<-- [end:fig5]


# --8<-- [start:fig6]
def figure6(model16, x_test, y_test):
    t = torch.linspace(0, 1, 10)[:, None]          # 10 values of t from 0 to 1
    xa, xb = x_test[IDX_A], x_test[IDX_B]
    pixel_row = (1 - t) * xa + t * xb              # straight line in pixel space
    with torch.no_grad():
        mu, _ = model16.encode(torch.stack([xa, xb]))
        z_path = (1 - t) * mu[0] + t * mu[1]       # straight line between the two means
        latent_row = model16.decode(z_path)        # ... decoded by the d = 16 model
    fig, axes = plt.subplots(2, 10, figsize=(13, 3.4))
    for j in range(10):
        for i, img in enumerate((pixel_row[j], latent_row[j])):
            ax = axes[i, j]
            ax.imshow(img.view(28, 28), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[0, j].set_title(f"t = {t[j, 0]:.2f}", fontsize=9)
    axes[0, 0].set_ylabel("pixel space", fontsize=10)
    axes[1, 0].set_ylabel("latent, d = 16", fontsize=10)
    fig.suptitle(f"Figure 6 - interpolating test #{IDX_A} (digit {int(y_test[IDX_A])}) -> "
                 f"test #{IDX_B} (digit {int(y_test[IDX_B])}): "
                 r"$(1-t)x_a + t\,x_b$ (top) vs decode$((1-t)\mu_a + t\,\mu_b)$ (bottom)",
                 y=1.04)
    save(fig, "fig6.png")
    return pixel_row, latent_row
# --8<-- [end:fig6]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    t_start = time.perf_counter()
    print(f"torch {torch.__version__}, {torch.get_num_threads()} CPU threads")

    # --8<-- [start:order]
    # ---------------- Exercise 1A: data ----------------
    train, val, test = load_split()
    train_xy, val_xy = flatten(train), flatten(val)
    print(f"[1A] train {tuple(train_xy[0].shape)}, val {tuple(val_xy[0].shape)}, "
          f"test {len(test)} images; pixel range [{train_xy[0].min():.0f}, {train_xy[0].max():.0f}]")

    # ---------------- Exercise 1B-C: the d = 16 VAE ----------------
    print("[1C] training d = 16")
    model16, h16 = fit_vae(16, train_xy, val_xy)
    print(f"[1C] training time d=16: {h16['seconds']:.1f} s")
    # Training is over: only now is the test set materialised and used.
    test_xy = flatten(test)
    x_test, y_test = test_xy
    rec16, kl16 = test_numbers(model16, test_xy, "1C d=16")
    # --8<-- [end:order]
    figure1(h16)

    # ---------------- Exercise 1D ----------------
    idx2, x_fig2, rec_fig2 = figure2(model16, x_test, y_test)
    print(f"[1D] Figure 2 test indices (digits 0-9): {idx2}")
    z_fig3, samples_fig3 = figure3(model16, 16)

    # ---------------- Exercise 1E: numbers behind the analysis ----------------
    vk = h16["val_kl"]
    print(f"[1E.1] val KL untrained (epoch 0) = {h16['init_val_kl']:.3f}, "
          f"val recon untrained = {h16['init_val_recon']:.2f}")
    print(f"[1E.1] val KL first epoch = {vk[0]:.2f}, last epoch = {vk[-1]:.2f}, "
          f"max = {max(vk):.2f} at epoch {int(np.argmax(vk)) + 1}")
    vr = h16["val_recon"]
    for e in range(1, EPOCHS):
        print(f"       epoch {e:2d}->{e + 1:2d}: d(val recon) = {vr[e] - vr[e - 1]:+7.2f}, "
              f"d(val KL) = {vk[e] - vk[e - 1]:+6.2f}")
    print(f"[1E.1] train KL epoch 1 = {h16['train_kl'][0]:.2f}, train recon epoch 1 = "
          f"{h16['train_recon'][0]:.2f}")

    with torch.no_grad():
        xv, _ = val_xy
        mu_v, lv_v = model16.encode(xv)
        var_mu = mu_v.var(dim=0)                          # spread of the means, per dim
        mean_var = lv_v.exp().mean(dim=0)                 # average posterior variance
        active = var_mu > 0.01
        print(f"[1E.2a] active latent dims (Var[mu_i] > 0.01): {int(active.sum())} of 16")
        print(f"[1E.2a] per-dim Var[mu_i]:   {np.round(var_mu.double().numpy(), 2).tolist()}")
        print(f"[1E.2a] per-dim E[sigma_i^2]: {np.round(mean_var.double().numpy(), 3).tolist()}")
        print(f"[1E.2a] aggregate-posterior variance per dim (Var[mu]+E[sigma^2]): "
              f"min {float((var_mu + mean_var).min()):.2f}, max {float((var_mu + mean_var).max()):.2f}")
        print(f"[1E.3] mean posterior sigma over active dims = "
              f"{float(lv_v[:, active].mul(0.5).exp().mean()):.3f}; "
              f"std of mu over active dims = {float(var_mu[active].sqrt().mean()):.3f}")
        # Aggregate posterior from half of validation; compare fresh codes of
        # the other half with draws from the prior.
        mu_a, lv_a = mu_v[:5000], lv_v[:5000]
        z_post = reparameterize(mu_v[5000:], lv_v[5000:])
        z_prior = torch.randn(5000, 16)
        lq_post = log_q_aggregate(z_post, mu_a, lv_a)
        lq_prior = log_q_aggregate(z_prior, mu_a, lv_a)
        lq_fig3 = log_q_aggregate(z_fig3, mu_a, lv_a)
        thr = torch.quantile(lq_post, 0.05)
        print(f"[1E.2a] median log q_agg(z): encoded val codes {lq_post.median():.1f}, "
              f"prior draws {lq_prior.median():.1f}, Fig-3 draws {lq_fig3.median():.1f}")
        print(f"[1E.2a] fraction below the 5th percentile of encoded codes: "
              f"prior draws {(lq_prior < thr).float().mean():.1%}, "
              f"Fig-3 draws {(lq_fig3 < thr).float().mean():.1%} "
              f"({int((lq_fig3 < thr).sum())} of 64)")
        low = [f"(r{k // 8 + 1},c{k % 8 + 1})" for k in torch.argsort(lq_fig3).tolist()
               if lq_fig3[k] < thr]
        print(f"[1E.2a] Fig-3 tiles below that threshold, lowest density first: {' '.join(low)}")
        # Between digits? Digit labels of the 10 nearest real codes (same 5 000 images).
        knn16 = KNeighborsClassifier(n_neighbors=10).fit(mu_a.numpy(), val_xy[1][:5000].numpy())
        share = lambda z: knn16.predict_proba(z.numpy()).max(axis=1)
        s_post, s_prior, s_fig3 = share(z_post), share(z_prior), share(z_fig3)
        print(f"[1E.2a] majority-digit share among the 10 nearest real codes (median): "
              f"encoded val codes {np.median(s_post):.0%}, prior draws {np.median(s_prior):.0%}")
        print(f"[1E.2a] no digit above 50% of the neighbours: encoded val codes "
              f"{np.mean(s_post <= 0.5):.1%}, prior draws {np.mean(s_prior <= 0.5):.1%}, "
              f"Fig-3 draws {np.mean(s_fig3 <= 0.5):.1%} ({int(np.sum(s_fig3 <= 0.5))} of 64)")
        pred3 = knn16.predict(z_fig3.numpy())
        amb = [f"(r{k // 8 + 1},c{k % 8 + 1}: {pred3[k]} {s_fig3[k]:.0%})"
               for k in np.argsort(s_fig3, kind="stable") if s_fig3[k] <= 0.5]
        print(f"[1E.2a] Fig-3 tiles with no majority digit (top label, share): {' '.join(amb)}")
        # What the decoder outputs: grey levels instead of black/white pixels.
        rec_v = model16.decode(mu_v)
        dec_prior = model16.decode(z_prior)
        print(f"[1E.2b] grey-pixel fraction (0.1<p<0.9): real val images "
              f"{gray_fraction(xv).mean():.1%}, decode(mu) val {gray_fraction(rec_v).mean():.1%}, "
              f"prior samples {gray_fraction(dec_prior).mean():.1%}, Fig-3 "
              f"{gray_fraction(samples_fig3).mean():.1%}, Fig-2 recon {gray_fraction(rec_fig2).mean():.1%}")
        print(f"[1E.2b] decoder Bernoulli entropy (nats/image): decode(mu) val "
              f"{bernoulli_entropy(rec_v).mean():.1f}, prior samples "
              f"{bernoulli_entropy(dec_prior).mean():.1f}, Fig-3 {bernoulli_entropy(samples_fig3).mean():.1f}")
        # Decode mu vs decode a sampled z (validation set).
        r_mu, kl_v = loss_terms(xv, rec_v, mu_v, lv_v)
        r_z, _ = loss_terms(xv, model16.decode(reparameterize(mu_v, lv_v)), mu_v, lv_v)
        print(f"[1E.3] val recon decoding mu = {r_mu:.2f}, decoding one sampled z = {r_z:.2f}")
        print(f"[2E.3] reference: val neg. ELBO decoding mu (recon + KL) = {r_mu + kl_v:.2f}")
        r_mu2, _ = loss_terms(x_fig2, rec_fig2, *model16.encode(x_fig2))
        print(f"[1E.3] recon of the 10 Figure-2 images from mu = {r_mu2:.2f}")

    # ---------------- Exercise 2A: the d = 2 VAE ----------------
    print("[2A] training d = 2")
    model2, h2 = fit_vae(2, train_xy, val_xy)
    print(f"[2A] training time d=2: {h2['seconds']:.1f} s")
    rec2, kl2 = test_numbers(model2, test_xy, "2A d=2")
    print(f"[2A] val KL d=2 first epoch = {h2['val_kl'][0]:.2f}, last = {h2['val_kl'][-1]:.2f}")
    print(f"[2A] information bound: test KL d=16 = {kl16 / np.log(2):.1f} bits, "
          f"d=2 = {kl2 / np.log(2):.1f} bits")

    # ---------------- Exercise 2B ----------------
    with torch.no_grad():
        mu2_test, _ = model2.encode(x_test)
        mu2_val, _ = model2.encode(val_xy[0])
        mu16_test, _ = model16.encode(x_test)
    window = float(norm.ppf(0.95))
    figure4(mu2_test.numpy(), y_test, window)
    inside = (mu2_test.abs() <= window).all(dim=1).float().mean()
    print(f"[2B] test mu inside the Fig-5 square |z_i| <= {window:.3f}: {inside:.1%}")
    print(f"[2B] std of mu per axis: {np.round(mu2_test.std(dim=0).double().numpy(), 2).tolist()}")

    # ---------------- Exercise 2C ----------------
    g, imgs5 = figure5(model2)
    print(f"[2C] grid values z = Phi^-1(q): {np.round(g, 3).tolist()}")
    print(f"[2C] prior mass per cell = (0.9/14)^2 = {(0.9 / 14) ** 2:.4%}")

    # ---------------- Exercise 2D ----------------
    pixel_row, latent_row = figure6(model16, x_test, y_test)
    print(f"[2D] interpolated test indices: a = {IDX_A} (digit {int(y_test[IDX_A])}), "
          f"b = {IDX_B} (digit {int(y_test[IDX_B])})")

    # ---------------- Exercise 2E: numbers behind the analysis ----------------
    # Overlap in Figure 4: for each test code, which digits are its 10 nearest neighbours?
    nn_ = NearestNeighbors(n_neighbors=11).fit(mu2_test.numpy())
    nbrs = nn_.kneighbors(mu2_test.numpy(), return_distance=False)[:, 1:]
    lab = y_test.numpy()
    conf = np.zeros((10, 10))
    for c in range(10):
        nb = lab[nbrs[lab == c]].ravel()
        conf[c] = np.bincount(nb, minlength=10) / nb.size
    pairs = sorted(((conf[a, b] + conf[b, a]) / 2, a, b) for a in range(10) for b in range(a + 1, 10))[::-1]
    print("[2E.1] most-mixed digit pairs in Fig. 4 (mean share of 10-NN from the other digit):")
    for s, a, b in pairs[:6]:
        print(f"       {a}-{b}: {s:.1%}  ({a}->{b} {conf[a, b]:.1%}, {b}->{a} {conf[b, a]:.1%})")
    print(f"[2E.1] own-digit share of 10-NN per digit: "
          f"{ {c: round(float(conf[c, c]), 3) for c in range(10)} }")
    # Which digits are encoded near each Figure-5 grid point? Majority label of
    # the 10 nearest validation codes, with its share (same layout as Fig. 5).
    knn2 = KNeighborsClassifier(n_neighbors=10).fit(mu2_val.numpy(), val_xy[1].numpy())
    z1, z2 = np.meshgrid(g, g[::-1])
    proba = knn2.predict_proba(np.stack([z1.ravel(), z2.ravel()], axis=1)).reshape(15, 15, 10)
    print("[2E.1] Fig-5 grid: majority digit of the 10 nearest val codes (share %), "
          "row 1 = top, column 1 = left")
    for r in range(15):
        cells = " ".join(f"{int(proba[r, c].argmax())}({proba[r, c].max():.0%})".ljust(8)
                         for c in range(15))
        print(f"       row {r + 1:2d} (z2={g[14 - r]:+.2f}): {cells}")
    for d_, mt, mv in ((2, mu2_test, mu2_val), (16, mu16_test, None)):
        if mv is None:
            with torch.no_grad():
                mv, _ = model16.encode(val_xy[0])
        knn = KNeighborsClassifier(n_neighbors=10).fit(mv.numpy(), val_xy[1].numpy())
        print(f"[2E.2] 10-NN digit accuracy on test mu (fit on val mu), d={d_}: "
              f"{knn.score(mt.numpy(), lab):.1%}")
    print(f"[2E.2] grey-pixel fraction of Fig-5 tiles {gray_fraction(imgs5).mean():.1%}, "
          f"Bernoulli entropy {bernoulli_entropy(imgs5).mean():.1f} nats")
    print(f"[2E.2] recon gap d=2 - d=16 = {rec2 - rec16:.2f} nats/image; "
          f"KL d=16 - d=2 = {kl16 - kl2:.2f} nats/image")
    def neg_elbo_per_frame(frames):
        """How typical the d=16 VAE finds each image: recon (decoding mu) + KL, in nats."""
        with torch.no_grad():
            mu_f, lv_f = model16.encode(frames)
            return [round(float(sum(loss_terms(frames[j:j + 1], model16.decode(mu_f[j:j + 1]),
                                               mu_f[j:j + 1], lv_f[j:j + 1]))), 1)
                    for j in range(len(frames))]
    print(f"[2E.3] grey-pixel fraction per frame, pixel row:  "
          f"{np.round(gray_fraction(pixel_row).double().numpy(), 3).tolist()}")
    print(f"[2E.3] grey-pixel fraction per frame, latent row: "
          f"{np.round(gray_fraction(latent_row).double().numpy(), 3).tolist()}")
    print(f"[2E.3] d=16 neg. ELBO (decode mu) of each pixel-row frame:  {neg_elbo_per_frame(pixel_row)}")
    print(f"[2E.3] d=16 neg. ELBO (decode mu) of each latent-row frame: {neg_elbo_per_frame(latent_row)}")
    ink_a, ink_b = x_test[IDX_A] > 0.5, x_test[IDX_B] > 0.5
    print(f"[2E.3] ink overlap |a & b| / |a | b| = {(ink_a & ink_b).sum() / (ink_a | ink_b).sum():.1%}")

    print(f"total run time: {time.perf_counter() - t_start:.1f} s "
          f"(training {h16['seconds'] + h2['seconds']:.1f} s)")


if __name__ == "__main__":
    main()
