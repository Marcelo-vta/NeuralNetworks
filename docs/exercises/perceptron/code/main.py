"""Exercise "2. Perceptron" -- single entry point.

Run from the repository root:

    python docs/exercises/perceptron/code/main.py

It creates the report's ONLY random generator, rng = np.random.default_rng(42),
and runs Exercise 1 and then Exercise 2 in statement order. Data generation and
weight initialization both draw from that rng, so every number printed below and
every figure in ../figures/ is regenerated identically on each run.
"""

from math import erf
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # write PNG files only; no window is ever opened
import matplotlib.pyplot as plt  # noqa: E402

from perceptron import Perceptron  # noqa: E402  (the from-scratch model, same file for both exercises)

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
SEED = 42
N_PER_CLASS = 1000
MAX_EPOCHS = 100

# One fixed color per class in every figure of the report.
CLASS_COLOR = {0: "#2a78d6", 1: "#eb6834"}
INK = "#222222"          # eta = 0.01 run / final weights
ALT_COLOR = "#6250d6"    # eta = 1.0 run (Exercise 1, item D2)
POCKET_COLOR = "#008300"  # pocket weights (Exercise 2)

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def out(tag, text):
    """Print one labelled result line; every number quoted in the report comes from here."""
    print(f"[{tag}] {text}")


def fmt_vec(v, nd=4):
    return "[" + ", ".join(f"{x:.{nd}f}" for x in v) + "]"


# --8<-- [start:data]
def make_two_gaussians(rng, mean0, mean1, cov, n_per_class):
    """Draw n_per_class 2-D points per class from N(mean_k, cov); return shuffled (X, y).

    X has shape (2 * n_per_class, 2); y holds the labels 0 / 1 as integers.
    """
    X0 = rng.multivariate_normal(mean0, cov, size=n_per_class)  # class 0 cloud
    X1 = rng.multivariate_normal(mean1, cov, size=n_per_class)  # class 1 cloud
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)])
    # The draw above lists all class-0 points before all class-1 points. Shuffle ONCE
    # (with the same rng) so the online perceptron does not see 1000 zeros and then
    # 1000 ones; training then visits the samples in this fixed order every epoch.
    order = rng.permutation(len(X))
    return X[order], y[order]
# --8<-- [end:data]


def describe_data(tag, X, y):
    """Print the sample mean / covariance of each class (sanity check of the parameters)."""
    for k in (0, 1):
        Xk = X[y == k]
        cov = np.cov(Xk.T)
        out(tag, f"class {k}: n = {len(Xk)}, sample mean = {fmt_vec(Xk.mean(axis=0), 3)}, "
                 f"sample cov = [[{cov[0, 0]:.3f}, {cov[0, 1]:.3f}], [{cov[1, 0]:.3f}, {cov[1, 1]:.3f}]]")
    out(tag, f"mean ||x|| over all points = {np.linalg.norm(X, axis=1).mean():.3f}")


# ----------------------------------------------------------------------------- hooks and diagnostics
# --8<-- [start:tracer]
class Tracer:
    """on_update hook that only RECORDS: (epoch, sample index, error, w1, w2, b) per update.

    It never changes the model, so a run with a Tracer follows exactly the same
    trajectory as a run without one.
    """

    def __init__(self):
        self.rows = []

    def __call__(self, model, epoch, i, error):
        self.rows.append((epoch, i, error, model.w[0], model.w[1], model.b))

    def array(self):
        return np.array(self.rows, dtype=float).reshape(-1, 6)
# --8<-- [end:tracer]


def state_accuracies(rows, X, y, chunk=2000):
    """Full-dataset accuracy of the weights stored after every update (vectorized)."""
    accs = []
    for s in range(0, len(rows), chunk):
        W, B = rows[s:s + chunk, 3:5], rows[s:s + chunk, 5]
        pred = Perceptron.step(X @ W.T + B)          # same activation as the model
        accs.append((pred == y[:, None]).mean(axis=0))
    return np.concatenate(accs) if accs else np.array([])


def scan_lines(X, y, n_dir=7200):
    """Reference values only (no training): scan line directions on a fine grid (n_dir steps).

    For each unit direction u, the best threshold on u . x is found by sorting.
    Returns (best accuracy any straight line reaches on this sample,
             largest margin, its direction u and threshold c). The margin is half the
    gap between the classes' projections: negative means no line separates them.
    """
    best_acc, best_gamma, best_u, best_c = 0.0, -np.inf, None, None
    n0 = (y == 0).sum()
    for th in np.linspace(0, 2 * np.pi, n_dir, endpoint=False):
        u = np.array([np.cos(th), np.sin(th)])
        p = X @ u
        ys = y[np.argsort(p)]
        # threshold after the k smallest projections: those k -> class 0, the rest -> class 1
        zeros_below = np.concatenate([[0], np.cumsum(ys == 0)])
        ones_above = (len(y) - n0) - (np.arange(len(y) + 1) - zeros_below)
        best_acc = max(best_acc, ((zeros_below + ones_above) / len(y)).max())
        lo1, hi0 = p[y == 1].min(), p[y == 0].max()
        if (lo1 - hi0) / 2 > best_gamma:
            best_gamma, best_u, best_c = (lo1 - hi0) / 2, u, (lo1 + hi0) / 2
    return best_acc, best_gamma, best_u, best_c


# ----------------------------------------------------------------------------- plotting helpers
# --8<-- [start:plot-helpers]
def scatter_classes(ax, X, y, s=10, alpha=0.55):
    """Scatter both classes with their fixed colors and legend labels."""
    for k in (0, 1):
        ax.scatter(X[y == k, 0], X[y == k, 1], s=s, alpha=alpha, color=CLASS_COLOR[k],
                   edgecolors="none", label=f"Class {k}")


def shade_regions(ax, w, b, xlim, ylim):
    """Lightly tint the half-plane predicted as each class by step(w . x + b)."""
    g1, g2 = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))
    pred = Perceptron(w, b).predict(np.c_[g1.ravel(), g2.ravel()]).reshape(g1.shape)
    ax.contourf(g1, g2, pred, levels=[-0.5, 0.5, 1.5],
                colors=[CLASS_COLOR[0], CLASS_COLOR[1]], alpha=0.08, zorder=0)


def draw_boundary(ax, w, b, xlim, ylim, **style):
    """Draw the line w1 x1 + w2 x2 + b = 0 clipped to the current axis window."""
    w1, w2 = w
    if abs(w2) >= abs(w1):                         # mostly horizontal: solve for x2
        x1 = np.array(xlim)
        x2 = -(w1 * x1 + b) / w2
    else:                                           # mostly vertical: solve for x1
        x2 = np.array(ylim)
        x1 = -(w2 * x2 + b) / w1
    ax.plot(x1, x2, **style)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def mark_misclassified(ax, X, y, w, b, s=26, lw=1.0, alpha=1.0, label_prefix="Misclassified"):
    """Overlay a black x on every point the weights (w, b) get wrong; return the count."""
    wrong = Perceptron(w, b).predict(X) != y
    ax.scatter(X[wrong, 0], X[wrong, 1], marker="x", s=s, linewidths=lw, color=INK,
               alpha=alpha, zorder=5, label=f"{label_prefix} ({wrong.sum()})")
    return int(wrong.sum())


def data_limits(X, pad=0.6):
    lo, hi = X.min(axis=0) - pad, X.max(axis=0) + pad
    lo_all, hi_all = min(lo), max(hi)               # same range on both axes (equal aspect)
    return (lo_all, hi_all), (lo_all, hi_all)


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)
# --8<-- [end:plot-helpers]


def plot_dataset(X, y, title, name, means):
    """Figures 1 and 4: the raw point clouds, one color per class, true means marked."""
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    scatter_classes(ax, X, y)
    for k, m in enumerate(means):
        ax.scatter(*m, marker="P", s=140, color=CLASS_COLOR[k], edgecolors=INK, linewidths=1.2,
                   zorder=6, label=f"True mean, class {k}: ({m[0]:g}, {m[1]:g})")
    xlim, ylim = data_limits(X)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    save(fig, name)


# ----------------------------------------------------------------------------- Exercise 1
def exercise1(rng):
    print("=" * 78)
    print("EXERCISE 1 -- separable data")
    print("=" * 78)

    # ---------------------------------------------------------------- A: data
    # --8<-- [start:ex1-data]
    cov1 = [[0.5, 0.0], [0.0, 0.5]]
    X, y = make_two_gaussians(rng, mean0=[1.5, 1.5], mean1=[5.0, 5.0], cov=cov1,
                              n_per_class=N_PER_CLASS)
    # --8<-- [end:ex1-data]
    describe_data("Ex1 A", X, y)
    plot_dataset(X, y, "Figure 1 — Exercise 1 data (separable)", "fig1.png",
                 means=[(1.5, 1.5), (5.0, 5.0)])

    # ---------------------------------------------------------------- C: train with eta = 0.01
    # --8<-- [start:ex1-train]
    w0 = rng.normal(0, 0.01, size=2)    # drawn ONCE; the eta = 1.0 run in D2 reuses this w0
    log = Tracer()                      # records which sample triggered each update (for D1)
    model = Perceptron().train(X, y, eta=0.01, max_epochs=MAX_EPOCHS, w0=w0, b0=0.0,
                               on_update=log)
    # --8<-- [end:ex1-train]
    acc = model.history["accuracy"]
    upd = model.history["updates"]
    acc0 = Perceptron(w0, 0.0).accuracy(X, y)
    out("Ex1 B", f"initial w0 = {fmt_vec(w0, 6)}, b0 = 0, ||w0|| = {np.linalg.norm(w0):.6f}, "
                 f"initial accuracy = {100 * acc0:.2f}%")
    out("Ex1 C", f"final w = {fmt_vec(model.w)}, final b = {model.b:.4f}")
    out("Ex1 C", f"epochs run = {model.epochs_} (last epoch with an update = {model.epochs_ - 1}; "
                 f"epoch {model.epochs_} is the update-free pass that stops training)")
    out("Ex1 C", f"final accuracy = {100 * acc[-1]:.2f}%")
    out("Ex1 C", "accuracy after each epoch = " + ", ".join(f"{100 * a:.2f}%" for a in acc))

    # ---------------------------------------------------------------- D1: updates shrink
    rows = log.array()
    pos = rows[:, 1] + 1                                     # 1-based position inside the epoch
    st_acc = state_accuracies(rows, X, y)
    out("Ex1 D1", f"updates per epoch = {upd} (total {sum(upd)})")
    windows = [(1, 100), (101, 500), (501, 1000), (1001, 2000)]
    counts = [int(((pos >= a) & (pos <= b) & (rows[:, 0] == 1)).sum()) for a, b in windows]
    out("Ex1 D1", "epoch 1 updates by sample position: " +
        ", ".join(f"{a}-{b}: {c}" for (a, b), c in zip(windows, counts)))
    out("Ex1 D1", f"last update at sample position {int(pos.max())} of epoch 1; "
                  f"accuracy after update 9 (position {int(pos[8])}) = {100 * st_acc[8]:.2f}%")
    # Group updates into bursts (consecutive updates less than 30 samples apart).
    breaks = np.flatnonzero(np.diff(pos) >= 30) + 1
    bursts = np.split(np.arange(len(pos)), breaks)
    out("Ex1 D1", f"{len(bursts)} bursts of updates: " + "; ".join(
        f"positions {int(pos[g[0]])}-{int(pos[g[-1]])}: {len(g)} updates, min acc inside "
        f"{100 * st_acc[g].min():.1f}%, acc after {100 * st_acc[g[-1]]:.1f}%" for g in bursts))
    dist = (2 * y - 1) * model.net_input(X) / np.linalg.norm(model.w)
    out("Ex1 D1", f"geometric margin of the eta=0.01 boundary (closest point) = {dist.min():.4f}")

    # ---------------------------------------------------------------- D2: same run, eta = 1.0
    # --8<-- [start:ex1-eta]
    log_big = Tracer()
    model_big = Perceptron().train(X, y, eta=1.0, max_epochs=MAX_EPOCHS, w0=w0, b0=0.0,
                                   on_update=log_big)   # same data, order, w0 and b0

    # Compare boundary DIRECTIONS: unit normal vectors and the angle between them.
    u_small = model.w / np.linalg.norm(model.w)
    u_big = model_big.w / np.linalg.norm(model_big.w)
    angle = np.degrees(np.arccos(np.clip(u_small @ u_big, -1.0, 1.0)))
    # --8<-- [end:ex1-eta]
    acc_big = model_big.history["accuracy"]
    rows_big = log_big.array()
    out("Ex1 D2", f"eta=1.0: final w = {fmt_vec(model_big.w)}, final b = {model_big.b:.4f}")
    out("Ex1 D2", f"eta=1.0: epochs run = {model_big.epochs_}, final accuracy = {100 * acc_big[-1]:.2f}%")
    out("Ex1 D2", f"eta=1.0: updates per epoch = {model_big.history['updates']}, last update at "
                  f"sample position {int(rows_big[:, 1].max()) + 1}")
    out("Ex1 D2", "eta=1.0: accuracy after each epoch = " + ", ".join(f"{100 * a:.2f}%" for a in acc_big))
    out("Ex1 D2", f"unit normal eta=0.01: w/||w|| = {fmt_vec(u_small)}")
    out("Ex1 D2", f"unit normal eta=1.0 : w/||w|| = {fmt_vec(u_big)}")
    out("Ex1 D2", f"angle between the two directions = {angle:.2f} deg")
    for name, m in (("eta=0.01", model), ("eta=1.0 ", model_big)):
        off = -m.b / np.linalg.norm(m.w)          # distance from the origin to the line
        t = -m.b / m.w.sum()                      # where the line crosses the diagonal x1 = x2
        out("Ex1 D2", f"{name}: ||w|| = {np.linalg.norm(m.w):.4f}, distance origin->line = {off:.4f}, "
                      f"crosses x1 = x2 at ({t:.4f}, {t:.4f})")
    out("Ex1 D2", f"effective init ||w0||/eta: eta=0.01 -> {np.linalg.norm(w0) / 0.01:.4f}, "
                  f"eta=1.0 -> {np.linalg.norm(w0) / 1.0:.4f}  (one update adds ||x||, mean "
                  f"{np.linalg.norm(X, axis=1).mean():.2f}, to w/eta)")

    # ---------------------------------------------------------------- D3: numerical check of the proof
    # --8<-- [start:ex1-zero]
    log_za, log_zb = Tracer(), Tracer()
    zero_a = Perceptron().train(X, y, eta=0.01, max_epochs=MAX_EPOCHS, w0=np.zeros(2), on_update=log_za)
    zero_b = Perceptron().train(X, y, eta=1.0, max_epochs=MAX_EPOCHS, w0=np.zeros(2), on_update=log_zb)
    ratio_w = zero_b.w / zero_a.w           # the proof predicts eta2 / eta1 = 100 exactly
    ratio_b = zero_b.b / zero_a.b
    same_updates = np.array_equal(log_za.array()[:, :3], log_zb.array()[:, :3])
    # --8<-- [end:ex1-zero]
    out("Ex1 D3", f"zero start, eta=0.01: w = {fmt_vec(zero_a.w, 6)}, b = {zero_a.b:.6f}, "
                  f"epochs = {zero_a.epochs_}, updates/epoch = {zero_a.history['updates']}")
    out("Ex1 D3", f"zero start, eta=1.0 : w = {fmt_vec(zero_b.w, 6)}, b = {zero_b.b:.6f}, "
                  f"epochs = {zero_b.epochs_}, updates/epoch = {zero_b.history['updates']}")
    out("Ex1 D3", f"ratio w(eta=1)/w(eta=0.01) = {fmt_vec(ratio_w, 10)}, ratio b = {ratio_b:.10f}, "
                  f"identical update sequences (epoch, sample, error): {same_updates}")
    same_as_zero = np.array_equal(log_zb.array()[:, :3], rows_big[:, :3])
    out("Ex1 D2", f"eta=1.0 run (w0 start) makes the same updates as the zero start: {same_as_zero}; "
                  f"w(eta=1, w0) - w(zero start) = {fmt_vec(model_big.w - zero_b.w, 6)} (= w0)")

    # Reference value for D1/Ex2-D2: Novikoff's bound (R / gamma)^2 with the widest-margin line.
    _, gamma, u_m, c_m = scan_lines(X, y)
    R = np.sqrt((np.linalg.norm(X, axis=1) ** 2 + 1).max())   # radius of the augmented points (x, 1)
    gamma_aug = gamma / np.sqrt(1 + c_m ** 2)                  # same line, unit vector in (w, b) space
    out("Ex1 D1", f"widest-margin line (scan): margin = {gamma:.4f}, normal = {fmt_vec(u_m)}; "
                  f"R = {R:.3f}, gamma_aug = {gamma_aug:.5f}, Novikoff bound (R/gamma)^2 = "
                  f"{(R / gamma_aug) ** 2:.0f} updates")

    # ---------------------------------------------------------------- Figure 2
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    xlim, ylim = data_limits(X)
    shade_regions(ax, model.w, model.b, xlim, ylim)
    scatter_classes(ax, X, y)
    draw_boundary(ax, model.w, model.b, xlim, ylim, color=INK, lw=2.2,
                  label="Boundary $w\\cdot x+b=0$, $\\eta=0.01$")
    draw_boundary(ax, model_big.w, model_big.b, xlim, ylim, color=ALT_COLOR, lw=1.6, ls="--",
                  label="Boundary with $\\eta=1.0$ (item D2)")
    n_wrong = mark_misclassified(ax, X, y, model.w, model.b, label_prefix="Misclassified, $\\eta=0.01$")
    ax.set_aspect("equal")
    ax.set_title(f"Figure 2 — Perceptron boundary, Exercise 1 (accuracy {100 * acc[-1]:.1f}%)")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    save(fig, "fig2.png")
    out("Ex1 C", f"misclassified points (eta=0.01) = {n_wrong}")

    # ---------------------------------------------------------------- Figure 3
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.6), gridspec_kw={"width_ratios": [1, 1.5]})
    ep = np.arange(0, len(acc) + 1)
    ax.plot(ep, 100 * np.array([acc0] + acc), marker="o", ms=9, lw=2.2, color=INK,
            label="$\\eta=0.01$ (item C)")
    ep_b = np.arange(0, len(acc_big) + 1)
    ax.plot(ep_b, 100 * np.array([acc0] + acc_big), marker="s", ms=5, lw=1.4, ls="--", color=ALT_COLOR,
            label="$\\eta=1.0$ (item D2)")
    upd_big = model_big.history["updates"]
    for k, e in enumerate(ep[1:]):
        # updates made DURING epoch e, written above its accuracy point
        ax.annotate(f"$\\eta=0.01$: {upd[k]} upd.\n$\\eta=1.0$: {upd_big[k]} upd.", (e, 100 * acc[k]),
                    textcoords="offset points", xytext=(0, 12), ha="center", va="bottom",
                    fontsize=9, color=INK)
    ax.set_xticks(ep)
    ax.set_xticklabels(["0\n(init)"] + [str(e) for e in ep[1:]])
    ax.set_xlim(-0.3, len(acc) + 0.4)
    ax.set_ylim(55, 112)
    ax.set_yticks(np.arange(60, 101, 10))
    ax.set_title("(a) Full-dataset accuracy after each epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="center right", fontsize=9)

    # (b) cumulative number of updates against the sample visit index, over all epochs run:
    # the curve only rises on mistakes, so a flat stretch = a run of correct predictions.
    n = len(X)
    for r, color, lab, ls in ((rows, INK, "$\\eta=0.01$", "-"), (rows_big, ALT_COLOR, "$\\eta=1.0$", "--")):
        visit = (r[:, 0] - 1) * n + r[:, 1] + 1             # global visit index of each update
        x_v = np.concatenate([[0], visit, [2 * n]])
        y_v = np.concatenate([[0], np.arange(1, len(r) + 1), [len(r)]])
        ax2.plot(x_v, y_v, drawstyle="steps-post", color=color, lw=2.0 if ls == "-" else 1.5, ls=ls,
                 label=f"{lab}: {len(r)} updates, last at visit {int(visit.max())}")
    ax2.axvline(n, color="#888888", lw=1, ls=":")
    ax2.text(n * 1.5, len(rows) * 0.55, "epoch 2:\n0 updates for both runs\n→ training stops",
             ha="center", fontsize=9, color=INK)
    ax2.text(n * 0.5, len(rows) * 1.08, "epoch 1", ha="center", fontsize=9, color=INK)
    ax2.set_xlim(0, 2 * n)
    ax2.set_ylim(0, len(rows) * 1.18)
    ax2.set_title("(b) Cumulative updates during training")
    ax2.set_xlabel("Sample visits (epoch 1 = visits 1–2000, epoch 2 = 2001–4000)")
    ax2.set_ylabel("Cumulative number of updates")
    ax2.legend(loc="lower right", fontsize=9)
    fig.suptitle("Figure 3 — Accuracy × epoch, Exercise 1 (separable data)", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig3.png")


# ----------------------------------------------------------------------------- Exercise 2
# --8<-- [start:pocket]
class Pocket:
    """Pocket algorithm as an on_update hook: keep the best (w, b) seen so far.

    Perceptron.train calls this object after every update. It measures the
    full-dataset accuracy of the CURRENT weights and, if it beats every accuracy
    seen before, copies (w, b) into the pocket. The Perceptron class is untouched.
    """

    def __init__(self, X, y, w0, b0):
        self.X, self.y = X, y
        # The starting weights count as "seen": the pocket starts with them.
        self.w = np.asarray(w0, dtype=float).copy()
        self.b = float(b0)
        self.acc = Perceptron(w0, b0).accuracy(X, y)
        self.epoch = 0                      # epoch in which the pocket was last filled
        self.n_updates = 0                  # updates seen so far (all epochs)
        self.update_index = 0               # global index of the update that filled it
        self.improvements = [(0, self.acc)]

    def __call__(self, model, epoch, i, error):
        self.n_updates += 1
        acc = model.accuracy(self.X, self.y)        # accuracy of the weights right after this update
        if acc > self.acc:                          # strictly better than anything seen before
            self.w, self.b = model.w.copy(), model.b   # <- the only addition: copy into the pocket
            self.acc, self.epoch, self.update_index = acc, epoch, self.n_updates
            self.improvements.append((epoch, acc))

    def best_by_epoch(self, n_epochs):
        """Best-so-far (pocket) accuracy at the end of each epoch 0..n_epochs."""
        best, j, curve = self.improvements[0][1], 0, []
        for e in range(0, n_epochs + 1):
            while j < len(self.improvements) and self.improvements[j][0] <= e:
                best = max(best, self.improvements[j][1])
                j += 1
            curve.append(best)
        return np.array(curve)
# --8<-- [end:pocket]


def exercise2(rng):
    print("=" * 78)
    print("EXERCISE 2 -- overlapping data")
    print("=" * 78)

    # ---------------------------------------------------------------- A: data
    # --8<-- [start:ex2-data]
    cov2 = [[1.5, 0.0], [0.0, 1.5]]
    X, y = make_two_gaussians(rng, mean0=[3.0, 3.0], mean1=[4.0, 4.0], cov=cov2,
                              n_per_class=N_PER_CLASS)
    # --8<-- [end:ex2-data]
    describe_data("Ex2 A", X, y)
    # Population optimum for equal isotropic covariances: the perpendicular bisector scores
    # Phi(||mu1 - mu0|| / (2 sigma)), Phi = standard normal CDF.
    z_bayes = np.sqrt(2.0) / (2 * np.sqrt(1.5))
    out("Ex2 A", f"population accuracy of the best line = Phi({z_bayes:.3f}) = "
                 f"{100 * 0.5 * (1 + erf(z_bayes / np.sqrt(2))):.1f}%")
    plot_dataset(X, y, "Figure 4 — Exercise 2 data (overlapping)", "fig4.png",
                 means=[(3.0, 3.0), (4.0, 4.0)])

    # ---------------------------------------------------------------- B: train + pocket
    # --8<-- [start:ex2-train]
    w0 = rng.normal(0, 0.01, size=2)            # Exercise 2's own draw from the same rng
    pocket = Pocket(X, y, w0, b0=0.0)
    model = Perceptron().train(X, y, eta=0.01, max_epochs=MAX_EPOCHS, w0=w0, b0=0.0,
                               on_update=pocket)  # same class, same call; only the hook is new
    final_acc = model.accuracy(X, y)
    pocket_acc = Perceptron(pocket.w, pocket.b).accuracy(X, y)   # re-evaluated independently
    # --8<-- [end:ex2-train]
    acc = np.array(model.history["accuracy"])
    upd = np.array(model.history["updates"])
    best_curve = pocket.best_by_epoch(model.epochs_)
    acc0 = pocket.improvements[0][1]
    out("Ex2 B", f"initial w0 = {fmt_vec(w0, 6)}, b0 = 0, initial accuracy = {100 * acc0:.2f}%")
    out("Ex2 B", f"epochs run = {model.epochs_} (stopping rule never triggered: min updates/epoch = {upd.min()})")
    out("Ex2 B", f"FINAL  weights: w = {fmt_vec(model.w)}, b = {model.b:.4f}, accuracy = {100 * final_acc:.2f}% "
                 f"({(model.predict(X) != y).sum()} misclassified)")
    out("Ex2 B", f"POCKET weights: w = {fmt_vec(pocket.w)}, b = {pocket.b:.4f}, accuracy = {100 * pocket_acc:.2f}% "
                 f"({(Perceptron(pocket.w, pocket.b).predict(X) != y).sum()} misclassified)")
    out("Ex2 B", f"pocket best found in epoch {pocket.epoch} (update #{pocket.update_index} of "
                 f"{pocket.n_updates}); pocket improved {len(pocket.improvements) - 1} times")
    out("Ex2 B", f"updates per epoch: first 5 = {upd[:5].tolist()}, last 5 = {upd[-5:].tolist()}, "
                 f"mean = {upd.mean():.1f}, min = {upd.min()}, max = {upd.max()}")
    best_line_acc, gamma2, _, _ = scan_lines(X, y)
    out("Ex2 B", f"reference: best accuracy of ANY straight line on this sample (scan of 7200 directions) = "
                 f"{100 * best_line_acc:.2f}% ({round((1 - best_line_acc) * len(y))} points still wrong); "
                 f"widest 'margin' = {gamma2:.3f} (< 0: not separable)")
    out("Ex2 C", f"end-of-epoch accuracy of current weights: mean = {100 * acc.mean():.2f}%, "
                 f"min = {100 * acc.min():.2f}% (epoch {acc.argmin() + 1}), "
                 f"max = {100 * acc.max():.2f}% (epoch {acc.argmax() + 1})")
    out("Ex2 C", "pocket (best-so-far) accuracy at end of epochs 1, 5, 10, 40, 100 = "
                 + ", ".join(f"{100 * best_curve[e]:.2f}%" for e in (1, 5, 10, 40, 100)))

    # ---------------------------------------------------------------- D1: where do the boundaries sit?
    center = X.mean(axis=0)
    norms = np.linalg.norm(X, axis=1)
    proj_std = np.std(X @ np.array([1.0, 1.0]) / np.sqrt(2))
    out("Ex2 D1", f"data center c = {fmt_vec(center, 3)} (||c|| = {np.linalg.norm(center):.3f}); "
                  f"mean ||x|| = {norms.mean():.3f}; per mistake |db| = 0.01 but ||dw|| = 0.01 ||x|| "
                  f"~ {0.01 * norms.mean():.4f} ({norms.mean():.2f} times larger)")
    out("Ex2 D1", f"one update changes the score at the center, w.c + b, by 0.01 (x.c + 1): "
                  f"mean |change| = {0.01 * np.abs(X @ center + 1).mean():.3f}; "
                  f"cloud std along (1,1)/sqrt(2) = {proj_std:.3f}")
    for name, w, b in (("final ", model.w, model.b), ("pocket", pocket.w, pocket.b)):
        p = Perceptron(w, b)
        signed = (w @ center + b) / np.linalg.norm(w)      # > 0: the center lies on the class-1 side
        t = -b / w.sum()                                    # crossing of the diagonal x1 = x2
        u = w / np.linalg.norm(w)
        tilt = np.degrees(np.arccos(abs(u @ np.array([1.0, 1.0])) / np.sqrt(2)))  # normal vs. diagonal
        out("Ex2 D1", f"{name}: normal tilted {tilt:.1f} deg from the diagonal direction (1,1)")
        out("Ex2 D1", f"{name}: predicts class 1 for {100 * p.predict(X).mean():.1f}% of points "
                      f"(recall class 0 = {100 * (p.predict(X[y == 0]) == 0).mean():.1f}%, class 1 = "
                      f"{100 * (p.predict(X[y == 1]) == 1).mean():.1f}%); signed distance center->line = "
                      f"{signed:.3f} ({signed / proj_std:.2f} std); crosses x1 = x2 at ({t:.3f}, {t:.3f}); "
                      f"||w|| = {np.linalg.norm(w):.4f}; w/||w|| = {fmt_vec(w / np.linalg.norm(w))}")

    # Diagnostic replay: training is deterministic, so re-running it with a recording hook
    # reproduces exactly the same trajectory (asserted) without touching the pocket run.
    tracer = Tracer()
    replay = Perceptron().train(X, y, eta=0.01, max_epochs=MAX_EPOCHS, w0=w0, b0=0.0, on_update=tracer)
    assert np.array_equal(replay.w, model.w) and replay.b == model.b
    tr = tracer.array()                                     # epoch, i, error, w1, w2, b
    st_acc = state_accuracies(tr, X, y)
    assert abs(st_acc.max() - pocket.acc) < 1e-12          # the pocket holds the best visited state
    w_norm = np.hypot(tr[:, 3], tr[:, 4])
    d_center = (tr[:, 3:5] @ center + tr[:, 5]) / w_norm    # signed distance center -> line
    last = tr[:, 0] == MAX_EPOCHS
    out("Ex2 D1", f"all {len(tr)} updates: false-negative corrections (error +1) = {(tr[:, 2] > 0).sum()}, "
                  f"false-positive corrections (error -1) = {(tr[:, 2] < 0).sum()}")
    out("Ex2 D1", f"epoch {MAX_EPOCHS}: b stays in [{tr[last, 5].min():.2f}, {tr[last, 5].max():.2f}] "
                  f"while ||w|| swings over [{w_norm[last].min():.4f}, {w_norm[last].max():.4f}]")
    out("Ex2 D1", f"epoch {MAX_EPOCHS}: signed distance center->line, 5th-95th pct = "
                  f"[{np.percentile(d_center[last], 5):.2f}, {np.percentile(d_center[last], 95):.2f}], "
                  f"median |jump| per update = {np.median(np.abs(np.diff(d_center[last]))):.2f}")
    origin_dist = -tr[:, 5] / w_norm                        # distance origin -> line, -b / ||w||
    out("Ex2 D1", f"epoch {MAX_EPOCHS}: distance origin->line (-b/||w||), 5th-95th pct = "
                  f"[{np.percentile(origin_dist[last], 5):.2f}, {np.percentile(origin_dist[last], 95):.2f}] "
                  f"(data center at ||c|| = {np.linalg.norm(center):.2f})")
    out("Ex2 D1", f"accuracy of the weights after each of the {len(st_acc)} updates: median = "
                  f"{100 * np.median(st_acc):.1f}%, 5th pct = {100 * np.percentile(st_acc, 5):.1f}%, "
                  f"95th pct = {100 * np.percentile(st_acc, 95):.1f}%, max = {100 * st_acc.max():.2f}%; "
                  f"share below 55% = {100 * (st_acc < 0.55).mean():.1f}%, share >= 70% = "
                  f"{100 * (st_acc >= 0.70).mean():.1f}%")

    # Order diagnostic: the same shuffled samples re-sorted by class (stable sort, so the
    # within-class order is the shuffled one), then trained with the same w0.
    srt = np.argsort(y, kind="stable")                      # all class 0, then all class 1
    blocky = Perceptron().train(X[srt], y[srt], eta=0.01, max_epochs=MAX_EPOCHS, w0=w0, b0=0.0)
    out("Ex2 D1", f"check, class-ordered visits (0s then 1s): final accuracy = "
                  f"{100 * blocky.accuracy(X, y):.2f}%, predicts class 1 for "
                  f"{100 * blocky.predict(X).mean():.1f}% of points")
    t_blk = -blocky.b / blocky.w.sum()                      # crossing of the diagonal x1 = x2
    out("Ex2 D1", f"check, class-ordered visits: final w = {fmt_vec(blocky.w)}, b = {blocky.b:.4f}; "
                  f"crosses x1 = x2 at ({t_blk:.2f}, {t_blk:.2f}); distance origin->line = "
                  f"{-blocky.b / np.linalg.norm(blocky.w):.2f}; signed distance center->line = "
                  f"{(blocky.w @ center + blocky.b) / np.linalg.norm(blocky.w):.2f}")
    upd_fn, upd_fp = (tr[:, 2] > 0).sum() / MAX_EPOCHS, (tr[:, 2] < 0).sum() / MAX_EPOCHS
    out("Ex2 D1", f"shuffled run, per epoch on average: {upd_fn:.1f} corrections on class-1 points (+1), "
                  f"{upd_fp:.1f} on class-0 points (-1)")

    # ---------------------------------------------------------------- D3: confirmation runs (not the argument)
    out("Ex2 D3", f"||w|| over all 100 epochs: max = {w_norm.max():.4f}, mean in epochs 1-50 = "
                  f"{w_norm[tr[:, 0] <= 50].mean():.4f}, mean in epochs 51-100 = "
                  f"{w_norm[tr[:, 0] > 50].mean():.4f}")
    long_run = Perceptron().train(X, y, eta=0.01, max_epochs=1000, w0=w0, b0=0.0)
    la = np.array(long_run.history["accuracy"])
    out("Ex2 D3", f"check, 1000 epochs, eta=0.01: final accuracy = {100 * la[-1]:.2f}%, "
                  f"epochs 901-1000 mean = {100 * la[900:].mean():.2f}% "
                  f"[min {100 * la[900:].min():.2f}%, max {100 * la[900:].max():.2f}%], "
                  f"updates in epoch 1000 = {long_run.history['updates'][-1]}")
    small = Perceptron().train(X, y, eta=0.001, max_epochs=MAX_EPOCHS, w0=w0, b0=0.0)
    sa = np.array(small.history["accuracy"])
    out("Ex2 D3", f"check, eta=0.001, 100 epochs: final accuracy = {100 * sa[-1]:.2f}%, "
                  f"epochs 51-100 mean = {100 * sa[50:].mean():.2f}% "
                  f"[min {100 * sa[50:].min():.2f}%, max {100 * sa[50:].max():.2f}%], "
                  f"updates in epoch 100 = {small.history['updates'][-1]}")

    # ---------------------------------------------------------------- Figure 5: both boundaries
    xlim, ylim = data_limits(X, pad=0.4)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.6), sharex=True, sharey=True)
    panels = (("(a) Final weights, after epoch 100", model.w, model.b, final_acc, "final"),
              (f"(b) Pocket weights, found in epoch {pocket.epoch}", pocket.w, pocket.b, pocket_acc, "pocket"))
    for ax, (title, w, b, a, which) in zip(axes, panels):
        shade_regions(ax, w, b, xlim, ylim)
        scatter_classes(ax, X, y, s=9, alpha=0.5)
        mark_misclassified(ax, X, y, w, b, s=12, lw=0.6, alpha=0.6,
                           label_prefix=f"Misclassified by the {which} weights")
        draw_boundary(ax, model.w, model.b, xlim, ylim, color=INK,
                      lw=2.6 if which == "final" else 1.5, ls="-" if which == "final" else ":",
                      label="Final boundary")
        draw_boundary(ax, pocket.w, pocket.b, xlim, ylim, color=POCKET_COLOR,
                      lw=2.6 if which == "pocket" else 1.5, ls="-" if which == "pocket" else ":",
                      label="Pocket boundary")
        ax.scatter(*center, marker="*", s=150, color="white", edgecolors=INK, linewidths=1.2, zorder=7,
                   label="Data center")
        ax.set_aspect("equal")
        ax.set_title(f"{title}: accuracy {100 * a:.2f}%")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92, markerscale=1.4)
    fig.suptitle("Figure 5 — Final vs. pocket decision boundaries, Exercise 2 (overlapping data)",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig5.png")

    # ---------------------------------------------------------------- Figure 6: accuracy curves
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    ep = np.arange(0, model.epochs_ + 1)
    lo = [np.percentile(st_acc[tr[:, 0] == e], 5) for e in ep[1:]]
    hi = [np.percentile(st_acc[tr[:, 0] == e], 95) for e in ep[1:]]
    ax.fill_between(ep[1:], 100 * np.array(lo), 100 * np.array(hi), color=INK, alpha=0.10, lw=0,
                    label="Current weights after each update (5–95% range within the epoch)")
    ax.plot(ep, 100 * np.concatenate([[acc0], acc]), lw=1.4, color=INK, marker="o", ms=3,
            label="Current weights, end of each epoch")
    ax.plot(ep, 100 * best_curve, lw=2.6, color=POCKET_COLOR, drawstyle="steps-post",
            label="Best so far (pocket)")
    ax.axhline(50, color="#888888", lw=1, ls="--", label="Chance level (50%)")
    ax.scatter([pocket.epoch], [100 * pocket.acc], s=90, color=POCKET_COLOR, edgecolors=INK, zorder=6,
               label=f"Pocket best: {100 * pocket.acc:.2f}% (epoch {pocket.epoch})")
    ax.set_xlim(-1, model.epochs_ + 1)
    ax.set_ylim(45, 80)
    ax.set_title("Figure 6 — Accuracy against epoch, Exercise 2 (overlapping data)")
    ax.set_xlabel("Epoch (0 = initial weights)")
    ax.set_ylabel("Full-dataset accuracy (%)")
    ax.legend(loc="upper center", fontsize=9, framealpha=0.92, ncol=2)
    save(fig, "fig6.png")


def main():
    # --8<-- [start:main]
    rng = np.random.default_rng(SEED)   # the report's single generator, shared by both exercises
    exercise1(rng)                      # draws: Ex1 data (2 x 1000 points + shuffle), then Ex1 w0
    exercise2(rng)                      # draws: Ex2 data (2 x 1000 points + shuffle), then Ex2 w0
    # --8<-- [end:main]
    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
