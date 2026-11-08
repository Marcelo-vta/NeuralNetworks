"""Single entry point for the "3. MLP" report.

Run from the repository root:
    python docs/exercises/mlp/code/main.py

It prints every number quoted in the report (one labelled line each) and writes
Figures 1-6 to docs/exercises/mlp/figures/. Order of execution:

    Exercise 1  numerical check of the hand derivation (check_backprop.py)
    Exercise 2  binary blobs: baseline, MLP [2, h, 1] sigmoid + BCE
    Exercise 3  3-class make_classification: baseline, MLP [4, h, 3] softmax + CE
    Exercise 4  same data, one extra hidden layer, multi-seed comparison

Seed policy: ONE generator, rng = np.random.default_rng(42), is created in main()
and passed down in that order; every weight initialisation of Exercises 2-4 draws
from it. The datasets use random_state=42 exactly as the statement gives them.
The only exception is the multi-seed comparison of Exercise 4 (the statement
asks for several seeds there): both architectures are initialised with the same
explicit seeds, SEEDS = 42..46. Full-batch gradient descent has no other source
of randomness, so the initial weights are the only thing a seed changes.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # write PNG files; no window needed
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from sklearn.datasets import make_blobs, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import check_backprop
from mlp import MLP, gradient_check, n_parameters, one_hot

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
DPI = 150
# One fixed colour per class in every figure; one per network in the loss plots.
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]     # class 0 blue, 1 orange, 2 aqua
NET_COLORS = {1: "#52514e", 2: "#6250d6"}           # 1 hidden layer gray, 2 violet
SEEDS = [42, 43, 44, 45, 46]                        # Exercise 4 multi-seed comparison


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] saved figures/{name}")


def header(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


# --8<-- [start:helpers]
def standardizer(X_ref):
    """Return a function that standardises with the mean/std of X_ref (the TRAIN set).

    tanh saturates for |z| >> 1, so inputs of unit scale keep the first layer in
    its responsive range. Test data is transformed with the train statistics,
    never its own.
    """
    mu, sd = X_ref.mean(axis=0), X_ref.std(axis=0)
    return lambda X: (X - mu) / sd


def targets(y, output, n_out):
    """Labels -> target matrix: column vector for sigmoid, one-hot for softmax."""
    return y[:, None].astype(float) if output == "sigmoid" else one_hot(y, n_out)


def select_hyperparameters(X_tr, y_tr, n_in, n_out, output, widths, lrs, epochs, rng):
    """Small grid search on a validation split carved out of the TRAIN set.

    The test set is never used here; it is touched once, by the final model.
    Selection rule: lowest validation loss (the cross-entropy being optimised).
    It varies smoothly, whereas accuracy on a few hundred points ties often.
    """
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=0.25, random_state=42, stratify=y_tr)
    scale = standardizer(X_fit)                       # statistics of the fit part only
    rows = []
    for width in widths:
        for lr in lrs:
            model = MLP([n_in, width, n_out], output=output, rng=rng)
            model.fit(scale(X_fit), targets(y_fit, output, n_out), epochs, lr)
            val_loss = model.loss(model.forward(scale(X_val)), targets(y_val, output, n_out))
            val_acc = np.mean(model.predict(scale(X_val)) == y_val)
            rows.append((width, lr, val_acc, val_loss))
            print(f"    validation  hidden={width:>2}  lr={lr:<4}  "
                  f"val_acc={val_acc:.4f}  val_loss={val_loss:.4f}")
    best = min(rows, key=lambda r: r[3])
    print(f"    selected    hidden={best[0]}  lr={best[1]}  "
          f"({len(X_fit)} fit / {len(X_val)} validation samples)")
    return best[0], best[1]
# --8<-- [end:helpers]


# =============================================================================
# Exercise 2 - binary classification a straight line cannot do
# =============================================================================
def exercise2(rng):
    header("EXERCISE 2 - binary blobs")

    # --8<-- [start:ex2_data]
    # A - the statement's generator, unchanged: class 0 = one cluster at the origin,
    # class 1 = two clusters at (3, 3) and (-3, -3), on opposite sides of class 0.
    centers = np.array([[0., 0.], [3., 3.], [-3., -3.]])
    X, c = make_blobs(n_samples=[500, 250, 250], centers=centers,
                      cluster_std=1.2, random_state=42)
    y = (c > 0).astype(int)      # class 0: 1 cluster | class 1: 2 clusters

    # 80/20 stratified split. The cluster id c rides along only for the analysis
    # in item D; adding an array to the call does not change the split indices.
    X_tr, X_te, y_tr, y_te, c_tr, c_te = train_test_split(
        X, y, c, test_size=0.2, random_state=42, stratify=y)
    scale = standardizer(X_tr)   # train mean/std, applied to train, test and grid
    # --8<-- [end:ex2_data]
    print(f"[Ex2-A] samples: total={len(X)}  train={len(X_tr)}  test={len(X_te)}")
    print(f"[Ex2-A] class counts: all={np.bincount(y)}  train={np.bincount(y_tr)}  "
          f"test={np.bincount(y_te)}")
    print(f"[Ex2-A] test points per cluster (0,0)/(3,3)/(-3,-3): {np.bincount(c_te)}")
    print(f"[Ex2-A] class means (train): class0={X_tr[y_tr == 0].mean(0).round(3)}  "
          f"class1={X_tr[y_tr == 1].mean(0).round(3)}")

    # Figure 1 - the 1000 points coloured by class
    fig, ax = plt.subplots(figsize=(6, 5.4))
    for k in (0, 1):
        ax.scatter(X[y == k, 0], X[y == k, 1], s=12, alpha=0.65, color=CLASS_COLORS[k],
                   edgecolors="none", label=f"Class {k} (n={np.sum(y == k)})")
    ax.scatter(centers[:, 0], centers[:, 1], marker="+", s=160, color="black",
               linewidths=2, label="cluster centres")
    ax.set(title="Figure 1 - make_blobs data (1000 points) coloured by class",
           xlabel="feature $x_1$", ylabel="feature $x_2$", aspect="equal")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(alpha=0.25)
    save(fig, "fig1.png")

    # --8<-- [start:ex2_baseline]
    # B - linear baseline: one straight boundary, fitted on the same standardised
    # inputs the MLP sees.
    baseline = LogisticRegression().fit(scale(X_tr), y_tr)
    base_pred = baseline.predict(scale(X_te))
    base_acc = np.mean(base_pred == y_te)
    # --8<-- [end:ex2_baseline]
    w, b0 = baseline.coef_[0], baseline.intercept_[0]
    proba = baseline.predict_proba(scale(X_te))[:, 1]
    print(f"[Ex2-B] logistic regression test accuracy = {base_acc:.4f}")
    print(f"[Ex2-B] logistic regression train accuracy = "
          f"{np.mean(baseline.predict(scale(X_tr)) == y_tr):.4f}")
    print(f"[Ex2-B] LR weights (standardised inputs) = {w.round(4)}, bias = {b0:.4f}")
    print(f"[Ex2-B] LR test P(class 1) ranges over [{proba.min():.3f}, {proba.max():.3f}]")
    for k, name in enumerate(["(0,0) class 0", "(3,3) class 1", "(-3,-3) class 1"]):
        m = c_te == k
        print(f"[Ex2-B] LR on test cluster {name:<16}: n={m.sum():>3}  "
              f"predicted class 1 = {np.mean(base_pred[m] == 1):.2f}  "
              f"accuracy = {np.mean(base_pred[m] == y_te[m]):.2f}")

    # --8<-- [start:ex2_train]
    # C - choose width and learning rate on a validation split of the train set,
    # then retrain on the whole train set and evaluate on the test set once.
    epochs = 2000
    width, lr = select_hyperparameters(X_tr, y_tr, n_in=2, n_out=1, output="sigmoid",
                                       widths=[4, 8, 16], lrs=[0.1, 0.5],
                                       epochs=epochs, rng=rng)
    layers = [2, width, 1]
    model = MLP(layers, hidden_activation="tanh", output="sigmoid", rng=rng)
    history = model.fit(scale(X_tr), targets(y_tr, "sigmoid", 1), epochs=epochs, lr=lr)
    test_acc = np.mean(model.predict(scale(X_te)) == y_te)
    # --8<-- [end:ex2_train]
    train_acc = np.mean(model.predict(scale(X_tr)) == y_tr)
    gc = gradient_check(model, scale(X_tr[:100]), targets(y_tr[:100], "sigmoid", 1))
    print(f"[Ex2-C] architecture = {layers}, tanh hidden, sigmoid output, BCE loss")
    print(f"[Ex2-C] parameters = {model.n_parameters()}")
    print(f"[Ex2-C] learning rate = {lr}, epochs = {epochs} (full batch)")
    print(f"[Ex2-C] train loss: epoch 1 = {history[0]:.4f}, epoch 100 = {history[99]:.4f}, "
          f"epoch 500 = {history[499]:.4f}, final = {history[-1]:.4f}")
    print(f"[Ex2-C] train accuracy = {train_acc:.4f}")
    print(f"[Ex2-C] test accuracy  = {test_acc:.4f}")
    print(f"[Ex2-C] gradient check (sigmoid+BCE): max |analytic - numeric| = {gc:.2e}")

    # Figure 2 - training loss per epoch
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(np.arange(1, epochs + 1), history, color=NET_COLORS[1], lw=2,
            label=f"MLP {layers}, lr = {lr}")
    ax.set(title="Figure 2 - Exercise 2 training loss (binary cross-entropy)",
           xlabel="epoch", ylabel="training loss (BCE)")
    ax.legend()
    ax.grid(alpha=0.25)
    save(fig, "fig2.png")

    # --8<-- [start:ex2_grid]
    # Figure 3 - evaluate the network on a dense grid covering the test points and
    # shade the two predicted regions.
    pad = 1.0
    g1 = np.linspace(X_te[:, 0].min() - pad, X_te[:, 0].max() + pad, 400)
    g2 = np.linspace(X_te[:, 1].min() - pad, X_te[:, 1].max() + pad, 400)
    G1, G2 = np.meshgrid(g1, g2)
    grid = np.column_stack([G1.ravel(), G2.ravel()])
    P = model.predict_proba(scale(grid))[:, 0].reshape(G1.shape)   # P(class 1)
    # --8<-- [end:ex2_grid]
    te_pred = model.predict(scale(X_te))
    wrong = te_pred != y_te

    fig, ax = plt.subplots(figsize=(6.6, 6))
    ax.contourf(G1, G2, P, levels=[0, 0.5, 1], alpha=0.22,
                cmap=ListedColormap(CLASS_COLORS[:2]))
    ax.contour(G1, G2, P, levels=[0.5], colors="black", linewidths=1.5)
    for k in (0, 1):
        m = y_te == k
        ax.scatter(X_te[m, 0], X_te[m, 1], s=18, color=CLASS_COLORS[k],
                   edgecolors="white", linewidths=0.4)
    ax.scatter(X_te[wrong, 0], X_te[wrong, 1], marker="x", s=40, color="black", linewidths=1.3)
    # The logistic-regression line w . (x - mu)/sd + b = 0, drawn in raw coordinates.
    mu, sd = X_tr.mean(0), X_tr.std(0)
    a1, a2 = w / sd
    const = b0 - np.dot(w, mu / sd)
    ax.plot(-(a2 * g2 + const) / a1, g2, ls="--", color="#0b0b0b", lw=1.4)
    print(f"[Ex2-D] LR line in raw coordinates: x1 = {-const / a1:.3f} {-a2 / a1:+.3f} * x2 "
          f"(crosses x2 = 0 at x1 = {-const / a1:.3f})")
    ax.set(xlim=(g1[0], g1[-1]), ylim=(g2[0], g2[-1]), aspect="equal",
           title=f"Figure 3 - MLP {layers} decision regions over the 200 test points",
           xlabel="feature $x_1$", ylabel="feature $x_2$")
    handles = [Patch(color=CLASS_COLORS[0], alpha=0.35, label="region predicted class 0"),
               Patch(color=CLASS_COLORS[1], alpha=0.35, label="region predicted class 1"),
               Line2D([], [], marker="o", ls="", color=CLASS_COLORS[0], label="Class 0 (test)"),
               Line2D([], [], marker="o", ls="", color=CLASS_COLORS[1], label="Class 1 (test)"),
               Line2D([], [], marker="x", ls="", color="black",
                      label=f"misclassified by MLP ({wrong.sum()})"),
               Line2D([], [], color="black", lw=1.5, label="MLP boundary P = 0.5"),
               Line2D([], [], color="#0b0b0b", ls="--", lw=1.4,
                      label=f"logistic-regression line ({100 * base_acc:.1f}%)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2,
              fontsize=8, frameon=False)
    save(fig, "fig3.png")

    # Geometry of the class-1 region, measured along the diagonal x1 = x2 = s
    # (the axis through all three cluster centres).
    s = np.linspace(-6, 6, 12001)
    p_diag = model.predict_proba(scale(np.column_stack([s, s])))[:, 0]
    cross = s[np.where(np.diff(p_diag >= 0.5))[0]]
    print(f"[Ex2-D] along x1 = x2 = s the prediction switches class at s = {np.round(cross, 3)}")
    if len(cross) == 2:
        print(f"[Ex2-D] class-0 band width along the diagonal = "
              f"{(cross[1] - cross[0]) * np.sqrt(2):.3f} (distance between the two boundaries)")
    for k, name in enumerate(["(0,0) class 0", "(3,3) class 1", "(-3,-3) class 1"]):
        m = c_te == k
        print(f"[Ex2-D] MLP on test cluster {name:<16}: accuracy = "
              f"{np.mean(te_pred[m] == y_te[m]):.2f}")
    print(f"[Ex2-D] fraction of the Figure-3 grid predicted class 1 = {np.mean(P >= 0.5):.3f}")
    return dict(layers=layers, lr=lr, epochs=epochs, test_acc=test_acc, base_acc=base_acc)


# =============================================================================
# Exercise 3 - multi-class, same network
# =============================================================================
# --8<-- [start:ex3_data]
EX3_KW = dict(n_samples=1500, n_features=4, n_informative=4, n_redundant=0,
              n_repeated=0, n_classes=3, n_clusters_per_class=2, class_sep=1.0,
              random_state=42)


def ex3_data():
    """The statement's make_classification call, unchanged, and the 80/20 split."""
    X, y = make_classification(**EX3_KW)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42,
                                              stratify=y)
    return X, y, X_tr, X_te, y_tr, y_te
# --8<-- [end:ex3_data]


# --8<-- [start:ex3_clusters]
def generating_clusters(X):
    """Recover which of the 6 make_classification clusters produced each row of X.

    make_classification draws cluster k's 250 points as rows 250k .. 250k+249 and
    only at the very end (after every random draw) shuffles rows and columns.
    Calling it again with shuffle=False therefore yields the same points in
    generation order; matching each row by its sorted values (invariant to the
    column shuffle) gives its generating cluster. Analysis only - the network is
    trained on the statement's (shuffled) data.
    """
    X_ordered, _ = make_classification(**{**EX3_KW, "shuffle": False})
    key = lambda row: tuple(np.round(np.sort(row), 10))
    position = {key(r): i for i, r in enumerate(X_ordered)}
    order = np.array([position[key(r)] for r in X])
    assert len(set(order)) == len(X), "row matching must be one-to-one"
    n_clusters = EX3_KW["n_classes"] * EX3_KW["n_clusters_per_class"]
    return order // (len(X) // n_clusters)        # cluster id 0..5; its class is id % 3
# --8<-- [end:ex3_clusters]


def plot_confusion(cm, title, fname):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({100 * cm[i, j] / cm[i].sum():.0f}%)",
                    ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > cm.max() / 2 else "#0b0b0b")
    labels = [f"Class {k}" for k in range(cm.shape[0])]
    ax.set_xticks(range(cm.shape[0]), labels)
    ax.set_yticks(range(cm.shape[0]), labels)
    ax.set(title=title, xlabel="predicted class", ylabel="true class")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="number of test samples")
    save(fig, fname)


def exercise3(rng):
    header("EXERCISE 3 - three classes, same MLP class")
    X, y, X_tr, X_te, y_tr, y_te = ex3_data()
    scale = standardizer(X_tr)
    print(f"[Ex3-A] samples: total={len(X)}  train={len(X_tr)}  test={len(X_te)}")
    print(f"[Ex3-A] class counts: all={np.bincount(y)}  train={np.bincount(y_tr)}  "
          f"test={np.bincount(y_te)}")
    print(f"[Ex3-A] train feature means = {X_tr.mean(0).round(3)}, stds = {X_tr.std(0).round(3)}")

    # --8<-- [start:ex3_train]
    # B/C - the SAME MLP class as Exercise 2; only constructor arguments change:
    # 3 output units, output="softmax" (which also selects categorical CE).
    epochs = 2000
    width, lr = select_hyperparameters(X_tr, y_tr, n_in=4, n_out=3, output="softmax",
                                       widths=[8, 16, 32], lrs=[0.1, 0.3, 0.5],
                                       epochs=epochs, rng=rng)
    layers = [4, width, 3]
    model = MLP(layers, hidden_activation="tanh", output="softmax", rng=rng)
    history = model.fit(scale(X_tr), one_hot(y_tr, 3), epochs=epochs, lr=lr)
    pred = model.predict(scale(X_te))
    test_acc = np.mean(pred == y_te)
    cm = confusion_matrix(y_te, pred)          # rows = true class, columns = predicted
    # --8<-- [end:ex3_train]
    train_acc = np.mean(model.predict(scale(X_tr)) == y_tr)
    gc = gradient_check(model, scale(X_tr[:100]), one_hot(y_tr[:100], 3))
    print(f"[Ex3-C] architecture = {layers}, tanh hidden, softmax output, categorical CE")
    print(f"[Ex3-C] parameters = {model.n_parameters()}")
    print(f"[Ex3-C] learning rate = {lr}, epochs = {epochs} (full batch)")
    print(f"[Ex3-C] train loss: epoch 1 = {history[0]:.4f}, epoch 100 = {history[99]:.4f}, "
          f"epoch 500 = {history[499]:.4f}, final = {history[-1]:.4f}")
    print(f"[Ex3-C] train accuracy = {train_acc:.4f}")
    print(f"[Ex3-C] test accuracy  = {test_acc:.4f}")
    print(f"[Ex3-C] gradient check (softmax+CE): max |analytic - numeric| = {gc:.2e}")
    print(f"[Ex3-C] confusion matrix (rows true 0..2, cols predicted 0..2):\n{cm}")

    # Figure 4 - training loss per epoch
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(np.arange(1, epochs + 1), history, color=NET_COLORS[1], lw=2,
            label=f"MLP {layers}, lr = {lr}")
    ax.set(title="Figure 4 - Exercise 3 training loss (categorical cross-entropy)",
           xlabel="epoch", ylabel="training loss (CE)")
    ax.legend()
    ax.grid(alpha=0.25)
    save(fig, "fig4.png")

    # Figure 5 - confusion matrix on the test set
    plot_confusion(cm, f"Figure 5 - test confusion matrix\nMLP {layers}, {len(y_te)} test points",
                   "fig5.png")

    # D - most-confused pair = largest cm[i, j] + cm[j, i] over i < j
    pair_counts = {(i, j): int(cm[i, j] + cm[j, i]) for i in range(3) for j in range(i + 1, 3)}
    for (i, j), n in pair_counts.items():
        print(f"[Ex3-D] confusions between class {i} and class {j}: "
              f"{cm[i, j]} + {cm[j, i]} = {n}")
    worst = max(pair_counts, key=pair_counts.get)
    print(f"[Ex3-D] most-confused pair = classes {worst} with {pair_counts[worst]} "
          f"of {np.sum(pred != y_te)} test errors")

    # Baseline on the same standardised inputs
    baseline = LogisticRegression(max_iter=1000).fit(scale(X_tr), y_tr)
    base_pred = baseline.predict(scale(X_te))
    base_acc = np.mean(base_pred == y_te)
    print(f"[Ex3-D] logistic regression test accuracy = {base_acc:.4f}")
    print(f"[Ex3-D] MLP - logistic regression = {100 * (test_acc - base_acc):+.2f} points")
    print(f"[Ex3-D] logistic-regression confusion matrix:\n{confusion_matrix(y_te, base_pred)}")

    # D - cluster structure (6 clusters in 4-D)
    cluster = generating_clusters(X)
    _, _, _, k_te = train_test_split(X, cluster, test_size=0.2, random_state=42, stratify=y)
    _, _, k_tr_flip, _ = train_test_split(X, cluster, test_size=0.2, random_state=42, stratify=y)
    print(f"[Ex3-D] labels changed by make_classification's flip_y=0.01: "
          f"{np.sum(y != cluster % 3)} of {len(y)} (train {np.sum(y_tr != k_tr_flip % 3)}, "
          f"test {np.sum(y_te != k_te % 3)})")
    cent = np.array([X[cluster == k].mean(0) for k in range(6)])
    for k in range(6):
        Xk = X[cluster == k]
        rms = np.sqrt(np.mean(np.sum((Xk - cent[k]) ** 2, axis=1)))
        m = k_te == k
        signs = "".join("+" if v > 0 else "-" for v in cent[k])
        print(f"[Ex3-D] cluster C{k}: class {k % 3}  centroid {cent[k].round(2)}  "
              f"vertex {signs}  rms radius {rms:.2f}  test n={m.sum():>2}  "
              f"MLP acc={np.mean(pred[m] == y_te[m]):.2f}  "
              f"LR acc={np.mean(base_pred[m] == y_te[m]):.2f}  "
              f"MLP predictions={np.bincount(pred[m], minlength=3)}")
    print("[Ex3-D] inter-class cluster pairs, sorted by centroid distance:")
    pairs = []
    for a in range(6):
        for b in range(a + 1, 6):
            if a % 3 == b % 3:
                continue
            d = np.linalg.norm(cent[b] - cent[a])
            # Project both clusters on the axis joining their centroids and count the
            # points that fall on the other cluster's side of the midpoint.
            u = (cent[b] - cent[a]) / d
            mid = (cent[a] + cent[b]) / 2
            wrong_a = np.mean((X[cluster == a] - mid) @ u > 0)
            wrong_b = np.mean((X[cluster == b] - mid) @ u < 0)
            pairs.append((d, a, b, (wrong_a + wrong_b) / 2))
    for d, a, b, ov in sorted(pairs):
        print(f"          C{a}(class {a % 3}) - C{b}(class {b % 3}): distance = {d:.2f}  "
              f"points past the midpoint = {100 * ov:.1f}%")
    # A linear softmax model gives every class a CONVEX region, so if both of a
    # class's cluster centres are classified correctly, the whole segment between
    # them is too. How close does that segment pass to the other classes' clusters?
    for cls in range(3):
        a, b = cls, cls + 3                       # the two clusters of this class
        v = cent[b] - cent[a]
        for o in range(6):
            if o % 3 == cls:
                continue
            t = np.clip((cent[o] - cent[a]) @ v / (v @ v), 0.0, 1.0)
            dist = np.linalg.norm(cent[a] + t * v - cent[o])
            print(f"[Ex3-D] segment C{a}-C{b} (class {cls}) passes {dist:.2f} from the "
                  f"centre of C{o} (class {o % 3}) at t = {t:.2f}")
    return dict(layers=layers, lr=lr, epochs=epochs, test_acc=test_acc, history=history,
                worst=worst, pair_counts=pair_counts, base_acc=base_acc, model=model)


# =============================================================================
# Exercise 4 - does a second hidden layer help?
# =============================================================================
def exercise4(rng, ex3):
    header("EXERCISE 4 - one extra hidden layer")
    X, y, X_tr, X_te, y_tr, y_te = ex3_data()          # same data and split as Ex. 3
    scale = standardizer(X_tr)
    Y_tr = one_hot(y_tr, 3)
    epochs, lr = ex3["epochs"], ex3["lr"]               # same training budget as Ex. 3
    width = ex3["layers"][1]

    # --8<-- [start:ex4_train]
    # A - the only change is depth: a second hidden layer of the same width.
    shallow = [4, width, 3]
    deep = [4, width, width, 3]
    model = MLP(deep, hidden_activation="tanh", output="softmax", rng=rng)
    history = model.fit(scale(X_tr), Y_tr, epochs=epochs, lr=lr)
    test_acc = np.mean(model.predict(scale(X_te)) == y_te)
    # --8<-- [end:ex4_train]
    gc = gradient_check(model, scale(X_tr[:100]), Y_tr[:100])
    print(f"[Ex4-A] architecture = {deep}, tanh hidden, softmax output, categorical CE")
    print(f"[Ex4-A] parameters = {model.n_parameters()}")
    print(f"[Ex4-A] learning rate = {lr}, epochs = {epochs} (same as Exercise 3)")
    print(f"[Ex4-A] train loss: epoch 1 = {history[0]:.4f}, epoch 100 = {history[99]:.4f}, "
          f"epoch 500 = {history[499]:.4f}, final = {history[-1]:.4f}")
    print(f"[Ex4-A] train accuracy = {np.mean(model.predict(scale(X_tr)) == y_tr):.4f}")
    print(f"[Ex4-A] test accuracy  = {test_acc:.4f}   (Exercise 3 network: "
          f"{ex3['test_acc']:.4f})")
    print(f"[Ex4-A] gradient check (2 hidden layers): max |analytic - numeric| = {gc:.2e}")

    # --8<-- [start:ex4_seeds]
    # B - controlled comparison over several initialisation seeds. Both
    # architectures get the same seeds; data, split, epochs and lr are fixed, so the
    # depth is the only thing that differs inside each pair of runs.
    results = {"shallow": [], "deep": []}
    curves = {"shallow": [], "deep": []}
    for seed in SEEDS:
        for name, layers in (("shallow", shallow), ("deep", deep)):
            net = MLP(layers, hidden_activation="tanh", output="softmax",
                      rng=np.random.default_rng(seed))
            h = net.fit(scale(X_tr), Y_tr, epochs=epochs, lr=lr)
            results[name].append((h[-1],
                                  np.mean(net.predict(scale(X_tr)) == y_tr),
                                  np.mean(net.predict(scale(X_te)) == y_te)))
            curves[name].append(h)
    # --8<-- [end:ex4_seeds]
    res = {k: np.array(v) for k, v in results.items()}   # columns: loss, train acc, test acc
    print(f"[Ex4-B] seeds = {SEEDS}  (mean +- sample std, ddof=1)")
    for name, layers in (("shallow", shallow), ("deep", deep)):
        n_par = n_parameters(layers)
        r = res[name]
        print(f"[Ex4-B] {name:<7} {str(layers):<14} hidden layers={len(layers) - 2}  "
              f"parameters={n_par}")
        print(f"          per-seed test acc  = {r[:, 2].round(4)}")
        print(f"          per-seed final loss = {r[:, 0].round(4)}")
        print(f"          final train loss = {r[:, 0].mean():.4f} +- {r[:, 0].std(ddof=1):.4f}")
        print(f"          train accuracy   = {r[:, 1].mean():.4f} +- {r[:, 1].std(ddof=1):.4f}")
        print(f"          test accuracy    = {r[:, 2].mean():.4f} +- {r[:, 2].std(ddof=1):.4f}")
        print(f"          train-test gap   = {(r[:, 1] - r[:, 2]).mean():.4f}")
    diff = res["deep"][:, 2] - res["shallow"][:, 2]
    print(f"[Ex4-B] per-seed test-accuracy difference (deep - shallow) = {diff.round(4)}")
    print(f"[Ex4-B] mean difference = {100 * diff.mean():+.2f} +- "
          f"{100 * diff.std(ddof=1):.2f} points; deep better in {np.sum(diff > 0)}, "
          f"tied in {np.sum(diff == 0)}, worse in {np.sum(diff < 0)} of {len(SEEDS)} seeds")
    dl = res["deep"][:, 0] - res["shallow"][:, 0]
    print(f"[Ex4-B] per-seed final-loss difference (deep - shallow) = {dl.round(4)}")
    # Stability under the shared learning rate: epochs in which the loss went UP.
    for name, h_main in (("shallow", ex3["history"]), ("deep", history)):
        runs = [h_main] + curves[name]
        ups = [int(np.sum(np.diff(h) > 0)) for h in runs]
        jumps = [float(np.max(np.diff(h))) for h in runs]
        worst_run = int(np.argmax(jumps))
        print(f"[Ex4-C] {name:<7} epochs with a loss increase (shared-rng run, then seeds) = {ups}")
        if max(jumps) > 0:
            print(f"[Ex4-C] {name:<7} largest single-epoch loss increase = {max(jumps):+.4f} "
                  f"(run {'shared-rng' if worst_run == 0 else SEEDS[worst_run - 1]}, epoch "
                  f"{int(np.argmax(np.diff(runs[worst_run]))) + 2})")
        else:
            print(f"[Ex4-C] {name:<7} the loss decreased at every epoch of every run")
    for name in ("shallow", "deep"):
        c = np.array(curves[name])
        print(f"[Ex4-C] {name:<7} mean loss at epoch 50 = {c[:, 49].mean():.4f}, "
              f"200 = {c[:, 199].mean():.4f}, 1000 = {c[:, 999].mean():.4f}, "
              f"{epochs} = {c[:, -1].mean():.4f}")

    # Reference ceiling: one Gaussian per GENERATING cluster (mean and covariance
    # estimated on the train points of that cluster), test point -> most likely
    # cluster -> its class. It knows the cluster ids, which the networks never see,
    # so it is not a competitor; it estimates how much the cluster overlap allows.
    cluster = generating_clusters(X)
    _, _, k_tr, _ = train_test_split(X, cluster, test_size=0.2, random_state=42, stratify=y)
    scores = []
    for k in range(6):
        Xk = X_tr[k_tr == k]
        mu_k, S_k = Xk.mean(axis=0), np.cov(Xk.T)
        d = X_te - mu_k
        maha = np.einsum("ij,jk,ik->i", d, np.linalg.inv(S_k), d)
        scores.append(-0.5 * (maha + np.linalg.slogdet(S_k)[1]))
    oracle_acc = np.mean(np.argmax(scores, axis=0) % 3 == y_te)
    print(f"[Ex4-C] Gaussian-cluster reference (knows cluster ids) test accuracy = "
          f"{oracle_acc:.4f}")
    print(f"[Ex4-C] gap to reference: shallow {100 * (oracle_acc - res['shallow'][:, 2].mean()):.2f}"
          f" points, deep {100 * (oracle_acc - res['deep'][:, 2].mean()):.2f} points")

    # Figure 6 - training-loss curves of both networks on the same axes
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    t = np.arange(1, epochs + 1)
    for name, hl in (("shallow", 1), ("deep", 2)):
        for h in curves[name]:
            ax.plot(t, h, color=NET_COLORS[hl], lw=0.7, alpha=0.35)
    ax.plot(t, ex3["history"], color=NET_COLORS[1], lw=2.2,
            label=f"Exercise 3: {shallow} (1 hidden layer)")
    ax.plot(t, history, color=NET_COLORS[2], lw=2.2,
            label=f"Exercise 4: {deep} (2 hidden layers)")
    ax.plot([], [], color="gray", lw=0.7, alpha=0.6,
            label=f"thin lines: seeds {SEEDS[0]}-{SEEDS[-1]}, same colours")
    ax.set(title=f"Figure 6 - training loss, 1 vs 2 hidden layers (lr = {lr})",
           xlabel="epoch", ylabel="training loss (CE)")
    ax.legend()
    ax.grid(alpha=0.25)
    save(fig, "fig6.png")
    return dict(layers=deep, test_acc=test_acc, diff=diff)


def main():
    rng = np.random.default_rng(42)          # the ONE generator of the report
    check_backprop.run()                     # Exercise 1 (hand derivation is in the report)
    ex2 = exercise2(rng)
    ex3 = exercise3(rng)
    ex4 = exercise4(rng, ex3)

    header("RESULTS SUMMARY (rows 4-9)")
    print(f"[Summary 4] Ex2 linear baseline test accuracy = {100 * ex2['base_acc']:.1f}%")
    print(f"[Summary 5] Ex2 {ex2['layers']} lr={ex2['lr']} epochs={ex2['epochs']}: "
          f"test accuracy = {100 * ex2['test_acc']:.1f}%")
    print(f"[Summary 6] Ex3 {ex3['layers']} lr={ex3['lr']} epochs={ex3['epochs']}: "
          f"test accuracy = {100 * ex3['test_acc']:.1f}%")
    print(f"[Summary 7] Ex3 most-confused pair = classes {ex3['worst']} "
          f"({ex3['pair_counts'][ex3['worst']]} test confusions)")
    print(f"[Summary 8] Ex4 {ex4['layers']}: test accuracy = {100 * ex4['test_acc']:.1f}%")
    print(f"[Summary 9] Ex4 - Ex3 test accuracy, mean over {len(SEEDS)} seeds = "
          f"{100 * ex4['diff'].mean():+.2f} points")


if __name__ == "__main__":
    main()
