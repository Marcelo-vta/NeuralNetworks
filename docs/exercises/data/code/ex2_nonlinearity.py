"""Exercise 2 - Non-linearity in higher dimensions (5-D).

Dataset I: two shifted multivariate Gaussians (classes A and B).
Dataset II: two concentric shells (classes C = core, D = shell).
Both are projected with PCA (Figure 4); the center distance and the radius
||x|| are measured in the original 5-D space (Figure 5); and the hand-written
function f(x) = ||x||^2 - 12.25 is evaluated as a separator of Dataset II.

scikit-learn is used ONLY for PCA. No model is trained: the threshold 12.25
comes from the specified radii (2 and 5), not from the data.
This module never creates its own random generator; ``main.py`` passes the
single ``rng`` of the report into ``ex2(rng, fig_dir)``.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

N_PER_CLASS = 500
DIM = 5

# --8<-- [start:params_I]
# Dataset I parameters, copied from the statement.
MU_A = np.zeros(DIM)
SIGMA_A = np.array([[1.0, 0.8, 0.1, 0.0, 0.0],
                    [0.8, 1.0, 0.3, 0.0, 0.0],
                    [0.1, 0.3, 1.0, 0.5, 0.0],
                    [0.0, 0.0, 0.5, 1.0, 0.2],
                    [0.0, 0.0, 0.0, 0.2, 1.0]])
MU_B = np.full(DIM, 1.5)
SIGMA_B = np.array([[1.5, -0.7, 0.2, 0.0, 0.0],
                    [-0.7, 1.5, 0.4, 0.0, 0.0],
                    [0.2, 0.4, 1.5, 0.6, 0.0],
                    [0.0, 0.0, 0.6, 1.5, 0.3],
                    [0.0, 0.0, 0.0, 0.3, 1.5]])
# --8<-- [end:params_I]

# --8<-- [start:params_II]
# Dataset II parameters: radius rho ~ N(mean, std) for each class.
RHO_C = (2.0, 0.4)    # class C, the core
RHO_D = (5.0, 0.4)    # class D, the shell
# Threshold on ||x||^2 taken from the SPECIFIED radii, not fitted to the data:
# halfway between the two mean radii, (2 + 5) / 2 = 3.5, squared = 12.25.
RADIUS_THRESHOLD = (RHO_C[0] + RHO_D[0]) / 2
SQ_THRESHOLD = RADIUS_THRESHOLD ** 2
# --8<-- [end:params_II]

# Fixed colors per class (same palette as Exercise 1); markers as a second cue.
CLASS_STYLE = {"A": ("#2a78d6", "o"), "B": ("#eb6834", "s"),
               "C": ("#1baf7a", "^"), "D": ("#9b2c8f", "D")}


# --8<-- [start:pd_check]
def check_positive_definite(name, cov):
    """A valid covariance matrix must be symmetric with all eigenvalues > 0."""
    assert np.allclose(cov, cov.T), f"{name} is not symmetric"
    eig = np.linalg.eigvalsh(cov)            # eigenvalues of a symmetric matrix
    print(f"[2A] {name} eigenvalues: {np.round(eig, 4).tolist()} "
          f"-> min = {eig.min():.4f}, positive definite = {bool(eig.min() > 0)}")
    assert eig.min() > 0, f"{name} is not positive definite"
    return eig
# --8<-- [end:pd_check]


# --8<-- [start:dataset_I]
def make_dataset_I(rng):
    """500 points of class A and 500 of class B from N(mu, Sigma) in 5-D."""
    XA = rng.multivariate_normal(MU_A, SIGMA_A, size=N_PER_CLASS)
    XB = rng.multivariate_normal(MU_B, SIGMA_B, size=N_PER_CLASS)
    X = np.vstack([XA, XB])
    y = np.repeat([0, 1], N_PER_CLASS)        # 0 = A, 1 = B
    return X, y
# --8<-- [end:dataset_I]


# --8<-- [start:dataset_II]
def sample_shell(rng, rho_mean, rho_std, n):
    """n points x = rho * u with u uniform on the unit sphere of R^5."""
    v = rng.standard_normal((n, DIM))                    # v ~ N(0, I_5)
    u = v / np.linalg.norm(v, axis=1, keepdims=True)     # u = v / ||v||, ||u|| = 1
    rho = rng.normal(rho_mean, rho_std, size=n)          # one radius per point
    return rho[:, None] * u, u


def make_dataset_II(rng):
    """500 core points (class C) and 500 shell points (class D)."""
    XC, uC = sample_shell(rng, *RHO_C, N_PER_CLASS)
    XD, uD = sample_shell(rng, *RHO_D, N_PER_CLASS)
    # Sanity check of the construction: every direction has unit norm.
    u_norms = np.linalg.norm(np.vstack([uC, uD]), axis=1)
    print(f"[2B] ||u|| over the 1000 directions: min = {u_norms.min():.6f}, "
          f"max = {u_norms.max():.6f}")
    X = np.vstack([XC, XD])
    y = np.repeat([0, 1], N_PER_CLASS)        # 0 = C, 1 = D
    return X, y
# --8<-- [end:dataset_II]


# --8<-- [start:pca]
def project_pca(X):
    """Fit a 2-component PCA on the 1000 points (labels are NOT used)."""
    pca = PCA(n_components=2)          # centers the data, no scaling: the five
    Z = pca.fit_transform(X)           # features already share the same units
    return Z, pca


def plot_figure4(projections, path):
    """Figure 4: the two PCA projections side by side, same axis limits."""
    lim = 1.08 * max(np.abs(Z).max() for Z, *_ in projections.values())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    for ax, (title, (Z, y, pca, names)) in zip(axes, projections.items()):
        for label, name in enumerate(names):
            color, marker = CLASS_STYLE[name]
            ax.scatter(*Z[y == label].T, s=10, alpha=0.55, color=color,
                       marker=marker, edgecolors="none", label=f"Class {name}")
        ev = pca.explained_variance_ratio_
        ax.set_title(f"{title}\nPC1 + PC2 explain {100 * ev.sum():.1f}% of the variance")
        ax.set_xlabel(f"PC1 ({100 * ev[0]:.1f}% of variance)")
        ax.set_ylabel(f"PC2 ({100 * ev[1]:.1f}% of variance)")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9, markerscale=1.5)
    fig.suptitle("Figure 4 - 2-D PCA projection of each 5-D dataset")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
# --8<-- [end:pca]


# --8<-- [start:geometry]
def center_distance(X, y):
    """||mu_1 - mu_2|| between the two SAMPLE class means, computed in 5-D."""
    return float(np.linalg.norm(X[y == 0].mean(axis=0) - X[y == 1].mean(axis=0)))


def radii(X):
    """Radius ||x|| of every point, in 5-D."""
    return np.linalg.norm(X, axis=1)


def plot_figure5(datasets, path):
    """Figure 5: radius histograms, both classes overlaid, one panel per dataset."""
    all_r = np.concatenate([radii(X) for X, *_ in datasets.values()])
    bins = np.linspace(0, all_r.max() * 1.02, 50)        # same bins in both panels
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for ax, (title, (X, y, names)) in zip(axes, datasets.items()):
        r = radii(X)
        for label, name in enumerate(names):
            ax.hist(r[y == label], bins=bins, alpha=0.6, color=CLASS_STYLE[name][0],
                    edgecolor="white", linewidth=0.5, label=f"Class {name}")
        ax.set_title(title)
        ax.set_xlabel("Radius $\\|x\\|$ (5-D Euclidean norm)")
        ax.set_ylabel("Number of points")
        ax.grid(alpha=0.3)
    axes[1].axvline(RADIUS_THRESHOLD, color="black", linestyle="--", linewidth=1.5,
                    label=f"$\\|x\\|$ = {RADIUS_THRESHOLD} ($\\|x\\|^2$ = {SQ_THRESHOLD})")
    for ax in axes:
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle("Figure 5 - Distribution of the radius $\\|x\\|$ per class, computed in 5-D")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
# --8<-- [end:geometry]


# --8<-- [start:separator]
def f_shells(X):
    """Hand-written separator of Dataset II: f(x) = sum_i x_i^2 - 12.25.

    f(x) < 0 -> class C (core), f(x) > 0 -> class D (shell). It is linear in
    the squared inputs x_i^2 but quadratic in the original inputs x_i.
    """
    return np.sum(X ** 2, axis=1) - SQ_THRESHOLD


def best_hyperplane_accuracy_shells(r_core=RHO_C[0], r_shell=RHO_D[0]):
    """Best accuracy ANY hyperplane can reach on ideal shells (radii exactly 2 and 5).

    Both classes are rotation-invariant, so every direction w is equivalent and
    only the offset t of the rule "shell if w.x > t" matters. For u uniform on
    the unit sphere of R^5, one coordinate has density (3/4)(1 - a^2) on [-1, 1],
    hence CDF F(a) = 1/2 + (3/4)(a - a^3/3). A point of radius r lands on the
    "shell" side when u_1 > t / r, so
        accuracy(t) = [P(shell side | shell) + P(core side | core)] / 2
                    = [1 - F(t / r_shell) + F(t / r_core)] / 2.
    This is a property of the distributions, not of the sample: more data
    cannot push a hyperplane above this value.
    """
    def F(a):
        a = np.clip(a, -1.0, 1.0)
        return 0.5 + 0.75 * (a - a ** 3 / 3)
    t = np.linspace(-r_shell, r_shell, 20001)
    acc = 0.5 * ((1 - F(t / r_shell)) + F(t / r_core))
    k = int(acc.argmax())
    return float(acc[k]), float(t[k])
# --8<-- [end:separator]


# --8<-- [start:ex2]
def ex2(rng, fig_dir):
    """Run items A to D in statement order and print every quoted number."""
    print("=" * 70)
    print("EXERCISE 2 - Non-linearity in 5-D")
    print("=" * 70)

    # ---- A: Dataset I ----------------------------------------------------------
    check_positive_definite("Sigma_A", SIGMA_A)
    check_positive_definite("Sigma_B", SIGMA_B)
    X1, y1 = make_dataset_I(rng)
    for label, (name, cov) in enumerate((("A", SIGMA_A), ("B", SIGMA_B))):
        Xk = X1[y1 == label]
        print(f"[2A] class {name}: n = {len(Xk)}, sample mean = "
              f"{np.round(Xk.mean(axis=0), 3).tolist()}")
        print(f"[2A] class {name}: sample corr(x1, x2) = "
              f"{np.corrcoef(Xk[:, 0], Xk[:, 1])[0, 1]:.3f} (specified: "
              f"{cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]):.3f})")

    # ---- B: Dataset II ---------------------------------------------------------
    X2, y2 = make_dataset_II(rng)
    for label, name in enumerate("CD"):
        r = radii(X2[y2 == label])
        print(f"[2B] class {name}: n = {len(r)}, radius mean = {r.mean():.3f}, "
              f"std = {r.std(ddof=1):.3f}, min = {r.min():.3f}, max = {r.max():.3f}")

    # ---- C: PCA, explained variance, center distance, radius histograms --------
    Z1, pca1 = project_pca(X1)
    Z2, pca2 = project_pca(X2)
    for name, pca in (("Dataset I", pca1), ("Dataset II", pca2)):
        ev = pca.explained_variance_ratio_
        print(f"[2C] {name}: explained variance PC1 = {ev[0]:.4f}, PC2 = {ev[1]:.4f}, "
              f"PC1+PC2 = {ev.sum():.4f}")
    # How well is the class-mean difference of Dataset I captured by PC1?
    delta = X1[y1 == 1].mean(axis=0) - X1[y1 == 0].mean(axis=0)
    cos_pc1 = abs(pca1.components_[0] @ delta) / np.linalg.norm(delta)
    print(f"[2C] Dataset I: |cos(PC1, mean_B - mean_A)| = {cos_pc1:.4f}")
    # How mixed does Dataset II LOOK in the PCA plane? Shell points that land
    # inside the 2-D circle of radius 3.5 overlap the core there.
    r2d = np.linalg.norm(Z2, axis=1)
    print(f"[2C] Dataset II: shell (D) points inside radius {RADIUS_THRESHOLD} of the "
          f"PCA plane: {np.mean(r2d[y2 == 1] < RADIUS_THRESHOLD):.4f}")
    plot_figure4({"Dataset I - shifted Gaussians": (Z1, y1, pca1, "AB"),
                  "Dataset II - concentric shells": (Z2, y2, pca2, "CD")},
                 fig_dir / "fig4.png")

    d1, d2 = center_distance(X1, y1), center_distance(X2, y2)
    print(f"[2C] Dataset I: ||mean_A - mean_B|| (5-D) = {d1:.4f} "
          f"(specified: {np.linalg.norm(MU_B - MU_A):.4f})")
    print(f"[2C] Dataset II: ||mean_C - mean_D|| (5-D) = {d2:.4f} (specified: 0)")
    for X, y, names in ((X1, y1, "AB"), (X2, y2, "CD")):
        r = radii(X)
        for label, name in enumerate(names):
            rk = r[y == label]
            print(f"[2C] radius class {name}: mean = {rk.mean():.3f}, "
                  f"min = {rk.min():.3f}, max = {rk.max():.3f}")
    plot_figure5({"Dataset I - shifted Gaussians": (X1, y1, "AB"),
                  "Dataset II - concentric shells": (X2, y2, "CD")},
                 fig_dir / "fig5.png")

    # ---- D: a separating function for Dataset II ------------------------------
    pred = (f_shells(X2) > 0).astype(int)       # 1 = shell (D), 0 = core (C)
    acc = float(np.mean(pred == y2))
    sq = np.sum(X2 ** 2, axis=1)
    print(f"[2D] rule ||x||^2 > {SQ_THRESHOLD} -> class D: accuracy on Dataset II = "
          f"{acc:.4f} ({int((pred == y2).sum())} of {len(y2)})")
    print(f"[2D] ||x||^2 range: core max = {sq[y2 == 0].max():.3f}, "
          f"shell min = {sq[y2 == 1].min():.3f}")
    # Contrast with a LINEAR rule built from the specified means (nothing fitted):
    # on Dataset I, the hyperplane that bisects mu_A and mu_B.
    w = MU_B - MU_A
    b = -w @ (MU_A + MU_B) / 2
    acc_lin_I = float(np.mean(((X1 @ w + b) > 0).astype(int) == y1))
    print(f"[2D] Dataset I, bisector hyperplane of the specified means: accuracy = "
          f"{acc_lin_I:.4f}")
    # On Dataset II both specified means are 0, so there is no such bisector.
    # The best ANY hyperplane can do (theory), and that rule applied to the sample
    # with the arbitrary direction w = e_1 (all directions are equivalent).
    best_acc, best_t = best_hyperplane_accuracy_shells()
    print(f"[2D] best possible hyperplane on ideal shells (r = 2 vs 5): "
          f"accuracy = {best_acc:.4f} at offset t = {best_t:.3f}")
    acc_lin_II = float(np.mean((X2[:, 0] > best_t).astype(int) == y2))
    print(f"[2D] Dataset II, hyperplane x_1 > {best_t:.3f} -> class D: accuracy = "
          f"{acc_lin_II:.4f}")
    return {"dist_I": d1, "dist_II": d2,
            "ev_I": float(pca1.explained_variance_ratio_.sum()),
            "ev_II": float(pca2.explained_variance_ratio_.sum())}
# --8<-- [end:ex2]
