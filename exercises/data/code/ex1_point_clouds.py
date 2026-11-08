"""Exercise 1 - Point clouds: geometry and spread in 2D.

Item A generates four 2-D Gaussian clouds (Figure 1). Item B rescales their
spread by s in {0.5, 1, 2, 4} (Figure 2), computes the separation ratio r_ij
and the mixing rate, and plots the mixing rate against s (Figure 3). Item C
draws the sketched decision boundaries on a copy of Figure 1 (fig1_sketch.png).

Nothing is trained here: every boundary and every number comes from the
specified means and standard deviations, or from plain NumPy geometry.
This module never creates its own random generator; ``main.py`` passes the
single ``rng`` of the report into ``ex1(rng, fig_dir)``.
"""

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# --8<-- [start:params]
# Parameters copied from the statement: one row per class, columns (x, y).
MEANS = np.array([[2.0, 3.0],    # class 0
                  [5.0, 6.0],    # class 1
                  [8.0, 1.0],    # class 2
                  [15.0, 4.0]])  # class 3
STDS = np.array([[0.8, 2.5],
                 [1.2, 1.9],
                 [0.9, 0.9],
                 [0.5, 2.0]])
N_PER_CLASS = 100                 # 4 classes x 100 = 400 samples
N_CLASSES = len(MEANS)
SCALES = (0.5, 1.0, 2.0, 4.0)     # spread multipliers of item B
# --8<-- [end:params]

# One fixed color and marker per class, reused in every Exercise 1 figure.
# The colors were checked for color-vision-deficiency separation; the marker
# shape is a second cue so that no class is identified by color alone.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#9b2c8f"]
MARKERS = ["o", "s", "^", "D"]


# --8<-- [start:generate]
def draw_noise(rng):
    """Draw the standard-normal noise z ~ N(0, 1) for every class, point and axis.

    Shape (4, 100, 2). Each axis gets its own independent draw, so the clouds
    are independent per-axis Gaussians (diagonal covariance), as specified.
    """
    return rng.standard_normal((N_CLASSES, N_PER_CLASS, 2))


def make_clouds(noise, scale=1.0):
    """Turn the noise into labelled points: x = mu_k + (s * sigma_k) * z.

    Broadcasting applies class k's mean and (scaled) standard deviation to all
    100 points of that class. Returns X with shape (400, 2) and integer labels y.
    """
    points = MEANS[:, None, :] + scale * STDS[:, None, :] * noise
    X = points.reshape(-1, 2)                                  # stack the 4 classes
    y = np.repeat(np.arange(N_CLASSES), N_PER_CLASS)           # 0...0, 1...1, ...
    return X, y
# --8<-- [end:generate]


def _scatter_classes(ax, X, y, size=14, alpha=0.7):
    """Scatter each class with its fixed color/marker and a legend label."""
    for k in range(N_CLASSES):
        ax.scatter(*X[y == k].T, s=size, alpha=alpha, color=COLORS[k],
                   marker=MARKERS[k], edgecolors="none", label=f"Class {k}")


def _mark_centers(ax, size=170, label=True):
    """Mark the cloud centers (the specified means) with a black-edged star."""
    for k, (mx, my) in enumerate(MEANS):
        ax.scatter(mx, my, s=size, marker="*", color=COLORS[k],
                   edgecolors="black", linewidths=1.2, zorder=5,
                   label="Class centers ($\\mu_k$)" if (label and k == 0) else None)


# --8<-- [start:fig1]
def plot_figure1(X, y, path):
    """Figure 1: the 400 points, one color per class, centers marked."""
    fig, ax = plt.subplots(figsize=(9.5, 6))
    _scatter_classes(ax, X, y)
    _mark_centers(ax)
    for k, (mx, my) in enumerate(MEANS):          # write the mean next to each star
        ax.annotate(f"$\\mu_{k}$ = ({mx:g}, {my:g})", (mx, my), xytext=(8, 8),
                    textcoords="offset points", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    ax.set_title("Figure 1 - Four Gaussian clouds (100 points each, s = 1)")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)  # outside
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
# --8<-- [end:fig1]


# --8<-- [start:fig2]
def plot_figure2(datasets, path):
    """Figure 2: one subplot per scale s, all sharing the SAME axis limits."""
    # Shared limits = bounding box of all four datasets plus a 5% margin, so the
    # widest cloud (s = 4) fits and the four panels are directly comparable.
    all_pts = np.vstack([X for X, _ in datasets.values()])
    lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
    pad = 0.05 * (hi - lo)
    xlim, ylim = (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    for ax, (s, (X, y)) in zip(axes.ravel(), datasets.items()):
        _scatter_classes(ax, X, y, size=9, alpha=0.6)
        _mark_centers(ax, size=110, label=False)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"s = {s:g}  (all $\\sigma$ multiplied by {s:g})")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Figure 2 - The same four classes at four spreads (identical axis limits)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return xlim, ylim
# --8<-- [end:fig2]


# --8<-- [start:ratio]
def separation_ratios(scale=1.0):
    """r_ij = ||mu_i - mu_j|| / (sigma_bar_i + sigma_bar_j) for the 6 pairs.

    Uses the SPECIFIED parameters (not sample estimates). sigma_bar_k is the
    average of the two axis standard deviations of class k, times the scale s,
    so r_ij(s) = r_ij(1) / s.
    """
    sigma_bar = scale * STDS.mean(axis=1)              # (sigma_x + sigma_y) / 2
    rows = []
    for i, j in combinations(range(N_CLASSES), 2):     # (0,1), (0,2), ..., (2,3)
        dist = np.linalg.norm(MEANS[i] - MEANS[j])
        rows.append((i, j, dist, sigma_bar[i] + sigma_bar[j],
                     dist / (sigma_bar[i] + sigma_bar[j])))
    return rows


def directional_ratio(i, j, scale=1.0):
    """Variant of r_ij that uses the spread ALONG the line joining the two centers.

    For a class with per-axis stds (sx, sy), the std of its projection onto the
    unit vector d = (dx, dy) is sqrt(dx^2 sx^2 + dy^2 sy^2). r_ij averages the
    two axes instead, so it ignores the direction in which each cloud is spread.
    """
    d = (MEANS[j] - MEANS[i]) / np.linalg.norm(MEANS[j] - MEANS[i])
    spread = lambda k: scale * np.sqrt(np.sum(d ** 2 * STDS[k] ** 2))  # noqa: E731
    return np.linalg.norm(MEANS[j] - MEANS[i]) / (spread(i) + spread(j))
# --8<-- [end:ratio]


# --8<-- [start:mixing]
def nearest_center(X):
    """Index of the closest TRUE class mean for every point (Euclidean)."""
    # (n, 1, 2) - (1, 4, 2) -> (n, 4, 2) differences -> (n, 4) distances.
    dists = np.linalg.norm(X[:, None, :] - MEANS[None, :, :], axis=2)
    return dists.argmin(axis=1)


def mixing_rate(X, y):
    """Fraction of points whose nearest center is NOT the mean of their class."""
    return float(np.mean(nearest_center(X) != y))


def mixing_matrix(X, y):
    """Counts M[true class, nearest-center class]; off-diagonal = mixed points."""
    M = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    np.add.at(M, (y, nearest_center(X)), 1)
    return M
# --8<-- [end:mixing]


# --8<-- [start:separable]
def convex_hull(P):
    """Vertices of the convex hull of 2-D points (Andrew's monotone chain)."""
    P = np.unique(P, axis=0)                 # sorted by x, then y

    def cross(o, a, b):                      # > 0 when o -> a -> b turns left
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in P:                              # lower chain, left to right
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in P[::-1]:                        # upper chain, right to left
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def linearly_separable(A, B):
    """Exact test: can ONE straight line put all of A on one side and all of B on the other?

    Two finite point sets are linearly separable iff their convex hulls do not
    intersect, and two convex polygons are disjoint iff the normal of one of
    their edges is a separating axis (separating-axis theorem). So we project
    both sets onto every edge normal of both hulls and look for a gap.
    Pure geometry on the sampled points: nothing is fitted or trained.
    """
    for H in (convex_hull(A), convex_hull(B)):
        edges = np.roll(H, -1, axis=0) - H           # edge vectors of the hull
        normals = np.c_[-edges[:, 1], edges[:, 0]]   # one normal per edge
        pa, pb = A @ normals.T, B @ normals.T        # projections, (n, n_edges)
        gap = (pa.max(axis=0) < pb.min(axis=0)) | (pb.max(axis=0) < pa.min(axis=0))
        if gap.any():
            return True
    return False


def non_separable_pairs(X, y):
    """Class pairs (i, j) that no single straight line can separate."""
    return [(i, j) for i, j in combinations(range(N_CLASSES), 2)
            if not linearly_separable(X[y == i], X[y == j])]
# --8<-- [end:separable]


# --8<-- [start:fig3]
def plot_figure3(rates, r_min_pair, r_min_value, first_bad_s, path):
    """Figure 3: mixing rate as a function of the scale factor s."""
    s = np.array(list(rates))
    pct = 100 * np.array(list(rates.values()))
    i, j = r_min_pair
    fig, ax = plt.subplots(figsize=(8, 5))
    # Shade the scales at which some class pair is no longer linearly separable.
    ax.axvspan(first_bad_s, s[-1] * 1.45, color="0.85", alpha=0.6, linewidth=0,
               label=f"Some class pair not linearly separable (s $\\geq$ {first_bad_s:g})")
    ax.plot(s, pct, color=COLORS[0], marker="o", markersize=8, linewidth=2,
            label="Mixing rate (nearest true center $\\neq$ own class)")
    # Label each point with its value and the smallest separation ratio at that s,
    # placed where the rising line is not: above the first and last points,
    # below-right of the middle ones.
    for n_pt, (sv, pv) in enumerate(zip(s, pct)):
        text = f"{pv:.2f}%\n($r_{{{i}{j}}}$ = {r_min_value / sv:.2f})"
        if n_pt in (0, len(s) - 1):
            ax.annotate(text, (sv, pv), xytext=(0, 14), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)
        else:
            ax.annotate(text.replace("\n", "  "), (sv, pv), xytext=(10, -16),
                        textcoords="offset points", ha="left", fontsize=9)
    ax.set_xscale("log", base=2)                  # s doubles at each step
    ax.set_xticks(s, [f"{v:g}" for v in s])
    ax.set_xlim(s[0] / 1.45, s[-1] * 1.45)        # room for the edge labels
    ax.set_ylim(0, max(pct) * 1.3)
    ax.set_title("Figure 3 - Mixing rate vs. spread scale factor s")
    ax.set_xlabel("Scale factor s (log$_2$ axis)")
    ax.set_ylabel("Mixing rate (% of the 400 points)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
# --8<-- [end:fig3]


# --8<-- [start:sketch]
def voronoi_edges(xlim, ylim, n=4000):
    """Pieces of the perpendicular bisectors that separate neighbouring means.

    For each pair (i, j) the bisector is the set of points equidistant to mu_i
    and mu_j. We sample that line densely inside the plot window and keep only
    the samples that are at least as close to mu_i (= mu_j) as to every other
    mean: those samples form the boundary segment between the two regions.
    Pairs whose bisector never survives (non-neighbours) are skipped.
    """
    edges = {}
    for i, j in combinations(range(N_CLASSES), 2):
        mid = (MEANS[i] + MEANS[j]) / 2
        d = MEANS[j] - MEANS[i]
        direction = np.array([-d[1], d[0]]) / np.linalg.norm(d)   # along the bisector
        t = np.linspace(-60, 60, n)
        pts = mid + t[:, None] * direction
        inside = ((pts[:, 0] >= xlim[0]) & (pts[:, 0] <= xlim[1]) &
                  (pts[:, 1] >= ylim[0]) & (pts[:, 1] <= ylim[1]))
        dists = np.linalg.norm(pts[:, None, :] - MEANS[None], axis=2)
        keep = inside & (dists[:, i] <= dists.min(axis=1) + 1e-9)
        if keep.any():
            edges[(i, j)] = pts[keep]
    return edges


def plot_sketch(X, y, path):
    """Copy of Figure 1 with the sketched (piecewise-linear) decision boundaries."""
    fig, ax = plt.subplots(figsize=(10.5, 6))
    xlim = (X[:, 0].min() - 1, X[:, 0].max() + 1)
    ylim = (X[:, 1].min() - 1, X[:, 1].max() + 1)

    # Light background tint: each grid cell takes the color of its nearest mean.
    gx, gy = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))
    region = nearest_center(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    ax.contourf(gx, gy, region, levels=np.arange(-0.5, N_CLASSES), alpha=0.12,
                cmap=ListedColormap(COLORS))

    _scatter_classes(ax, X, y)
    _mark_centers(ax)

    # The sketched boundaries: bisector segments between neighbouring means.
    for n_edge, ((i, j), seg) in enumerate(voronoi_edges(xlim, ylim).items()):
        ax.plot(seg[:, 0], seg[:, 1], color="black", linewidth=2,
                label="Sketched boundaries" if n_edge == 0 else None)

    # Points that fall on the wrong side of the sketch: the unavoidable errors.
    wrong = nearest_center(X) != y
    ax.scatter(*X[wrong].T, s=70, facecolors="none", edgecolors="black",
               linewidths=1.2, label=f"Wrong side of the sketch ({wrong.sum()} points)")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title("Figure 1 with sketched decision boundaries (s = 1)")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)  # outside
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return int(wrong.sum())
# --8<-- [end:sketch]


# --8<-- [start:ex1]
def ex1(rng, fig_dir):
    """Run items A, B and C in statement order and print every quoted number."""
    print("=" * 70)
    print("EXERCISE 1 - Point clouds")
    print("=" * 70)

    # ---- A: generate the clouds (the only random draw of Exercise 1) --------
    noise = draw_noise(rng)
    X, y = make_clouds(noise, scale=1.0)
    print(f"[1A] samples: {len(X)} total, per class: {np.bincount(y).tolist()}")
    for k in range(N_CLASSES):
        m, sd = X[y == k].mean(axis=0), X[y == k].std(axis=0, ddof=1)
        print(f"[1A] class {k}: sample mean = ({m[0]:.2f}, {m[1]:.2f}), "
              f"sample std = ({sd[0]:.2f}, {sd[1]:.2f})  "
              f"[specified mu = {MEANS[k].tolist()}, sigma = {STDS[k].tolist()}]")
    plot_figure1(X, y, fig_dir / "fig1.png")

    # ---- B: the same classes at four spreads ---------------------------------
    # The SAME noise is reused for every s, so the four datasets differ only in
    # spread: point i of class k sits at mu_k + s * sigma_k * z_i for all s.
    datasets = {s: make_clouds(noise, scale=s) for s in SCALES}
    xlim, ylim = plot_figure2(datasets, fig_dir / "fig2.png")
    print(f"[1B] Figure 2 shared limits: x in [{xlim[0]:.2f}, {xlim[1]:.2f}], "
          f"y in [{ylim[0]:.2f}, {ylim[1]:.2f}]")

    rows = separation_ratios(scale=1.0)
    print(f"[1B] sigma_bar_k = (sigma_x + sigma_y) / 2 at s = 1: "
          f"{dict(enumerate(STDS.mean(axis=1).round(4).tolist()))}")
    print("[1B] separation ratio at s = 1 (specified parameters):")
    for i, j, dist, denom, r in rows:
        print(f"[1B]   pair ({i},{j}): ||mu_i-mu_j|| = {dist:.4f}, "
              f"sigma_bar_i+sigma_bar_j = {denom:.4f}, r = {r:.4f}")
    i, j, *_, r_min = min(rows, key=lambda row: row[-1])
    print(f"[1B] smallest r_ij at s = 1: pair ({i},{j}) r = {r_min:.4f}")
    for s in SCALES:
        print(f"[1B] smallest r_ij at s = {s:g}: r_{i}{j} = r/s = {r_min / s:.4f}")
    print(f"[1B] r_{i}{j}(s) = 1 when s = {r_min:.4f}")
    for a, b, *_ in rows:
        print(f"[1B]   directional ratio (spread along mu_i -> mu_j) pair ({a},{b}) "
              f"at s = 1: {directional_ratio(a, b):.4f}")

    rates = {s: mixing_rate(*datasets[s]) for s in SCALES}
    for s in SCALES:
        n_mixed = int(round(rates[s] * len(X)))
        print(f"[1B] mixing rate at s = {s:g}: {rates[s]:.4f} "
              f"({100 * rates[s]:.2f}%, {n_mixed} of {len(X)} points)")
    for s in SCALES:
        M = mixing_matrix(*datasets[s])
        mixed = {f"{a}->{b}": int(M[a, b]) for a in range(N_CLASSES)
                 for b in range(N_CLASSES) if a != b and M[a, b] > 0}
        print(f"[1B] s = {s:g} mixed points by (true -> nearest center): {mixed}")
    # Exact pairwise check: if every pair is linearly separable, a set of straight
    # lines (one per pair, one-vs-one voting) classifies all 400 points correctly.
    bad_pairs = {s: non_separable_pairs(*datasets[s]) for s in SCALES}
    for s in SCALES:
        print(f"[1B] s = {s:g} pairs NOT linearly separable: "
              f"{bad_pairs[s] if bad_pairs[s] else 'none'}")
    first_bad_s = min(s for s in SCALES if bad_pairs[s])
    print(f"[1B] first scale with a pair that no straight line separates: s = {first_bad_s:g} "
          f"(smallest r there: r_{i}{j} = {r_min / first_bad_s:.4f})")
    plot_figure3(rates, (i, j), r_min, first_bad_s, fig_dir / "fig3.png")

    # ---- C: sketch on Figure 1 ------------------------------------------------
    n_wrong = plot_sketch(X, y, fig_dir / "fig1_sketch.png")
    edges = sorted(voronoi_edges((X[:, 0].min() - 1, X[:, 0].max() + 1),
                                 (X[:, 1].min() - 1, X[:, 1].max() + 1)))
    print(f"[1C] sketched boundary segments between classes: {edges}")
    print(f"[1C] points on the wrong side of the sketch at s = 1: {n_wrong}")
    return rates, rows
# --8<-- [end:ex1]
