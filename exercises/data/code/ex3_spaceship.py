"""Exercise 3 - Preparing Spaceship Titanic (Kaggle) for a tanh network.

Pipeline, in this order:
  A  describe the raw file (target balance, feature types, missing values);
  B  stratified 80/20 split with a fixed seed, BEFORE any statistic is fitted;
  C  impute -> TotalSpend -> log1p -> standardize / impute -> one-hot,
     every transformer fitted on the training split only;
  D  Figure 6 (FoodCourt before/after) and the final checks.

scikit-learn is used ONLY for preprocessing (imputers, encoder, scaler,
ColumnTransformer, train_test_split). No model is trained.
This module never creates its own random generator; ``main.py`` passes the
single ``rng`` of the report into ``ex3(rng, fig_dir, csv_path)``.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

KAGGLE_URL = "https://www.kaggle.com/competitions/spaceship-titanic"

# --8<-- [start:columns]
TARGET = "Transported"                       # bool: True / False
SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
NUM_COLS = ["Age"] + SPEND_COLS              # numerical features
CAT_COLS = ["HomePlanet", "CryoSleep", "Destination", "VIP"]   # categorical
DROP_COLS = ["PassengerId", "Cabin", "Name"]  # identifiers / free text: dropped
TEST_SIZE = 0.20                             # 80/20 split
# --8<-- [end:columns]


# --8<-- [start:load]
def load_raw(csv_path):
    """Read train.csv. Stateless type fix only: no statistic is computed here.

    CryoSleep and VIP hold True/False with gaps. Reading them as text and
    casting all four categorical columns to plain Python objects makes them
    behave the same way downstream (missing stays NaN, values become labels).
    """
    df = pd.read_csv(csv_path, dtype={"CryoSleep": str, "VIP": str})
    df[CAT_COLS] = df[CAT_COLS].astype(object)
    df[TARGET] = df[TARGET].astype(bool)
    return df
# --8<-- [end:load]


# --8<-- [start:describe]
def class_balance(y):
    """Counts and shares of each label of Transported."""
    counts = y.value_counts().sort_index()
    return pd.DataFrame({"count": counts, "share": counts / counts.sum()})


def missing_table(df):
    """Missing values per column: absolute count and percentage of rows."""
    n_missing = df.isna().sum()
    return pd.DataFrame({"missing": n_missing,
                         "percent": 100 * n_missing / len(df)}).sort_values(
                             "missing", ascending=False)


def spending_stats(X):
    """Mean, median, max and skewness of the five spending columns (NaN skipped)."""
    return pd.DataFrame({"mean": X[SPEND_COLS].mean(),
                         "median": X[SPEND_COLS].median(),
                         "max": X[SPEND_COLS].max(),
                         "skew": X[SPEND_COLS].skew()})
# --8<-- [end:describe]


# --8<-- [start:split]
def split(df, seed):
    """Stratified 80/20 split - the FIRST operation after loading.

    stratify=y keeps the share of Transported identical in both parts; the fixed
    seed makes the split reproducible. Nothing has been imputed, encoded or
    scaled yet, so no test-set information can leak into the transformers.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)               # True -> 1, False -> 0
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=seed)
# --8<-- [end:split]


# --8<-- [start:pipeline]
def drop_identifiers(X):
    """Drop PassengerId, Cabin and Name (stateless, same on train and test)."""
    return X.drop(columns=DROP_COLS)


def add_total_spend(X):
    """Append TotalSpend = RoomService + FoodCourt + ShoppingMall + Spa + VRDeck.

    Runs AFTER imputation, so a single missing spend no longer turns the sum
    into NaN. It is a row-wise sum: there is nothing to fit.
    """
    X = np.asarray(X, dtype=float)
    return np.column_stack([X, X.sum(axis=1)])


def _total_spend_names(transformer, input_features):
    """Column names after add_total_spend (keeps get_feature_names_out working)."""
    return np.append(np.asarray(input_features, dtype=object), "TotalSpend")


def build_preprocessor():
    """One ColumnTransformer = three per-column-type pipelines.

    Age      : median imputation -> standardization
    Spending : median imputation -> + TotalSpend -> log1p -> standardization
    Category : most-frequent imputation -> one-hot (unknown -> all zeros)
    Columns not listed (the identifiers, if still present) are dropped.
    """
    age_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),                 # (x - mean_train) / std_train
    ])
    spend_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("total", FunctionTransformer(add_total_spend,
                                      feature_names_out=_total_spend_names)),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("age", age_pipe, ["Age"]),
         ("spend", spend_pipe, SPEND_COLS),
         ("cat", cat_pipe, CAT_COLS)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
# --8<-- [end:pipeline]


# --8<-- [start:fit]
def preprocess(X_train, X_test):
    """Fit every transformer on the TRAINING split, then apply it to both."""
    X_train, X_test = drop_identifiers(X_train), drop_identifiers(X_test)
    pre = build_preprocessor()
    pre.fit(X_train)                       # medians, modes, categories, means, stds
    Xtr = pre.transform(X_train)           # learned from train only ...
    Xte = pre.transform(X_test)            # ... and re-used, unchanged, on test
    names = list(pre.get_feature_names_out())
    return pre, Xtr, Xte, names
# --8<-- [end:fit]


# --8<-- [start:tails]
def tail_compression(X_train, Xtr, names):
    """Largest standardized value of each spending column, without vs. with log1p.

    "Without": impute with the training median, then standardize the raw column.
    "With": the actual pipeline output (impute -> log1p -> standardize).
    A value of z means "z training standard deviations above the mean"; the
    larger it is, the deeper that passenger pushes a tanh unit into saturation.
    """
    out = {}
    for col in SPEND_COLS:
        raw = X_train[col].fillna(X_train[col].median())
        z_raw = ((raw - raw.mean()) / raw.std(ddof=0)).max()
        out[col] = (float(z_raw), float(Xtr[:, names.index(col)].max()))
    return out
# --8<-- [end:tails]


# --8<-- [start:unknown]
def unseen_category_demo(pre, X_test):
    """Show what the encoder does with a category it never saw during fit."""
    row = drop_identifiers(X_test.iloc[[0]]).copy()
    row["HomePlanet"] = "Pluto"              # not a category of the training set
    names = list(pre.get_feature_names_out())
    block = [k for k, n in enumerate(names) if n.startswith("HomePlanet_")]
    encoded = pre.transform(row)[0, block]
    return [names[k] for k in block], encoded
# --8<-- [end:unknown]


# --8<-- [start:fig6]
def plot_figure6(pre, X_train, Xtr, names, path):
    """Figure 6: FoodCourt on the training split - raw, after log1p, after scaling."""
    raw = X_train["FoodCourt"].dropna().to_numpy()
    # Output of the spending pipeline up to (and including) log1p, before scaling.
    spend_pipe = pre.named_transformers_["spend"]
    logged = spend_pipe[:-1].transform(drop_identifiers(X_train)[SPEND_COLS])
    logged = logged[:, SPEND_COLS.index("FoodCourt")]
    scaled = Xtr[:, names.index("FoodCourt")]

    panels = [
        (raw, "Raw FoodCourt (missing values left out)", "FoodCourt spending"),
        (logged, "After median imputation + log(1 + x)", "log(1 + FoodCourt)"),
        (scaled, "After log(1 + x) + standardization", "Standardized log(1 + FoodCourt)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for n_panel, (ax, (values, title, xlabel)) in enumerate(zip(axes, panels)):
        ax.hist(values, bins=60, color="#2a78d6", edgecolor="white", linewidth=0.4,
                label=f"Training split (n = {len(values)})")
        if n_panel == 0:                   # raw panel: log counts make the tail visible
            ax.set_yscale("log")
            ax.set_ylabel("Number of passengers (log scale)")
        else:
            ax.set_ylabel("Number of passengers")
        s = pd.Series(values)
        mean, median, skew = (round(v, 2) + 0.0 for v in (s.mean(), s.median(), s.skew()))
        # Spending 0 is the minimum at every stage (log1p(0) = 0, then shifted).
        zeros = 100 * np.mean(values == values.min())
        ax.set_title(f"{title}\nmean = {mean:.2f}, median = {median:.2f}, "
                     f"skew = {skew:.2f}, zero spend = {zeros:.1f}%", fontsize=9)
        print(f"[3D] Figure 6 panel '{title}': n = {len(values)}, mean = {mean:.2f}, "
              f"median = {median:.2f}, max = {values.max():.2f}, skew = {skew:.2f}, "
              f"zero spend = {zeros:.1f}%")
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Figure 6 - FoodCourt before and after preprocessing (training split)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
# --8<-- [end:fig6]


# --8<-- [start:checks]
def final_checks(Xtr, Xte, names):
    """No NaN, final shapes, and value ranges after scaling (train AND test)."""
    scaled_cols = [names.index(c) for c in ["Age", *SPEND_COLS, "TotalSpend"]]
    report = {
        "nan_train": int(np.isnan(Xtr).sum()),
        "nan_test": int(np.isnan(Xte).sum()),
        "shape_train": Xtr.shape,
        "shape_test": Xte.shape,
        # scaled numerical block (Age, 5 spends, TotalSpend)
        "num_train": (Xtr[:, scaled_cols].min(), Xtr[:, scaled_cols].max()),
        "num_test": (Xte[:, scaled_cols].min(), Xte[:, scaled_cols].max()),
        # whole matrix (the one-hot columns are 0/1)
        "all_train": (Xtr.min(), Xtr.max()),
        "all_test": (Xte.min(), Xte.max()),
    }
    assert report["nan_train"] == 0 and report["nan_test"] == 0, "NaN left after preprocessing"
    return report
# --8<-- [end:checks]


# --8<-- [start:ex3]
def ex3(rng, fig_dir, csv_path):
    """Run items A to D in statement order and print every quoted number."""
    print("=" * 70)
    print("EXERCISE 3 - Spaceship Titanic")
    print("=" * 70)
    pd.set_option("display.width", 120)

    # ---- A: get to know the data (descriptive counts of the raw file) ----------
    df = load_raw(csv_path)
    print(f"[3A] raw shape: {df.shape}")
    bal = class_balance(df[TARGET])
    for label, row in bal.iterrows():
        print(f"[3A] Transported = {label}: {int(row['count'])} rows "
              f"({100 * row['share']:.2f}%)")
    print(f"[3A] numerical features: {NUM_COLS}")
    print(f"[3A] categorical features: {CAT_COLS}")
    print(f"[3A] identifier / text columns (dropped in C): {DROP_COLS}")
    print("[3A] missing values per column (full train.csv):")
    print(missing_table(df).to_string(float_format=lambda v: f"{v:.2f}"))
    # A.5 describes the dataset as delivered (full file). Descriptive only: none
    # of these numbers is used by any transformer.
    print("[3A] spending columns, full train.csv (descriptive, before the split):")
    print(spending_stats(df).to_string(float_format=lambda v: f"{v:.2f}"))
    age = df["Age"]
    print(f"[3A] Age (full file): mean = {age.mean():.2f}, median = {age.median():.2f}, "
          f"min = {age.min():.0f}, max = {age.max():.0f}, skew = {age.skew():.2f}")
    cryo = df[df["CryoSleep"] == "True"]
    print(f"[3A] CryoSleep = True: {len(cryo)} passengers, largest recorded spend in any "
          f"spending column = {cryo[SPEND_COLS].max().max():.1f}")

    # ---- B: split BEFORE any transformation -----------------------------------
    # scikit-learn's random_state accepts an int or a legacy RandomState, not a
    # numpy Generator, so an integer seed is drawn from the report's single rng.
    # It is fixed by default_rng(42) and the order of the draws in Exercises 1-2.
    seed = int(rng.integers(0, 2**31 - 1))
    X_train, X_test, y_train, y_test = split(df, seed)
    print(f"[3B] split seed (drawn from rng): {seed}")
    print(f"[3B] train rows: {len(X_train)}, test rows: {len(X_test)}")
    print(f"[3B] share Transported = True: train {y_train.mean():.4f}, "
          f"test {y_test.mean():.4f}")
    # Results-summary row 11: FoodCourt on the TRAINING split, before transforming.
    fc = X_train["FoodCourt"]
    print(f"[3B] FoodCourt on the training split, before transforming: "
          f"mean = {fc.mean():.2f}, median = {fc.median():.2f}")

    # ---- C: preprocess (fit on train, apply to test) ---------------------------
    pre, Xtr, Xte, names = preprocess(X_train, X_test)
    age_imp = pre.named_transformers_["age"]["impute"]
    spend_imp = pre.named_transformers_["spend"]["impute"]
    cat_imp = pre.named_transformers_["cat"]["impute"]
    onehot = pre.named_transformers_["cat"]["onehot"]
    print(f"[3C] imputer (median, train): Age = {age_imp.statistics_[0]:.2f}")
    print(f"[3C] imputer (median, train): spending = "
          f"{dict(zip(SPEND_COLS, spend_imp.statistics_.round(2).tolist()))}")
    print(f"[3C] imputer (most frequent, train): "
          f"{dict(zip(CAT_COLS, [str(v) for v in cat_imp.statistics_]))}")
    print(f"[3C] one-hot categories (train): "
          f"{ {c: [str(v) for v in cats] for c, cats in zip(CAT_COLS, onehot.categories_)} }")
    unseen = {c: sorted(set(X_test[c].dropna().astype(str)) - {str(v) for v in cats})
              for c, cats in zip(CAT_COLS, onehot.categories_)}
    print(f"[3C] test categories never seen in train: {unseen}")
    demo_names, demo_values = unseen_category_demo(pre, X_test)
    print(f"[3C] HomePlanet = 'Pluto' (unseen) encodes as "
          f"{dict(zip(demo_names, demo_values.tolist()))}")
    print(f"[3C] features after preprocessing ({len(names)}): {names}")
    scaled_cols = [names.index(c) for c in ["Age", *SPEND_COLS, "TotalSpend"]]
    print(f"[3C] scaled columns on train: mean = "
          f"{(np.round(Xtr[:, scaled_cols].mean(axis=0), 4) + 0.0).tolist()}, std = "
          f"{np.round(Xtr[:, scaled_cols].std(axis=0), 4).tolist()}")
    for col, (z_raw, z_log) in tail_compression(X_train, Xtr, names).items():
        print(f"[3C] largest standardized {col} on train: without log1p = {z_raw:.2f}, "
              f"with log1p = {z_log:.2f}")

    # ---- D: verify and visualize ------------------------------------------------
    plot_figure6(pre, X_train, Xtr, names, fig_dir / "fig6.png")
    chk = final_checks(Xtr, Xte, names)
    print(f"[3D] NaN count: train = {chk['nan_train']}, test = {chk['nan_test']}")
    print(f"[3D] final shape: train = {chk['shape_train']}, test = {chk['shape_test']}")
    print(f"[3D] scaled numerical columns: train min = {chk['num_train'][0]:.4f}, "
          f"max = {chk['num_train'][1]:.4f}; test min = {chk['num_test'][0]:.4f}, "
          f"max = {chk['num_test'][1]:.4f}")
    print(f"[3D] whole feature matrix: train min = {chk['all_train'][0]:.4f}, "
          f"max = {chk['all_train'][1]:.4f}; test min = {chk['all_test'][0]:.4f}, "
          f"max = {chk['all_test'][1]:.4f}")
    for col in ["Age", *SPEND_COLS, "TotalSpend"]:      # which column sets the range
        k = names.index(col)
        print(f"[3D] scaled {col}: train [{Xtr[:, k].min():.4f}, {Xtr[:, k].max():.4f}], "
              f"test [{Xte[:, k].min():.4f}, {Xte[:, k].max():.4f}]")
    print(f"[3D] raw Age range: train [{X_train['Age'].min():.0f}, {X_train['Age'].max():.0f}], "
          f"test [{X_test['Age'].min():.0f}, {X_test['Age'].max():.0f}]")

    # ---- Values for the Results summary (rows 10-13) ---------------------------
    print(f"[SUMMARY 10] share of Transported = True (full file): "
          f"{100 * bal.loc[True, 'share']:.2f}%")
    print(f"[SUMMARY 11] FoodCourt on train, before transforming: mean = "
          f"{fc.mean():.2f}, median = {fc.median():.2f}")
    print(f"[SUMMARY 12] training feature matrix shape: {chk['shape_train']}")
    print(f"[SUMMARY 13] after scaling: train [{chk['num_train'][0]:.4f}, "
          f"{chk['num_train'][1]:.4f}], test [{chk['num_test'][0]:.4f}, "
          f"{chk['num_test'][1]:.4f}] (scaled numerical columns)")
    return chk
# --8<-- [end:ex3]
