"""Single entry point of the "1. Data" report.

Run from the repository root:

    python docs/exercises/data/code/main.py

It creates the report's ONE random generator, rng = np.random.default_rng(42),
and passes it to the three exercises in statement order (1 -> 2 -> 3). The
exercise modules never create a generator of their own, so every figure in
../figures/ and every number printed below is identical on every run.

Exercise 3 needs Kaggle's Spaceship Titanic train.csv in code/raw/. When the
file is missing, Exercises 1 and 2 still run and the script prints how to get it.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files only: no window, same PNGs everywhere

import numpy as np  # noqa: E402

from ex1_point_clouds import ex1  # noqa: E402
from ex2_nonlinearity import ex2  # noqa: E402
from ex3_spaceship import KAGGLE_URL, ex3  # noqa: E402

CODE_DIR = Path(__file__).resolve().parent
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
CSV_PATH = CODE_DIR / "raw" / "train.csv"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)       # the single generator of the report

    ex1(rng, FIG_DIR)                     # Figures 1-3 (+ the Figure 1 sketch)
    ex2(rng, FIG_DIR)                     # Figures 4-5

    if CSV_PATH.exists():
        ex3(rng, FIG_DIR, CSV_PATH)       # Figure 6
    else:
        print("=" * 70)
        print("EXERCISE 3 SKIPPED - train.csv not found")
        print("=" * 70)
        print(f"Expected file: {CSV_PATH}")
        print("To get it (requires a Kaggle account):")
        print(f"  1. Open {KAGGLE_URL}")
        print("  2. Sign in, open the 'Data' tab and accept the competition rules.")
        print("  3. Download train.csv (or the zip and extract train.csv from it).")
        print(f"  4. Save it as {CSV_PATH.relative_to(CODE_DIR.parents[3]).as_posix()}")
        print("     (the folder code/raw/ is git-ignored), then run this script again.")
        print("  CLI alternative (needs a Kaggle API token): "
              "kaggle competitions download -c spaceship-titanic")
        print("     downloads spaceship-titanic.zip; extract train.csv from it.")


if __name__ == "__main__":
    main()
