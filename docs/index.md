# Artificial Neural Networks & Deep Learning

???+ info inline end "Edition"

    **2026.2**

    [Course statements :material-open-in-new:](https://insper.github.io/ann-dl/2026.2/){:target='_blank'}

This site is the **portfolio** of my deliverables for Insper's *Artificial Neural Networks and
Deep Learning* course. Each exercise is one menu item; the repository that builds it holds the
report source, the code as runnable files, and the git history.

## Identification

The **exercises are individual**; the project team is listed on the [project page](projects/index.md).

| Full name | E-mail | GitHub |
|-----------|--------|--------|
| Marcelo Alonso | <mrclvtalonso@gmail.com> | [Marcelo-vta](https://github.com/Marcelo-vta) |

## Deliverables

### Exercises — individual

- [ ] [1. Data](exercises/data/index.md) — synthetic point clouds, non-linearity in 5-D, preprocessing a real dataset
- [ ] [2. Perceptron](exercises/perceptron/index.md) — a from-scratch perceptron on separable and overlapping data
- [ ] [3. MLP](exercises/mlp/index.md) — backpropagation by hand and a from-scratch MLP
- [ ] [4. VAE](exercises/vae/index.md) — a variational autoencoder on MNIST

### [Project](projects/index.md) — team

- [ ] [EDA](projects/eda/index.md)
- [ ] [Classification](projects/classification/index.md) **or** [Regression](projects/regression/index.md)
- [ ] [Generative](projects/generative/index.md)

## Re-running the code

Every figure and number in the reports is produced by the scripts under
`docs/exercises/<slug>/code/`. From the repository root:

``` bash
python -m venv env
source env/bin/activate            # Windows: .\env\Scripts\activate
python -m pip install -r requirements.txt

python docs/exercises/data/code/main.py
python docs/exercises/perceptron/code/main.py
python docs/exercises/mlp/code/main.py
python docs/exercises/vae/code/main.py
```

All seeds are fixed, so a re-run reproduces the reported numbers. The Data exercise also needs
the Kaggle *Spaceship Titanic* `train.csv` (see that report for where to put it).
