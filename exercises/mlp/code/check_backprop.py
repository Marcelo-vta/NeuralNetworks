"""Independent numerical check of Exercise 1 (one step of backpropagation).

The deliverable for Exercise 1 is the derivation written by hand in the report.
This script does NOT replace it: it recomputes the same quantities with NumPy so
that every number in the report can be verified, and it checks each analytic
gradient against a central finite difference.

Run on its own from the repository root:
    python docs/exercises/mlp/code/check_backprop.py
(main.py also calls run() so that one command prints everything.)
"""

import numpy as np

# --8<-- [start:params]
# Values given by the statement. Row i of W1 holds the weights of hidden neuron i,
# so the hidden pre-activation is z1 = W1 @ x + b1.
x = np.array([0.5, -0.2])
y = 1.0
W1 = np.array([[0.3, -0.1],
               [0.2, 0.4]])
b1 = np.array([0.1, -0.2])
W2 = np.array([0.5, -0.3])
b2 = 0.2
eta = 0.3
# --8<-- [end:params]


# --8<-- [start:forward_backward]
def forward(W1, b1, W2, b2):
    """Forward pass: z1 -> h1 -> u2 -> y_hat -> L (MSE with N = 1)."""
    z1 = W1 @ x + b1             # hidden pre-activations, shape (2,)
    h1 = np.tanh(z1)             # hidden activations
    u2 = W2 @ h1 + b2            # output pre-activation (scalar)
    y_hat = np.tanh(u2)          # network output
    L = (y - y_hat) ** 2         # squared error of the single sample
    return z1, h1, u2, y_hat, L


def backward(h1, y_hat, W2):
    """Analytic gradients, chained outward from dL/dy_hat."""
    dL_dyhat = -2.0 * (y - y_hat)            # d/dy_hat of (y - y_hat)^2
    dL_du2 = dL_dyhat * (1.0 - y_hat ** 2)   # tanh'(u2) = 1 - tanh(u2)^2
    dL_dW2 = dL_du2 * h1                     # u2 = W2 . h1 + b2
    dL_db2 = dL_du2
    dL_dh1 = dL_du2 * W2                     # each h1_j feeds u2 through W2_j
    dL_dz1 = dL_dh1 * (1.0 - h1 ** 2)        # element-wise tanh'
    dL_dW1 = np.outer(dL_dz1, x)             # z1_i = sum_j W1_ij x_j + b1_i
    dL_db1 = dL_dz1
    return dict(dL_dyhat=dL_dyhat, dL_du2=dL_du2, dL_dW2=dL_dW2, dL_db2=dL_db2,
                dL_dh1=dL_dh1, dL_dz1=dL_dz1, dL_dW1=dL_dW1, dL_db1=dL_db1)
# --8<-- [end:forward_backward]


def finite_differences(eps=1e-6):
    """Central differences (L(p + eps) - L(p - eps)) / (2 eps) for every parameter."""
    names, numeric = [], []
    params = {"W1": W1, "b1": b1, "W2": W2}
    for name, P in params.items():
        for idx in np.ndindex(P.shape):
            plus, minus = P.copy(), P.copy()
            plus[idx] += eps
            minus[idx] -= eps
            args_p = {k: (plus if k == name else v) for k, v in params.items()}
            args_m = {k: (minus if k == name else v) for k, v in params.items()}
            Lp = forward(args_p["W1"], args_p["b1"], args_p["W2"], b2)[-1]
            Lm = forward(args_m["W1"], args_m["b1"], args_m["W2"], b2)[-1]
            names.append(f"{name}{list(idx)}")
            numeric.append((Lp - Lm) / (2 * eps))
    Lp = forward(W1, b1, W2, b2 + eps)[-1]
    Lm = forward(W1, b1, W2, b2 - eps)[-1]
    names.append("b2")
    numeric.append((Lp - Lm) / (2 * eps))
    return names, np.array(numeric)


def run():
    """Print every Exercise 1 number (6 decimals) and the finite-difference check."""
    f6 = lambda v: np.array2string(np.asarray(v, dtype=float), precision=6,
                                   floatmode="fixed", suppress_small=False)
    print("=" * 72)
    print("EXERCISE 1 - numerical check of the hand derivation")
    print("=" * 72)

    # A - forward pass
    z1, h1, u2, y_hat, L = forward(W1, b1, W2, b2)
    print("[Ex1-A] z1                 =", f6(z1))
    print("[Ex1-A] exp(2*z1)          =", f6(np.exp(2 * z1)))
    print("[Ex1-A] h1 = tanh(z1)      =", f6(h1))
    print("[Ex1-A] u2                 =", f6(u2))
    print("[Ex1-A] exp(2*u2)          =", f6(np.exp(2 * u2)))
    print("[Ex1-A] y_hat = tanh(u2)   =", f6(y_hat))
    print("[Ex1-A] y - y_hat          =", f6(y - y_hat))
    print("[Ex1-A] L                  =", f6(L))

    # B - backward pass
    g = backward(h1, y_hat, W2)
    print("[Ex1-B] 1 - y_hat^2        =", f6(1 - y_hat ** 2))
    print("[Ex1-B] 1 - h1^2           =", f6(1 - h1 ** 2))
    for key in ["dL_dyhat", "dL_du2", "dL_dW2", "dL_db2", "dL_dh1", "dL_dz1", "dL_db1"]:
        print(f"[Ex1-B] {key:<18} =", f6(g[key]))
    print("[Ex1-B] dL_dW1             =", f6(g["dL_dW1"]).replace("\n", "\n" + " " * 29))

    # Finite-difference check of all 9 scalar gradients
    analytic = np.concatenate([g["dL_dW1"].ravel(), g["dL_db1"], g["dL_dW2"], [g["dL_db2"]]])
    names, numeric = finite_differences()
    print("[Ex1-B] finite-difference check (central, eps = 1e-6):")
    for n, a, num in zip(names, analytic, numeric):
        print(f"          {n:<9} analytic = {a: .8f}   numeric = {num: .8f}   |diff| = {abs(a - num):.1e}")
    max_diff = np.max(np.abs(analytic - numeric))
    print(f"[Ex1-B] max |analytic - numeric| = {max_diff:.2e}")
    assert max_diff < 1e-8, "analytic and numeric gradients disagree"

    # C - gradient-descent update with eta = 0.3
    W2_new = W2 - eta * g["dL_dW2"]
    b2_new = b2 - eta * g["dL_db2"]
    W1_new = W1 - eta * g["dL_dW1"]
    b1_new = b1 - eta * g["dL_db1"]
    print("[Ex1-C] W2 updated         =", f6(W2_new))
    print("[Ex1-C] b2 updated         =", f6(b2_new))
    print("[Ex1-C] W1 updated         =", f6(W1_new).replace("\n", "\n" + " " * 29))
    print("[Ex1-C] b1 updated         =", f6(b1_new))

    # Second forward pass with the updated parameters
    z1n, h1n, u2n, yhn, Ln = forward(W1_new, b1_new, W2_new, b2_new)
    print("[Ex1-C] second pass z1     =", f6(z1n))
    print("[Ex1-C] second pass h1     =", f6(h1n))
    print("[Ex1-C] second pass u2     =", f6(u2n))
    print("[Ex1-C] second pass y_hat  =", f6(yhn))
    print("[Ex1-C] second pass L      =", f6(Ln))
    print(f"[Ex1-C] loss change        = {Ln - L:+.6f}  ({100 * (Ln - L) / L:+.2f}%)")
    # First-order prediction of the change: dL ~ -eta * ||grad||^2
    sq_norm = np.sum(analytic ** 2)
    print(f"[Ex1-C] ||grad||^2         = {sq_norm:.6f}")
    print(f"[Ex1-C] first-order dL     = -eta*||grad||^2 = {-eta * sq_norm:+.6f}")
    return dict(z1=z1, h1=h1, u2=u2, y_hat=y_hat, L=L, grads=g,
                W1_new=W1_new, b1_new=b1_new, W2_new=W2_new, b2_new=b2_new, L_new=Ln)


if __name__ == "__main__":
    run()
