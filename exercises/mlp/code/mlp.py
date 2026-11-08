"""A multi-layer perceptron written from scratch with NumPy.

One class serves Exercises 2, 3 and 4. Nothing about the architecture is
hard-coded: the layer sizes, the hidden activation and the output head are
constructor arguments.

    output="sigmoid"  ->  1 sigmoid unit  + binary cross-entropy       (Exercise 2)
    output="softmax"  ->  K softmax units + categorical cross-entropy  (Exercises 3, 4)

Both heads are the *canonical* pairing of activation and loss, for which the
gradient of the loss with respect to the output pre-activation is the same
expression, dL/du = (y_hat - y) / N. That is why forward(), backward() and
step() below contain no branch on the task: Exercises 2, 3 and 4 run exactly
the same lines of code.

Conventions
-----------
* Samples are rows: X has shape (N, d). Targets Y have shape (N, K), where
  K = 1 for the sigmoid head and Y is one-hot for the softmax head.
* W[l] has shape (n_out, n_in): row i holds the weights of neuron i, as in
  Exercise 1 (z = W x + b). For a batch of row vectors this is Z = X W^T + b.
* Layer l = 0 .. n_layers-1 maps activation A[l] to A[l+1]; A[0] = X.
"""

import numpy as np


# --8<-- [start:activations]
# ---------------------------------------------------------------------------
# Activations and their derivatives. Each derivative is written in terms of the
# activation's OUTPUT a = f(z), which is what the forward pass caches.
# ---------------------------------------------------------------------------
def tanh(z):
    return np.tanh(z)


def tanh_prime_from_output(a):
    return 1.0 - a ** 2                    # d/dz tanh(z) = 1 - tanh(z)^2


def relu(z):
    return np.maximum(0.0, z)


def relu_prime_from_output(a):
    return (a > 0).astype(a.dtype)         # 1 where the unit is active, else 0


HIDDEN = {"tanh": (tanh, tanh_prime_from_output),
          "relu": (relu, relu_prime_from_output)}


def sigmoid(u):
    # Numerically stable logistic function: never exponentiates a large positive number.
    out = np.empty_like(u)
    pos = u >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-u[pos]))
    e = np.exp(u[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def softmax(u):
    # Subtracting the row maximum leaves softmax unchanged and avoids overflow.
    e = np.exp(u - u.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
# --8<-- [end:activations]


# --8<-- [start:losses]
# ---------------------------------------------------------------------------
# Losses, averaged over the N samples. Probabilities are clipped away from 0
# and 1 so that log() is always finite.
# ---------------------------------------------------------------------------
EPS = 1e-12


def binary_cross_entropy(y_hat, Y):
    p = np.clip(y_hat, EPS, 1.0 - EPS)
    return float(-np.mean(Y * np.log(p) + (1.0 - Y) * np.log(1.0 - p)))


def categorical_cross_entropy(y_hat, Y):
    p = np.clip(y_hat, EPS, 1.0)
    return float(-np.mean(np.sum(Y * np.log(p), axis=1)))


# Output head = (output activation, matching loss). With these canonical
# pairings the output delta is always (y_hat - Y) / N  (see backward()).
OUTPUT = {"sigmoid": (sigmoid, binary_cross_entropy),
          "softmax": (softmax, categorical_cross_entropy)}
# --8<-- [end:losses]


class MLP:
    # --8<-- [start:init]
    def __init__(self, layer_sizes, hidden_activation="tanh", output="sigmoid", rng=None):
        """layer_sizes = [n_inputs, n_hidden_1, ..., n_hidden_k, n_outputs]."""
        if len(layer_sizes) < 3:
            raise ValueError("an MLP needs at least one hidden layer")
        if output == "sigmoid" and layer_sizes[-1] != 1:
            raise ValueError("the sigmoid head has exactly one output unit")
        self.layer_sizes = list(layer_sizes)
        self.n_layers = len(layer_sizes) - 1           # number of weight matrices
        self.act, self.act_prime = HIDDEN[hidden_activation]
        self.out_act, self.loss_fn = OUTPUT[output]
        self.output = output
        rng = np.random.default_rng() if rng is None else rng

        # Xavier/Glorot-normal initialisation: Var(w) = 2 / (n_in + n_out) keeps the
        # variance of the signal roughly constant from layer to layer for tanh.
        # Biases start at zero; the random weights already break the symmetry.
        self.W, self.b = [], []
        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            std = np.sqrt(2.0 / (n_in + n_out))
            self.W.append(rng.normal(0.0, std, size=(n_out, n_in)))
            self.b.append(np.zeros(n_out))

    def n_parameters(self):
        """Total number of trainable scalars (weights + biases)."""
        return int(sum(W.size + b.size for W, b in zip(self.W, self.b)))
    # --8<-- [end:init]

    # --8<-- [start:forward]
    def forward(self, X):
        """Forward pass. Caches every layer's activation for backward()."""
        A = [X]                                   # A[0] is the input itself
        for l in range(self.n_layers):
            Z = A[l] @ self.W[l].T + self.b[l]    # pre-activation, shape (N, n_out)
            if l < self.n_layers - 1:
                A.append(self.act(Z))             # hidden layer: tanh (or relu)
            else:
                A.append(self.out_act(Z))         # output layer: sigmoid or softmax
        self._cache = A
        return A[-1]                              # y_hat
    # --8<-- [end:forward]

    # --8<-- [start:loss]
    def loss(self, y_hat, Y):
        """BCE or categorical CE, chosen by the `output` constructor argument."""
        return self.loss_fn(y_hat, Y)
    # --8<-- [end:loss]

    # --8<-- [start:backward]
    def backward(self, Y):
        """Backward pass. Returns (dW, db): one gradient array per layer.

        Must be called right after forward(X) on the same batch.
        """
        A = self._cache
        N = Y.shape[0]
        dW = [None] * self.n_layers
        db = [None] * self.n_layers

        # Output delta dL/dU for the last layer. For sigmoid+BCE AND for
        # softmax+categorical-CE this is (y_hat - Y) / N  (derived in the report),
        # so the same line serves the binary and the multi-class network.
        delta = (A[-1] - Y) / N

        for l in range(self.n_layers - 1, -1, -1):
            # Z_l = A_l W_l^T + b_l  =>  dL/dW_l = delta^T A_l,  dL/db_l = sum over samples.
            dW[l] = delta.T @ A[l]
            db[l] = delta.sum(axis=0)
            if l > 0:
                # Send the error back through W_l to the previous activation,
                # then through that layer's activation derivative (element-wise).
                delta = (delta @ self.W[l]) * self.act_prime(A[l])
        return dW, db
    # --8<-- [end:backward]

    # --8<-- [start:step]
    def step(self, dW, db, lr):
        """Plain gradient descent on every weight matrix and bias vector."""
        for l in range(self.n_layers):
            self.W[l] -= lr * dW[l]
            self.b[l] -= lr * db[l]
    # --8<-- [end:step]

    # --8<-- [start:fit]
    def fit(self, X, Y, epochs, lr):
        """Full-batch gradient descent. Returns the training loss of every epoch.

        The loss recorded for epoch t is computed on the forward pass of that
        epoch, i.e. just before the epoch's parameter update.
        """
        history = np.empty(epochs)
        for t in range(epochs):
            y_hat = self.forward(X)               # 1. forward pass (caches activations)
            history[t] = self.loss(y_hat, Y)      # 2. record the training loss
            dW, db = self.backward(Y)             # 3. gradients of every layer
            self.step(dW, db, lr)                 # 4. gradient-descent update
        return history
    # --8<-- [end:fit]

    # --8<-- [start:predict]
    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        """Class labels: threshold 0.5 for the sigmoid head, argmax for softmax."""
        p = self.forward(X)
        if self.output == "sigmoid":
            return (p[:, 0] >= 0.5).astype(int)
        return p.argmax(axis=1)
    # --8<-- [end:predict]


def n_parameters(layer_sizes):
    """Trainable scalars of an MLP with these layer sizes: (n_in + 1) * n_out per layer."""
    return int(sum((a + 1) * b for a, b in zip(layer_sizes[:-1], layer_sizes[1:])))


def one_hot(y, n_classes):
    """Integer labels (N,) -> one-hot matrix (N, n_classes)."""
    Y = np.zeros((y.shape[0], n_classes))
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


def gradient_check(model, X, Y, eps=1e-6):
    """Compare backward() with central finite differences on EVERY parameter.

    Returns the largest absolute difference between the analytic and the
    numeric gradient. Used in main.py as a unit test of the hand-written
    backward pass, for both output heads. It perturbs one scalar at a time and
    restores it, so the model is left unchanged; it draws no random numbers.
    """
    model.forward(X)
    dW, db = model.backward(Y)
    worst = 0.0
    for P, G in list(zip(model.W, dW)) + list(zip(model.b, db)):
        for idx in np.ndindex(P.shape):
            old = P[idx]
            P[idx] = old + eps
            L_plus = model.loss(model.forward(X), Y)
            P[idx] = old - eps
            L_minus = model.loss(model.forward(X), Y)
            P[idx] = old
            numeric = (L_plus - L_minus) / (2 * eps)
            worst = max(worst, abs(numeric - G[idx]))
    return worst
