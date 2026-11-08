"""Single-layer perceptron written from scratch with NumPy only.

The same class is used, unchanged, by Exercise 1 (separable data) and
Exercise 2 (overlapping data). Exercise 2 adds the pocket algorithm through
the ``on_update`` hook, so not a single line of this file differs between the
two exercises.
"""

import numpy as np


# --8<-- [start:perceptron]
class Perceptron:
    """Binary perceptron with labels in {0, 1}: y_hat = step(w . x + b)."""

    def __init__(self, w=None, b=0.0):
        # Weights can be given directly (e.g. to evaluate saved pocket weights);
        # train() overwrites them with the requested initialization.
        self.w = None if w is None else np.asarray(w, dtype=float).copy()
        self.b = float(b)
        self.history = {"accuracy": [], "updates": []}
        self.epochs_ = 0

    # --8<-- [start:predict]
    @staticmethod
    def step(z):
        """Heaviside activation: 1 where z >= 0, 0 otherwise (scalar or array)."""
        return np.where(z >= 0, 1, 0)

    def net_input(self, X):
        """Pre-activation z = w . x + b, for one sample (shape (2,)) or many (n, 2)."""
        return X @ self.w + self.b

    def predict(self, X):
        """Class prediction y_hat in {0, 1}."""
        return self.step(self.net_input(X))

    def accuracy(self, X, y):
        """Fraction of samples in (X, y) that the current (w, b) classifies correctly."""
        return float(np.mean(self.predict(X) == y))
    # --8<-- [end:predict]

    # --8<-- [start:train]
    def train(self, X, y, eta, max_epochs, w0, b0=0.0, on_update=None):
        """Online perceptron training on (X, y), visiting samples in the given order.

        eta        -- learning rate.
        max_epochs -- hard cap on the number of passes over the data.
        w0, b0     -- initial weights and bias (w0 is drawn by the caller from the
                      report's single rng, so this class never creates randomness).
        on_update  -- optional callback on_update(model, epoch, i, error), called
                      right after every actual update (Exercise 2: pocket copy).
        """
        # Start from the given initialization (copied so the caller's w0 is untouched).
        self.w = np.asarray(w0, dtype=float).copy()
        self.b = float(b0)
        self.history = {"accuracy": [], "updates": []}

        for epoch in range(1, max_epochs + 1):
            n_updates = 0
            # One epoch = one pass over the samples, always in the same fixed order.
            for i in range(len(X)):
                x_i, y_i = X[i], int(y[i])
                y_hat = int(self.predict(x_i))      # current prediction for this sample
                error = y_i - y_hat                 # 0 if correct, +1 (missed a 1), -1 (false 1)

                # --8<-- [start:update]
                if error != 0:
                    # Error-driven update for {0, 1} labels:
                    #   w <- w + eta (y - y_hat) x,   b <- b + eta (y - y_hat)
                    # error = +1 pulls w towards x; error = -1 pushes w away from x.
                    self.w = self.w + eta * error * x_i
                    self.b = self.b + eta * error
                    n_updates += 1
                    if on_update is not None:
                        on_update(self, epoch, i, error)
                # --8<-- [end:update]

            # Bookkeeping: full-dataset accuracy and number of updates after every epoch.
            self.history["accuracy"].append(self.accuracy(X, y))
            self.history["updates"].append(n_updates)
            self.epochs_ = epoch

            # Stopping rule: a full pass without a single update means every sample
            # is classified correctly, so the weights can never change again.
            if n_updates == 0:
                break
        return self
    # --8<-- [end:train]
# --8<-- [end:perceptron]
