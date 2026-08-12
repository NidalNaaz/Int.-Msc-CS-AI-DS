import numpy as np
import time

# ============================================================
# 1. CREATE A LARGE DATASET
# ============================================================

np.random.seed(42)

n_samples = 100_000
n_features = 20

X = np.random.randn(n_samples, n_features)

true_weights = np.random.randn(n_features, 1)
true_bias = 5

noise = np.random.randn(n_samples, 1) * 0.5

y = X @ true_weights + true_bias + noise

# Standardize features
X = (X - X.mean(axis=0)) / X.std(axis=0)


# ============================================================
# 2. LOSS FUNCTION
# ============================================================

def mse_loss(X, y, w, b):
    predictions = X @ w + b
    return np.mean((predictions - y) ** 2)


# ============================================================
# 3. BATCH GRADIENT DESCENT
# ============================================================

def batch_gd(X, y, epochs=20, lr=0.01):

    n = len(X)

    w = np.zeros((X.shape[1], 1))
    b = 0.0

    updates = 0

    start = time.perf_counter()

    for epoch in range(epochs):

        predictions = X @ w + b
        error = predictions - y

        dw = (2 / n) * (X.T @ error)
        db = (2 / n) * np.sum(error)

        w -= lr * dw
        b -= lr * db

        updates += 1

    end = time.perf_counter()

    final_loss = mse_loss(X, y, w, b)

    return end - start, final_loss, updates


# ============================================================
# 4. STOCHASTIC GRADIENT DESCENT
# ============================================================

def stochastic_gd(X, y, epochs=20, lr=0.01):

    n = len(X)

    w = np.zeros((X.shape[1], 1))
    b = 0.0

    updates = 0

    start = time.perf_counter()

    for epoch in range(epochs):

        indices = np.random.permutation(n)

        for i in indices:

            xi = X[i:i+1]
            yi = y[i:i+1]

            prediction = xi @ w + b
            error = prediction - yi

            dw = 2 * (xi.T @ error)
            db = 2 * np.sum(error)

            w -= lr * dw
            b -= lr * db

            updates += 1

    end = time.perf_counter()

    final_loss = mse_loss(X, y, w, b)

    return end - start, final_loss, updates


# ============================================================
# 5. MINI-BATCH GRADIENT DESCENT
# ============================================================

def mini_batch_gd(X, y, batch_size=64, epochs=20, lr=0.01):

    n = len(X)

    w = np.zeros((X.shape[1], 1))
    b = 0.0

    updates = 0

    start = time.perf_counter()

    for epoch in range(epochs):

        indices = np.random.permutation(n)

        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, n, batch_size):

            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            predictions = X_batch @ w + b
            error = predictions - y_batch

            batch_n = len(X_batch)

            dw = (2 / batch_n) * (X_batch.T @ error)
            db = (2 / batch_n) * np.sum(error)

            w -= lr * dw
            b -= lr * db

            updates += 1

    end = time.perf_counter()

    final_loss = mse_loss(X, y, w, b)

    return end - start, final_loss, updates


# ============================================================
# 6. RUN ALL THREE METHODS
# ============================================================

print("Running Batch Gradient Descent...")
batch_time, batch_loss, batch_updates = batch_gd(X, y)

print("Running Stochastic Gradient Descent...")
sgd_time, sgd_loss, sgd_updates = stochastic_gd(X, y)

print("Running Mini-Batch Gradient Descent...")
mini_time, mini_loss, mini_updates = mini_batch_gd(X, y)


# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 65)
print("RESULTS")
print("=" * 65)

print(f"{'Method':<25}{'Time (s)':<15}{'Final Loss':<15}{'Updates'}")
print("-" * 65)

print(
    f"{'Batch GD':<25}"
    f"{batch_time:<15.4f}"
    f"{batch_loss:<15.6f}"
    f"{batch_updates}"
)

print(
    f"{'Stochastic GD':<25}"
    f"{sgd_time:<15.4f}"
    f"{sgd_loss:<15.6f}"
    f"{sgd_updates}"
)

print(
    f"{'Mini-Batch GD (64)':<25}"
    f"{mini_time:<15.4f}"
    f"{mini_loss:<15.6f}"
    f"{mini_updates}"
)
