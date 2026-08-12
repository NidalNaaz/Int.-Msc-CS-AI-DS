import numpy as np
import time

# ============================================================
# 1. CREATE DATASET
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
# 3. BATCH GD
# ============================================================

def batch_gd(X, y, max_updates=20000, lr=0.01):

    n = len(X)

    w = np.zeros((X.shape[1], 1))
    b = 0.0

    start = time.perf_counter()

    for update in range(max_updates):

        predictions = X @ w + b
        error = predictions - y

        dw = (2 / n) * (X.T @ error)
        db = (2 / n) * np.sum(error)

        w -= lr * dw
        b -= lr * db

    end = time.perf_counter()

    loss = mse_loss(X, y, w, b)

    return end - start, loss, max_updates


# ============================================================
# 4. STOCHASTIC GD
# ============================================================

def stochastic_gd(X, y, max_updates=20000, lr=0.01):

    n = len(X)

    w = np.zeros((X.shape[1], 1))
    b = 0.0

    start = time.perf_counter()

    for update in range(max_updates):

        i = np.random.randint(0, n)

        xi = X[i:i+1]
        yi = y[i:i+1]

        prediction = xi @ w + b
        error = prediction - yi

        dw = 2 * (xi.T @ error)
        db = 2 * np.sum(error)

        w -= lr * dw
        b -= lr * db

    end = time.perf_counter()

    loss = mse_loss(X, y, w, b)

    return end - start, loss, max_updates


# ============================================================
# 5. MINI-BATCH GD
# ============================================================

def mini_batch_gd(X, y, batch_size=64, max_updates=20000, lr=0.01):

    n = len(X)

    w = np.zeros((X.shape[1], 1))
    b = 0.0

    start = time.perf_counter()

    for update in range(max_updates):

        indices = np.random.randint(0, n, batch_size)

        X_batch = X[indices]
        y_batch = y[indices]

        predictions = X_batch @ w + b
        error = predictions - y_batch

        dw = (2 / batch_size) * (X_batch.T @ error)
        db = (2 / batch_size) * np.sum(error)

        w -= lr * dw
        b -= lr * db

    end = time.perf_counter()

    loss = mse_loss(X, y, w, b)

    return end - start, loss, max_updates


# ============================================================
# 6. RUN EXPERIMENT
# ============================================================

print("Running Batch Gradient Descent...")
batch_time, batch_loss, batch_updates = batch_gd(X, y)

print("Running Stochastic Gradient Descent...")
sgd_time, sgd_loss, sgd_updates = stochastic_gd(X, y)

print("Running Mini-Batch Gradient Descent...")
mini_time, mini_loss, mini_updates = mini_batch_gd(X, y)


# ============================================================
# 7. RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FAIR COMPARISON - SAME NUMBER OF PARAMETER UPDATES")
print("=" * 70)

print(
    f"{'Method':<25}"
    f"{'Time (s)':<15}"
    f"{'Final Loss':<15}"
    f"{'Updates'}"
)

print("-" * 70)

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
