import tensorflow as tf
import matplotlib.pyplot as plt

# ============================================================
# a. LOAD AND PREPROCESS MNIST DATASET
# ============================================================

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values from [0, 255] to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten 28x28 images into 784-dimensional vectors
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

print("Training data shape:", x_train.shape)
print("Testing data shape :", x_test.shape)


# ============================================================
# b. DESIGN NEURAL NETWORK
# ============================================================

def create_model(optimizer):

    model = tf.keras.Sequential([
        # Input layer: 784 neurons (28 x 28 pixels)
        tf.keras.layers.Input(shape=(784,)),

        # One hidden layer with ReLU activation
        tf.keras.layers.Dense(128, activation="relu"),

        # Output layer: 10 neurons with Softmax
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# c & d. TRAIN USING DIFFERENT OPTIMIZERS
# ============================================================

EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001

# Same network, epochs, batch size and learning rate are used
# for every experiment. ONLY the optimizer is changed.

optimizers = {

    "SGD with Momentum":
        tf.keras.optimizers.SGD(
            learning_rate=LEARNING_RATE,
            momentum=0.9
        ),

    "Adagrad":
        tf.keras.optimizers.Adagrad(
            learning_rate=LEARNING_RATE
        ),

    "RMSprop":
        tf.keras.optimizers.RMSprop(
            learning_rate=LEARNING_RATE
        ),

    "Adam":
        tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        )
}


histories = {}
models = {}


for name, optimizer in optimizers.items():

    print("\n" + "=" * 50)
    print("Training using:", name)
    print("=" * 50)

    model = create_model(optimizer)

    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,

        # 10% of training data is used for validation
        validation_split=0.1,

        verbose=1
    )

    histories[name] = history.history
    models[name] = model


# ============================================================
# e. COMPARE TRAINING/VALIDATION ACCURACY AND LOSS
# ============================================================

print("\n\nFINAL RESULTS")
print("=" * 80)

print(
    f"{'Optimizer':<22}"
    f"{'Train Acc':<15}"
    f"{'Val Acc':<15}"
    f"{'Train Loss':<15}"
    f"{'Val Loss':<15}"
)

print("-" * 80)

for name, history in histories.items():

    train_acc = history["accuracy"][-1]
    val_acc = history["val_accuracy"][-1]
    train_loss = history["loss"][-1]
    val_loss = history["val_loss"][-1]

    print(
        f"{name:<22}"
        f"{train_acc:<15.4f}"
        f"{val_acc:<15.4f}"
        f"{train_loss:<15.4f}"
        f"{val_loss:<15.4f}"
    )


# ============================================================
# f. PLOT ACCURACY VS EPOCH
# ============================================================

plt.figure(figsize=(10, 6))

for name, history in histories.items():

    plt.plot(
        history["accuracy"],
        label=name
    )

plt.title("Training Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy")
plt.legend()
plt.grid()
plt.show()


# Validation accuracy plot
plt.figure(figsize=(10, 6))

for name, history in histories.items():

    plt.plot(
        history["val_accuracy"],
        label=name
    )

plt.title("Validation Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.grid()
plt.show()


# ============================================================
# LOSS VS EPOCH
# ============================================================

# Training loss
plt.figure(figsize=(10, 6))

for name, history in histories.items():

    plt.plot(
        history["loss"],
        label=name
    )

plt.title("Training Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend()
plt.grid()
plt.show()


# Validation loss
plt.figure(figsize=(10, 6))

for name, history in histories.items():

    plt.plot(
        history["val_loss"],
        label=name
    )

plt.title("Validation Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.legend()
plt.grid()
plt.show()


# ============================================================
# g. IDENTIFY FASTEST CONVERGING OPTIMIZER
# ============================================================

# We define convergence as reaching 95% training accuracy.
# The optimizer that reaches this accuracy in the fewest epochs
# is considered the fastest converging optimizer.

target_accuracy = 0.95

convergence_epochs = {}

for name, history in histories.items():

    accuracy = history["accuracy"]

    reached = [
        epoch + 1
        for epoch, acc in enumerate(accuracy)
        if acc >= target_accuracy
    ]

    if reached:
        convergence_epochs[name] = reached[0]
    else:
        convergence_epochs[name] = float("inf")


fastest_optimizer = min(
    convergence_epochs,
    key=convergence_epochs.get
)

print("\n" + "=" * 60)
print("CONVERGENCE ANALYSIS")
print("=" * 60)

for name, epoch in convergence_epochs.items():

    if epoch == float("inf"):
        print(f"{name:<22}: Did not reach 95% accuracy")
    else:
        print(f"{name:<22}: Reached 95% accuracy at epoch {epoch}")


print("\nFastest converging optimizer:", fastest_optimizer)


# ============================================================
# JUSTIFICATION
# ============================================================

# Adam will commonly converge fastest in this experiment because
# it combines Momentum and RMSprop-like adaptive learning rates.
# Momentum helps accelerate movement in useful directions, while
# adaptive learning rates adjust the step size for each parameter.
#
# HOWEVER, the code above determines the actual fastest optimizer
# from the experiment rather than assuming that Adam must win.
# The optimizer reaching 95% training accuracy in the fewest epochs
# is reported as the fastest converging optimizer.
