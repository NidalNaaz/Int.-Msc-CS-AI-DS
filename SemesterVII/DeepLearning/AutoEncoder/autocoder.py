import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# --------------------------------------------------
# 1. Load MNIST dataset
# --------------------------------------------------
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension: (60000, 28, 28, 1)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# --------------------------------------------------
# 2. Add Gaussian noise
# --------------------------------------------------
noise_factor = 0.5

noise_train = np.random.normal(
    loc=0.0,
    scale=noise_factor,
    size=x_train.shape
)

noise_test = np.random.normal(
    loc=0.0,
    scale=noise_factor,
    size=x_test.shape
)

x_train_noisy = x_train + noise_train
x_test_noisy = x_test + noise_test

# Clip values back to [0, 1]
x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)

# --------------------------------------------------
# 3. Construct the Denoising Autoencoder
# --------------------------------------------------
model = models.Sequential([

    # Encoder
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2), padding="same"),

    layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2), padding="same"),

    # Decoder
    layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    layers.UpSampling2D((2, 2)),

    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.UpSampling2D((2, 2)),

    # Reconstruct original image
    layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")
])

model.summary()

# --------------------------------------------------
# 4. Compile the model
# --------------------------------------------------
model.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)

# --------------------------------------------------
# 5. Train
# --------------------------------------------------
history = model.fit(
    x_train_noisy,
    x_train,
    epochs=10,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test_noisy, x_test)
)

# --------------------------------------------------
# 6. Reconstruct / denoise test images
# --------------------------------------------------
decoded_images = model.predict(x_test_noisy)

# --------------------------------------------------
# 7. Display Original / Noisy / Denoised
# --------------------------------------------------
n = 10

plt.figure(figsize=(20, 6))

for i in range(n):

    # Original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap="gray")
    plt.axis("off")
    if i == 0:
        ax.set_title("Original")

    # Noisy
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test_noisy[i].squeeze(), cmap="gray")
    plt.axis("off")
    if i == 0:
        ax.set_title("Noisy")

    # Denoised
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(decoded_images[i].squeeze(), cmap="gray")
    plt.axis("off")
    if i == 0:
        ax.set_title("Denoised")

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 8. Plot training loss
# --------------------------------------------------
plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Denoising Autoencoder Training")
plt.legend()
plt.grid()

plt.show()
