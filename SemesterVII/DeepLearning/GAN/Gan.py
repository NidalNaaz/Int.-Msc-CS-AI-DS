import tensorflow as tf
from tensorflow.keras import layers, Sequential
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load and Preprocess MNIST
# -----------------------------
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Normalize images to [-1,1]
x_train = (x_train.astype("float32") - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=-1)

BUFFER_SIZE = 60000
BATCH_SIZE = 256

dataset = tf.data.Dataset.from_tensor_slices(x_train)
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# -----------------------------
# Generator
# -----------------------------
def build_generator():

    model = Sequential([
        layers.Dense(7*7*256, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Reshape((7,7,256)),

        layers.Conv2DTranspose(128, (5,5), strides=(1,1),
                               padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(64, (5,5), strides=(2,2),
                               padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(1, (5,5), strides=(2,2),
                               padding='same', use_bias=False,
                               activation='tanh')
    ])

    return model

# -----------------------------
# Discriminator
# -----------------------------
def build_discriminator():

    model = Sequential([
        layers.Conv2D(64, (5,5), strides=(2,2),
                      padding='same',
                      input_shape=(28,28,1)),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5,5), strides=(2,2),
                      padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)
    ])

    return model

generator = build_generator()
discriminator = build_discriminator()

# -----------------------------
# Loss Functions
# -----------------------------
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

# -----------------------------
# Optimizers
# -----------------------------
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# -----------------------------
# Training Step
# -----------------------------
noise_dim = 100

gen_losses = []
disc_losses = []

@tf.function
def train_step(images):

    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_gen = gen_tape.gradient(
        gen_loss,
        generator.trainable_variables)

    gradients_disc = disc_tape.gradient(
        disc_loss,
        discriminator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(gradients_gen,
            generator.trainable_variables))

    discriminator_optimizer.apply_gradients(
        zip(gradients_disc,
            discriminator.trainable_variables))

    return gen_loss, disc_loss

# -----------------------------
# Training Loop
# -----------------------------
EPOCHS = 50

for epoch in range(EPOCHS):

    g_loss_epoch = []
    d_loss_epoch = []

    for image_batch in dataset:

        # Skip incomplete batch
        if image_batch.shape[0] != BATCH_SIZE:
            continue

        g_loss, d_loss = train_step(image_batch)

        g_loss_epoch.append(g_loss.numpy())
        d_loss_epoch.append(d_loss.numpy())

    gen_losses.append(np.mean(g_loss_epoch))
    disc_losses.append(np.mean(d_loss_epoch))

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"G Loss: {gen_losses[-1]:.4f} | "
          f"D Loss: {disc_losses[-1]:.4f}")

# -----------------------------
# Generate Images
# -----------------------------
noise = tf.random.normal([16, noise_dim])
generated_images = generator(noise, training=False)

generated_images = (generated_images + 1) / 2

plt.figure(figsize=(6,6))

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(generated_images[i,:,:,0], cmap='gray')
    plt.axis('off')

plt.suptitle("Generated Handwritten Digits")
plt.show()

# -----------------------------
# Generator Loss Plot
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(gen_losses, label='Generator Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Generator Loss")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Discriminator Loss Plot
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(disc_losses, label='Discriminator Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Discriminator Loss")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Final Loss Values
# -----------------------------
print("\nTraining Complete")
print("Final Generator Loss:", gen_losses[-1])
print("Final Discriminator Loss:", disc_losses[-1])
