import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    return img


def show_images(images, titles, cmap="gray"):
    n = len(images)

    plt.figure(figsize=(5 * n, 4))

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, n, i + 1)
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
# A. IMAGE NEGATIVE
# ============================================================

# 1. Image Negative
def image_negative(img):
    return 255 - img


img = load_gray("image.jpg")

negative = image_negative(img)

show_images(
    [img, negative],
    ["Original Image", "Negative Image"]
)


# 2 & 3. Histogram comparison

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(img.ravel(), bins=256, range=(0, 256))
plt.title("Original Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(negative.ravel(), bins=256, range=(0, 256))
plt.title("Negative Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ============================================================
# B. LOG TRANSFORMATION
# ============================================================

# 4. Log Transformation
def log_transform(img, c=1):
    img_float = img.astype(np.float32)

    transformed = c * np.log1p(img_float)

    # Normalize to 0-255
    transformed = cv2.normalize(
        transformed,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return transformed.astype(np.uint8)


log_img = log_transform(img, c=1)

show_images(
    [img, log_img],
    ["Original Image", "Log Transformation"]
)


# 5 & 6. Different values of c

c_values = [0.5, 1, 2, 5]

plt.figure(figsize=(12, 8))

for i, c in enumerate(c_values):

    result = log_transform(img, c)

    plt.subplot(2, 2, i + 1)
    plt.imshow(result, cmap="gray")
    plt.title(f"Log Transformation, c={c}")
    plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# C. POWER-LAW TRANSFORMATION
# ============================================================

# 7. Power-law / Gamma transformation
def power_law_transform(img, gamma, c=1):

    img_float = img.astype(np.float32) / 255.0

    transformed = c * np.power(img_float, gamma)

    transformed = np.clip(transformed * 255, 0, 255)

    return transformed.astype(np.uint8)


gammas = [0.2, 0.5, 1.0, 2.0, 5.0]

plt.figure(figsize=(15, 6))

for i, gamma in enumerate(gammas):

    result = power_law_transform(img, gamma)

    plt.subplot(1, len(gammas), i + 1)
    plt.imshow(result, cmap="gray")
    plt.title(f"γ = {gamma}")
    plt.axis("off")

plt.tight_layout()
plt.show()


# 8 & 9. Gamma comparison
gamma_less_than_1 = power_law_transform(img, 0.5)
gamma_greater_than_1 = power_law_transform(img, 2.0)

show_images(
    [img, gamma_less_than_1, gamma_greater_than_1],
    ["Original", "Gamma < 1", "Gamma > 1"]
)


# ============================================================
# D. SPATIAL FILTERING
# ============================================================

# Create noisy images

# Gaussian noise
gaussian_noise = np.random.normal(
    0, 25, img.shape
).astype(np.float32)

gaussian_noisy = img.astype(np.float32) + gaussian_noise
gaussian_noisy = np.clip(gaussian_noisy, 0, 255).astype(np.uint8)


# Salt-and-pepper noise
sp_noisy = img.copy()

probability = 0.05

random_matrix = np.random.rand(*img.shape)

sp_noisy[random_matrix < probability / 2] = 0
sp_noisy[random_matrix > 1 - probability / 2] = 255


# 11. Mean, Median and Gaussian filters

mean_filtered = cv2.blur(gaussian_noisy, (5, 5))

median_filtered = cv2.medianBlur(sp_noisy, 5)

gaussian_filtered = cv2.GaussianBlur(
    gaussian_noisy,
    (5, 5),
    0
)


show_images(
    [
        gaussian_noisy,
        mean_filtered,
        gaussian_filtered
    ],
    [
        "Gaussian Noise",
        "Mean Filter",
        "Gaussian Filter"
    ]
)


show_images(
    [
        sp_noisy,
        median_filtered
    ],
    [
        "Salt & Pepper Noise",
        "Median Filter"
    ]
)


# ============================================================
# 12. CUSTOM SHARPENING FILTER
# ============================================================

sharpen_kernel = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
])

sharpened = cv2.filter2D(
    img,
    -1,
    sharpen_kernel
)

show_images(
    [img, sharpened],
    ["Original", "Sharpened"]
)


# ============================================================
# 13. LAPLACIAN OPERATORS
# ============================================================

laplacian_4 = np.array([
    [ 0,  1,  0],
    [ 1, -4,  1],
    [ 0,  1,  0]
])

laplacian_8 = np.array([
    [ 1,  1,  1],
    [ 1, -8,  1],
    [ 1,  1,  1]
])


edge_4 = cv2.filter2D(img, cv2.CV_64F, laplacian_4)
edge_8 = cv2.filter2D(img, cv2.CV_64F, laplacian_8)

edge_4 = np.uint8(np.absolute(edge_4))
edge_8 = np.uint8(np.absolute(edge_8))


show_images(
    [img, edge_4, edge_8],
    ["Original", "Laplacian 4-connected", "Laplacian 8-connected"]
)


# ============================================================
# E. IMAGE ENHANCEMENT
# ============================================================

# 14. IMAGE SUBTRACTION
# ------------------------------------------------------------

# Simulate a second image
img2 = img.copy()

# Add an artificial object/change
cv2.rectangle(
    img2,
    (50, 50),
    (150, 150),
    255,
    -1
)

difference = cv2.absdiff(img, img2)

show_images(
    [img, img2, difference],
    ["Before", "After", "Detected Change"]
)


# ============================================================
# 15. IMAGE WATERMARKING
# ============================================================

watermark = np.zeros_like(img)

cv2.putText(
    watermark,
    "WATERMARK",
    (30, 100),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    255,
    2
)

# Add watermark
watermarked = cv2.addWeighted(
    img,
    0.8,
    watermark,
    0.2,
    0
)

show_images(
    [img, watermark, watermarked],
    ["Original", "Watermark", "Watermarked Image"]
)


# ============================================================
# 16. IMAGE AVERAGING
# ============================================================

# Generate multiple noisy images

num_images = 10

noisy_images = []

for i in range(num_images):

    noise = np.random.normal(
        0,
        25,
        img.shape
    )

    noisy = img.astype(np.float32) + noise

    noisy = np.clip(
        noisy,
        0,
        255
    ).astype(np.uint8)

    noisy_images.append(noisy)


# Average images
average_image = np.mean(
    noisy_images,
    axis=0
).astype(np.uint8)


show_images(
    [
        noisy_images[0],
        noisy_images[4],
        average_image
    ],
    [
        "Noisy Image 1",
        "Noisy Image 5",
        "Averaged Image"
    ]
)


print("\nLAB CYCLE 1 COMPLETED SUCCESSFULLY!")
