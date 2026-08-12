import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. LOAD IMAGE
# ---------------------------------------------------------

image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: image.jpg not found.")
    exit()

print("Image loaded successfully.")
print("Image size:", image.shape)


# ---------------------------------------------------------
# 2. CONTRAST STRETCHING
# ---------------------------------------------------------

min_val = np.min(image)
max_val = np.max(image)

contrast_stretched = (
    (image - min_val) * 255.0 / (max_val - min_val)
)

contrast_stretched = np.clip(
    contrast_stretched, 0, 255
).astype(np.uint8)


# ---------------------------------------------------------
# 3. CLAHE
# ---------------------------------------------------------

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

clahe_image = clahe.apply(image)


# ---------------------------------------------------------
# 4. ENTROPY FUNCTION
# ---------------------------------------------------------

def calculate_entropy(img):

    histogram = cv2.calcHist(
        [img], [0], None, [256], [0, 256]
    ).flatten()

    probabilities = histogram / histogram.sum()

    probabilities = probabilities[
        probabilities > 0
    ]

    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    return entropy


# ---------------------------------------------------------
# 5. CALCULATE METRICS
# ---------------------------------------------------------

original_entropy = calculate_entropy(image)
stretch_entropy = calculate_entropy(contrast_stretched)
clahe_entropy = calculate_entropy(clahe_image)

original_std = np.std(image)
stretch_std = np.std(contrast_stretched)
clahe_std = np.std(clahe_image)


# ---------------------------------------------------------
# 6. PRINT RESULTS
# ---------------------------------------------------------

print("\n========== CONTRAST ENHANCEMENT RESULTS ==========")

print("\nEntropy:")
print(f"Original           : {original_entropy:.4f} bits")
print(f"Contrast Stretching: {stretch_entropy:.4f} bits")
print(f"CLAHE              : {clahe_entropy:.4f} bits")

print("\nIntensity Standard Deviation:")
print(f"Original           : {original_std:.4f}")
print(f"Contrast Stretching: {stretch_std:.4f}")
print(f"CLAHE              : {clahe_std:.4f}")


# ---------------------------------------------------------
# 7. HISTOGRAMS
# ---------------------------------------------------------

original_hist = cv2.calcHist(
    [image], [0], None, [256], [0, 256]
)

stretch_hist = cv2.calcHist(
    [contrast_stretched], [0], None, [256], [0, 256]
)

clahe_hist = cv2.calcHist(
    [clahe_image], [0], None, [256], [0, 256]
)


# ---------------------------------------------------------
# 8. DISPLAY RESULTS
# ---------------------------------------------------------

plt.figure(figsize=(14, 9))


# Original

plt.subplot(2, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")


# Contrast stretching

plt.subplot(2, 3, 2)
plt.imshow(contrast_stretched, cmap="gray")
plt.title("Contrast Stretching")
plt.axis("off")


# CLAHE

plt.subplot(2, 3, 3)
plt.imshow(clahe_image, cmap="gray")
plt.title("CLAHE")
plt.axis("off")


# Original histogram

plt.subplot(2, 3, 4)
plt.plot(original_hist)
plt.title("Original Histogram")
plt.xlim([0, 256])


# Stretching histogram

plt.subplot(2, 3, 5)
plt.plot(stretch_hist)
plt.title("Contrast Stretching Histogram")
plt.xlim([0, 256])


# CLAHE histogram

plt.subplot(2, 3, 6)
plt.plot(clahe_hist)
plt.title("CLAHE Histogram")
plt.xlim([0, 256])


plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 9. SAVE OUTPUTS
# ---------------------------------------------------------

cv2.imwrite(
    "contrast_stretched.png",
    contrast_stretched
)

cv2.imwrite(
    "clahe.png",
    clahe_image
)

print("\nOutput files saved:")
print("contrast_stretched.png")
print("clahe.png")
