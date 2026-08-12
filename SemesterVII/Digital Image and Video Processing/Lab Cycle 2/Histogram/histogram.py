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
# 2. ORIGINAL HISTOGRAM
# ---------------------------------------------------------

original_hist = cv2.calcHist([image], [0], None, [256], [0, 256])


# ---------------------------------------------------------
# 3. HISTOGRAM EQUALIZATION
# ---------------------------------------------------------

equalized = cv2.equalizeHist(image)

equalized_hist = cv2.calcHist(
    [equalized], [0], None, [256], [0, 256]
)


# ---------------------------------------------------------
# 4. CREATE REFERENCE IMAGE
# ---------------------------------------------------------
# We create a reference image from the same image by
# changing its intensity distribution.

reference = cv2.convertScaleAbs(
    image,
    alpha=1.5,
    beta=20
)

reference_hist = cv2.calcHist(
    [reference], [0], None, [256], [0, 256]
)


# ---------------------------------------------------------
# 5. HISTOGRAM MATCHING
# ---------------------------------------------------------

# Calculate normalized cumulative histograms

source_hist = cv2.calcHist(
    [image], [0], None, [256], [0, 256]
).flatten()

target_hist = reference_hist.flatten()

source_cdf = np.cumsum(source_hist)
target_cdf = np.cumsum(target_hist)

source_cdf = source_cdf / source_cdf[-1]
target_cdf = target_cdf / target_cdf[-1]


# Create mapping between source and target intensities

mapping = np.zeros(256, dtype=np.uint8)

for i in range(256):

    difference = np.abs(target_cdf - source_cdf[i])

    mapping[i] = np.argmin(difference)


# Apply mapping

matched = mapping[image]

matched_hist = cv2.calcHist(
    [matched], [0], None, [256], [0, 256]
)


# ---------------------------------------------------------
# 6. ENTROPY FUNCTION
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


# Calculate entropy

original_entropy = calculate_entropy(image)
equalized_entropy = calculate_entropy(equalized)
reference_entropy = calculate_entropy(reference)
matched_entropy = calculate_entropy(matched)


# ---------------------------------------------------------
# 7. PRINT RESULTS
# ---------------------------------------------------------

print("\n========== ENTROPY COMPARISON ==========")

print(f"Original Image   : {original_entropy:.4f} bits")
print(f"Equalized Image  : {equalized_entropy:.4f} bits")
print(f"Reference Image  : {reference_entropy:.4f} bits")
print(f"Matched Image    : {matched_entropy:.4f} bits")


# ---------------------------------------------------------
# 8. DISPLAY IMAGES
# ---------------------------------------------------------

plt.figure(figsize=(14, 10))


plt.subplot(3, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")


plt.subplot(3, 3, 2)
plt.imshow(equalized, cmap="gray")
plt.title("Histogram Equalized")
plt.axis("off")


plt.subplot(3, 3, 3)
plt.imshow(reference, cmap="gray")
plt.title("Reference")
plt.axis("off")


plt.subplot(3, 3, 4)
plt.plot(original_hist)
plt.title("Original Histogram")
plt.xlim([0, 256])


plt.subplot(3, 3, 5)
plt.plot(equalized_hist)
plt.title("Equalized Histogram")
plt.xlim([0, 256])


plt.subplot(3, 3, 6)
plt.plot(reference_hist)
plt.title("Reference Histogram")
plt.xlim([0, 256])


plt.subplot(3, 3, 7)
plt.imshow(matched, cmap="gray")
plt.title("Histogram Matched")
plt.axis("off")


plt.subplot(3, 3, 8)
plt.plot(matched_hist)
plt.title("Matched Histogram")
plt.xlim([0, 256])


plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 9. SAVE OUTPUTS
# ---------------------------------------------------------

cv2.imwrite("original.png", image)
cv2.imwrite("equalized.png", equalized)
cv2.imwrite("reference.png", reference)
cv2.imwrite("matched.png", matched)

print("\nOutput files saved:")
print("original.png")
print("equalized.png")
print("reference.png")
print("matched.png")
