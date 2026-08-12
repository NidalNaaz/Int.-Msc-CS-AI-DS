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

image = image.astype(np.float32)

M, N = image.shape

print("Image size:", M, "x", N)


# ---------------------------------------------------------
# 2. COMPUTE 2D DFT
# ---------------------------------------------------------

F = np.fft.fft2(image)

# Move low frequencies to the center
F_shifted = np.fft.fftshift(F)

# Magnitude spectrum for visualization
spectrum = np.log(1 + np.abs(F_shifted))


# ---------------------------------------------------------
# 3. CREATE DISTANCE MATRIX
# ---------------------------------------------------------

center_x = M // 2
center_y = N // 2

Y, X = np.ogrid[:M, :N]

distance = np.sqrt(
    (X - center_y) ** 2 +
    (Y - center_x) ** 2
)


# ---------------------------------------------------------
# 4. LOW-PASS FILTER
# ---------------------------------------------------------

# Radius of low-frequency region
D0 = 40

low_pass_mask = np.zeros((M, N), np.float32)

low_pass_mask[distance <= D0] = 1


# ---------------------------------------------------------
# 5. HIGH-PASS FILTER
# ---------------------------------------------------------

high_pass_mask = np.ones((M, N), np.float32)

high_pass_mask[distance <= D0] = 0


# ---------------------------------------------------------
# 6. BAND-PASS FILTER
# ---------------------------------------------------------

D_low = 20
D_high = 80

band_pass_mask = np.zeros((M, N), np.float32)

band_pass_mask[
    (distance >= D_low) &
    (distance <= D_high)
] = 1


# ---------------------------------------------------------
# 7. APPLY FILTERS
# ---------------------------------------------------------

low_pass_result = F_shifted * low_pass_mask

high_pass_result = F_shifted * high_pass_mask

band_pass_result = F_shifted * band_pass_mask


# ---------------------------------------------------------
# 8. INVERSE DFT
# ---------------------------------------------------------

low_pass_image = np.fft.ifft2(
    np.fft.ifftshift(low_pass_result)
)

high_pass_image = np.fft.ifft2(
    np.fft.ifftshift(high_pass_result)
)

band_pass_image = np.fft.ifft2(
    np.fft.ifftshift(band_pass_result)
)


# Take magnitude because IDFT can contain tiny
# numerical imaginary components

low_pass_image = np.abs(low_pass_image)

high_pass_image = np.abs(high_pass_image)

band_pass_image = np.abs(band_pass_image)


# Normalize for display

low_pass_image = cv2.normalize(
    low_pass_image,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

high_pass_image = cv2.normalize(
    high_pass_image,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

band_pass_image = cv2.normalize(
    band_pass_image,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)


# ---------------------------------------------------------
# 9. DISPLAY RESULTS
# ---------------------------------------------------------

plt.figure(figsize=(14, 10))


plt.subplot(2, 4, 1)

plt.imshow(
    image,
    cmap="gray"
)

plt.title("Original Image")

plt.axis("off")


plt.subplot(2, 4, 2)

plt.imshow(
    spectrum,
    cmap="gray"
)

plt.title("Frequency Spectrum")

plt.axis("off")


plt.subplot(2, 4, 3)

plt.imshow(
    low_pass_mask,
    cmap="gray"
)

plt.title("Low-Pass Mask")

plt.axis("off")


plt.subplot(2, 4, 4)

plt.imshow(
    low_pass_image,
    cmap="gray"
)

plt.title("Low-Pass Result")

plt.axis("off")


plt.subplot(2, 4, 5)

plt.imshow(
    high_pass_mask,
    cmap="gray"
)

plt.title("High-Pass Mask")

plt.axis("off")


plt.subplot(2, 4, 6)

plt.imshow(
    high_pass_image,
    cmap="gray"
)

plt.title("High-Pass Result")

plt.axis("off")


plt.subplot(2, 4, 7)

plt.imshow(
    band_pass_mask,
    cmap="gray"
)

plt.title("Band-Pass Mask")

plt.axis("off")


plt.subplot(2, 4, 8)

plt.imshow(
    band_pass_image,
    cmap="gray"
)

plt.title("Band-Pass Result")

plt.axis("off")


plt.tight_layout()

plt.show()


# ---------------------------------------------------------
# 10. SAVE OUTPUTS
# ---------------------------------------------------------

cv2.imwrite(
    "low_pass.png",
    low_pass_image
)

cv2.imwrite(
    "high_pass.png",
    high_pass_image
)

cv2.imwrite(
    "band_pass.png",
    band_pass_image
)

cv2.imwrite(
    "low_pass_mask.png",
    (low_pass_mask * 255).astype(np.uint8)
)

cv2.imwrite(
    "high_pass_mask.png",
    (high_pass_mask * 255).astype(np.uint8)
)

cv2.imwrite(
    "band_pass_mask.png",
    (band_pass_mask * 255).astype(np.uint8)
)

print("\nFiltering completed.")

print("Saved:")
print("low_pass.png")
print("high_pass.png")
print("band_pass.png")
print("low_pass_mask.png")
print("high_pass_mask.png")
print("band_pass_mask.png")
