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

# Resize for manual DFT because direct DFT is computationally
# expensive for large images.
image = cv2.resize(image, (64, 64))

image = image.astype(np.float64)

M, N = image.shape

print("Image size used for DFT:", M, "x", N)


# ---------------------------------------------------------
# 2. IMPLEMENT 2D DFT
# ---------------------------------------------------------

def dft_2d(image):

    M, N = image.shape

    # Row-wise DFT
    row_dft = np.zeros((M, N), dtype=complex)

    for u in range(M):

        for x in range(M):

            row_dft[u, :] += (
                image[x, :]
                * np.exp(-2j * np.pi * u * x / M)
            )

    # Column-wise DFT
    result = np.zeros((M, N), dtype=complex)

    for u in range(M):

        for v in range(N):

            result[u, v] = np.sum(
                row_dft[u, :]
                * np.exp(-2j * np.pi * v * np.arange(N) / N)
            )

    return result


# ---------------------------------------------------------
# 3. IMPLEMENT INVERSE 2D DFT
# ---------------------------------------------------------

def idft_2d(F):

    M, N = F.shape

    # Inverse transform

    result = np.zeros((M, N), dtype=complex)

    for x in range(M):

        for y in range(N):

            total = 0

            for u in range(M):

                for v in range(N):

                    total += (
                        F[u, v]
                        * np.exp(
                            2j * np.pi *
                            (u * x / M + v * y / N)
                        )
                    )

            result[x, y] = total / (M * N)

    return np.real(result)


# ---------------------------------------------------------
# 4. COMPUTE DFT
# ---------------------------------------------------------

print("\nComputing 2D DFT...")

F = dft_2d(image)

print("DFT completed.")


# ---------------------------------------------------------
# 5. COMPUTE INVERSE DFT
# ---------------------------------------------------------

print("Computing inverse DFT...")

reconstructed = idft_2d(F)

reconstructed = np.clip(
    reconstructed, 0, 255
)

print("Inverse DFT completed.")


# ---------------------------------------------------------
# 6. CALCULATE MAGNITUDE SPECTRUM
# ---------------------------------------------------------

F_shifted = np.fft.fftshift(F)

magnitude = np.log(
    1 + np.abs(F_shifted)
)


# ---------------------------------------------------------
# 7. COMPARE WITH NUMPY FFT
# ---------------------------------------------------------

numpy_dft = np.fft.fft2(image)

error = np.mean(
    np.abs(F - numpy_dft)
)

reconstruction_error = np.mean(
    np.abs(image - reconstructed)
)

print("\n========== VERIFICATION ==========")

print(
    f"Difference from NumPy FFT: {error:.10f}"
)

print(
    f"Reconstruction error: {reconstruction_error:.10f}"
)


# ---------------------------------------------------------
# 8. DISPLAY RESULTS
# ---------------------------------------------------------

plt.figure(figsize=(12, 8))


plt.subplot(2, 2, 1)

plt.imshow(
    image,
    cmap="gray"
)

plt.title("Original Image")

plt.axis("off")


plt.subplot(2, 2, 2)

plt.imshow(
    magnitude,
    cmap="gray"
)

plt.title("2D DFT Magnitude Spectrum")

plt.axis("off")


plt.subplot(2, 2, 3)

plt.imshow(
    reconstructed,
    cmap="gray"
)

plt.title("Reconstructed Image using IDFT")

plt.axis("off")


plt.subplot(2, 2, 4)

plt.imshow(
    np.abs(image - reconstructed),
    cmap="gray"
)

plt.title("Reconstruction Error")

plt.axis("off")


plt.tight_layout()

plt.show()


# ---------------------------------------------------------
# 9. SAVE OUTPUTS
# ---------------------------------------------------------

cv2.imwrite(
    "dft_magnitude.png",
    cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)
)

cv2.imwrite(
    "idft_reconstructed.png",
    reconstructed.astype(np.uint8)
)

print("\nOutput files saved:")
print("dft_magnitude.png")
print("idft_reconstructed.png")
