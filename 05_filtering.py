# ============================================================
# Filtering
#
# 1. Average Blur
# 2. Gaussian Blur
# 3. Median Filter and Noise Removal
# 4. Sharpening
# 5. Unsharp Masking
# ============================================================

import cv2
import numpy as np

image = cv2.imread("images/cat.png")

# 1. Average Blur: All nearby pixels have equal weight
average_blur = cv2.blur(
    image,
    (5, 5)  # Kernel: 5 columns × 5 rows = 25 pixel positions
)

# 2. Gaussian Blur: Center pixels have more importance
gaussian_blur = cv2.GaussianBlur(
    image,
    (5, 5),  # Kernel: 5 columns × 5 rows = 25 pixel positions
    0        # Automatically calculate Gaussian sigma
)

# Blur strength (approximate):
# (1, 1)   = No blur
# (3, 3)   = Light blur
# (5, 5)   = Moderate blur
# (9, 9)   = Strong blur
# (15, 15) = Very strong blur


# 3. Median Filter: Remove noise using the middle value of nearby pixels
median_filtered = cv2.medianBlur(
    image,
    5  # Kernel: 5 × 5 neighboring pixel positions
)

# Kernel sizes (approximate):
# 3 = Light filtering
# 5 = Moderate filtering
# 7 = Stronger filtering
# 9 = Even stronger filtering


# Demonstration: Add noise and then remove it
noisy = image.copy()

noise = np.random.random(image.shape[:2])

noisy[noise < 0.05] = 0       # Add random black dots
noisy[noise > 0.95] = 255     # Add random white dots

noise_removed = cv2.medianBlur(noisy, 5)


# 4. Sharpening: Enhance image details
sharpened = cv2.detailEnhance(
    image,
    sigma_s=10,     # Size of the surrounding area considered
    sigma_r=0.15    # Sensitivity to color and brightness differences
)


# 5. Unsharp Masking

# Step 1: Create a blurred version of the original image
blurred = cv2.GaussianBlur(
    image,
    (5, 5),  # Kernel: 5 columns × 5 rows
    0        # Automatically calculate Gaussian sigma
)

# Step 2: Sharpen by subtracting the blurred image
unsharp = cv2.addWeighted(
    image, 1.5,      # Original image × 1.5
    blurred, -0.5,   # Blurred image × -0.5
    0               # No brightness adjustment
)


# Display results
cv2.imshow("Original", image)
cv2.imshow("1. Average Blur", average_blur)
cv2.imshow("2. Gaussian Blur", gaussian_blur)
cv2.imshow("3. Median Filter", median_filtered)
cv2.imshow("3. Noisy Image", noisy)
cv2.imshow("3. Noise Removed", noise_removed)
cv2.imshow("4. Sharpening", sharpened)
cv2.imshow("5. Unsharp Masking", unsharp)

cv2.waitKey(0)
cv2.destroyAllWindows()