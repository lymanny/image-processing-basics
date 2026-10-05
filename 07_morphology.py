# ============================================================
# Morphology
#
# 1. Dilation
# 2. Erosion
# 3. Opening
# 4. Closing
# 5. Histogram
# 6. Histogram Equalization
# 7. Image Blending
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1–4. Morphology: Dilation, Erosion, Opening, Closing
# ============================================================

# Read leaf image
image = cv2.imread("images/leaf.png")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert grayscale image to binary
_, binary = cv2.threshold(
    gray,
    127,
    255,
    cv2.THRESH_BINARY
)

# Create morphology kernel
# (3, 3) = small effect
# (5, 5) = stronger effect
# (7, 7) = even stronger effect
kernel = np.ones((3, 3), np.uint8)


# 1. Dilation: Expand white areas
dilated = cv2.dilate(
    binary,
    kernel,
    iterations=1
)


# 2. Erosion: Shrink white areas
eroded = cv2.erode(
    binary,
    kernel,
    iterations=1
)


# 3. Opening
# Erosion → Dilation
# Main purpose: remove small white noise
opening = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel
)


# 4. Closing
# Dilation → Erosion
# Main purpose: fill small black holes or gaps
closing = cv2.morphologyEx(
    binary,
    cv2.MORPH_CLOSE,
    kernel
)


# Display morphology results
cv2.imshow("Original Leaf", image)
cv2.imshow("Binary", binary)
cv2.imshow("1. Dilation", dilated)
cv2.imshow("2. Erosion", eroded)
cv2.imshow("3. Opening", opening)
cv2.imshow("4. Closing", closing)


# ============================================================
# 5. Histogram
# ============================================================

# Calculate grayscale histogram
histogram = cv2.calcHist(
    [gray],     # Input image
    [0],        # Grayscale channel
    None,       # No mask = whole image
    [256],      # 256 brightness values
    [0, 256]    # Brightness range: 0–255
)

plt.figure()
plt.plot(histogram)
plt.title("5. Image Histogram")
plt.xlabel("Brightness Value")
plt.ylabel("Number of Pixels")


# ============================================================
# 6. Histogram Equalization
# ============================================================

# Improve image contrast
equalized = cv2.equalizeHist(gray)

# Calculate histogram after equalization
hist_equalized = cv2.calcHist(
    [equalized],
    [0],
    None,
    [256],
    [0, 256]
)

cv2.imshow("6. Original Grayscale", gray)
cv2.imshow("6. Histogram Equalization", equalized)

# Compare before and after histograms
plt.figure()
plt.plot(histogram, label="Original")
plt.plot(hist_equalized, label="Equalized")
plt.title("6. Histogram Equalization")
plt.xlabel("Brightness Value")
plt.ylabel("Number of Pixels")
plt.legend()


# ============================================================
# 7. Image Blending
# ============================================================

# Read two color images
image1 = cv2.imread("images/cat.png")
image2 = cv2.imread("images/leaf.png")

# Resize image2 to match image1
image2 = cv2.resize(
    image2,
    (image1.shape[1], image1.shape[0])
)

# Blend images
blended = cv2.addWeighted(
    image1, 0.7,  # 70% cat
    image2, 0.3,  # 30% leaf
    0             # No extra brightness
)

cv2.imshow("7. Image 1", image1)
cv2.imshow("7. Image 2", image2)
cv2.imshow("7. Blended", blended)


# Show histogram graphs
plt.show()

# Wait and close OpenCV windows
cv2.waitKey(0)
cv2.destroyAllWindows()