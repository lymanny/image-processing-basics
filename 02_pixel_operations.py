# ============================================================
# Pixel Operations
#
# 1. Grayscale Conversion
# 2. Brightness and Contrast Adjustment
# 3. Saturation Arithmetic
# 4. Inversion
# 5. Thresholding
# ============================================================

import cv2


# ============================================================
# Read Image
# ============================================================

image = cv2.imread("images/skin.png")


# ============================================================
# 1. Grayscale Conversion
# ============================================================

# Convert BGR color image to 1-channel grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ============================================================
# 2. Brightness and Contrast Adjustment
# ============================================================

# alpha = contrast
# beta = brightness
adjusted = cv2.convertScaleAbs(
    image,
    alpha=1.2,
    beta=15
)


# ============================================================
# 3. Saturation Arithmetic
# ============================================================

# Add 30 to make the image brighter and use saturation
# arithmetic to keep pixel values between 0 and 255
brighter = cv2.add(image, 30)


# ============================================================
# 4. Inversion
# ============================================================

# Change each pixel to its opposite value
# New pixel = 255 - original pixel
inverted = cv2.bitwise_not(image)


# ============================================================
# 5. Thresholding
# ============================================================

# Thresholding uses the 1-channel grayscale image
# 127 = cutoff value
# 255 = white value
_, binary = cv2.threshold(
    gray,
    127,
    255,
    cv2.THRESH_BINARY
)


# ============================================================
# Show Results
# ============================================================

cv2.imshow("Original", image)
cv2.imshow("Grayscale", gray)
cv2.imshow("Brightness and Contrast", adjusted)
cv2.imshow("Saturation Arithmetic", brighter)
cv2.imshow("Inverted", inverted)
cv2.imshow("Threshold", binary)

# Keep all windows open until you press any key
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()