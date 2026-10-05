# ============================================================
# Edge Detection
#
# 1. Sobel
# 2. Laplacian
# 3. Canny
# 4. Finding Boundaries / Contours
# ============================================================

import cv2

image = cv2.imread("images/cat.png")

# Convert BGR image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# 1. Sobel
# Sobel X: Detect brightness changes left ↔ right
# Mainly highlights vertical edges
sobel_x = cv2.Sobel(
    gray,
    cv2.CV_64F,  # Keep positive and negative edge values
    1, 0,        # X direction
    ksize=3      # Use a 3 × 3 area to detect edges, not a blur level
)

# Sobel Y: Detect brightness changes top ↕ bottom
# Mainly highlights horizontal edges
sobel_y = cv2.Sobel(
    gray,
    cv2.CV_64F,
    0, 1,        # Y direction
    ksize=3
)

# Convert Sobel results so they can be displayed clearly
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)


# 2. Laplacian
# Detect edges in all directions
laplacian = cv2.Laplacian(
    gray,
    cv2.CV_64F  # Keep positive and negative edge values
)

# Convert result for display
laplacian = cv2.convertScaleAbs(laplacian)


# 3. Canny
# Detect clear and thin edges
canny = cv2.Canny(
    gray,
    100,  # Lower threshold
    200   # Upper threshold
)

# Canny rule:
# Below 100     = Remove
# 100 to 200    = Maybe keep if connected to a strong edge
# Above 200     = Keep


# 4. Finding Boundaries / Contours
# Find connected boundaries from the Canny edges
contours, _ = cv2.findContours(
    canny,
    cv2.RETR_EXTERNAL,      # Keep outer boundaries
    cv2.CHAIN_APPROX_SIMPLE # Store contour points efficiently
)

# Create a copy so the original image is not changed
boundary_image = image.copy()

# Draw all detected contours
cv2.drawContours(
    boundary_image,
    contours,
    -1,          # Draw all contours
    (0, 255, 0), # Green boundary lines
    2            # Line thickness
)


# Display results
cv2.imshow("Original", image)

cv2.imshow("1. Sobel X", sobel_x)
cv2.imshow("1. Sobel Y", sobel_y)

cv2.imshow("2. Laplacian", laplacian)

cv2.imshow("3. Canny", canny)

cv2.imshow("4. Boundaries = Contours", boundary_image)

cv2.waitKey(0)
cv2.destroyAllWindows()