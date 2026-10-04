# ============================================================
# Geometric Transformations
#
# 1. Resize and Interpolation
# 2. Translation
# 3. Rotation
# 4. Flip
# 5. Crop
# 6. Random Crop
# 7. Application to Medical Images
# ============================================================

import cv2
import numpy as np
import random

image = cv2.imread("images/cat.png")

height, width = image.shape[:2]

# 1. Resize and Interpolation
resized = cv2.resize(
    image,
    (224, 224),
    interpolation=cv2.INTER_AREA  # Good for shrinking images
)

# 2. Translation
translation_matrix = np.float32([
    [1, 0, 50],  # X: +50 = right, -50 = left
    [0, 1, 30]   # Y: +30 = down, -30 = up
])

translated = cv2.warpAffine(
    image,
    translation_matrix,
    (width + 50, height + 30)  # Expand canvas to keep the full image
)

# 3. Rotation
rotation_matrix = cv2.getRotationMatrix2D(
    (width / 2, height / 2),  # Image center
    45,                       # Angle: + = counterclockwise, - = clockwise
                              # Examples: 30, 45, 90, 180, -45
    0.5                       # Scale: 1 = original, 0.5 = smaller, 2 = larger
)

rotated = cv2.warpAffine(
    image,
    rotation_matrix,
    (width, height)  # Keep original canvas size
)

# 4. Flip
flipped_horizontal = cv2.flip(image, 1)  # Left ↔ Right
flipped_vertical = cv2.flip(image, 0)    # Up ↕ Down
flipped_both = cv2.flip(image, -1)       # Both directions

# 5. Crop
# Select rows 100–299 and columns 100–299
cropped = image[100:300, 100:300]

# 6. Random Crop
crop_height = 200
crop_width = 200

# Choose a random starting position
random_y = random.randint(0, height - crop_height)
random_x = random.randint(0, width - crop_width)

# Cut out a random 200 × 200 area
random_cropped = image[
    random_y:random_y + crop_height,
    random_x:random_x + crop_width
]

# 7. Application to Medical Images
# Resize the skin image for a CNN requiring 224 × 224 input
medical_resized = cv2.resize(
    image,
    (224, 224),
    interpolation=cv2.INTER_AREA
)

# Create an augmented image using horizontal flipping
medical_augmented = cv2.flip(medical_resized, 1)

# Display results
cv2.imshow("Original", image)
cv2.imshow("1. Resized", resized)
cv2.imshow("2. Translated", translated)
cv2.imshow("3. Rotated", rotated)
cv2.imshow("4. Horizontal Flip", flipped_horizontal)
cv2.imshow("4. Vertical Flip", flipped_vertical)
cv2.imshow("4. Both Flip", flipped_both)
cv2.imshow("5. Cropped", cropped)
cv2.imshow("6. Random Crop", random_cropped)
cv2.imshow("7. Medical Image Augmentation", medical_augmented)

cv2.waitKey(0)
cv2.destroyAllWindows()