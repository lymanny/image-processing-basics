# ============================================================
# Image Representation
#
# 1. Reading image
# 2. Array structure and dtype
# 3. BGR / RGB channel order
# 4. Accessing and modifying pixel values
# ============================================================

import cv2
import matplotlib.pyplot as plt


# ============================================================
# 1. Read Image
# ============================================================

# OpenCV reads color images in BGR order
image = cv2.imread("images/skin.png")


# ============================================================
# 2. Array Structure and dtype
# ============================================================

# Example: (480, 640, 3)
# 480 = Height
# 640 = Width
# 3 = Channels
print("Shape:", image.shape)

# Example: uint8
# Pixel values usually range from 0 to 255
print("Data type:", image.dtype)


# ============================================================
# 3. BGR / RGB Channel Order
# ============================================================

# Convert BGR to RGB
# Matplotlib expects RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ============================================================
# 4. Accessing and Modifying Pixel Values
# ============================================================

# Access the pixel at row 100, column 100
pixel = image[100, 100]

# Print pixel value in BGR order
print("Pixel value:", pixel)

# Make a copy so the original image stays unchanged
modified_image = image.copy()

# Select rows 100–149 and columns 100–249
# Change the selected area to red
# OpenCV uses BGR, so [0, 0, 255] = Red
modified_image[100:150, 100:250] = [0, 0, 255]


# ============================================================
# 5. Show Before and After
# ============================================================

cv2.imshow("Before", image)
cv2.imshow("After", modified_image)

# Keep windows open until you press any key
cv2.waitKey(0)

# Close OpenCV windows
cv2.destroyAllWindows()