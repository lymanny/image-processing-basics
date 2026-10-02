# ============================================================
# Color Spaces
#
# 1. HSV
# 2. Lab
# 3. Per-channel Adjustment
# 4. Color Range Masking
# ============================================================

import cv2
import numpy as np

image = cv2.imread("images/skin.png")


# ============================================================
# 1. HSV
# ============================================================

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

print("HSV pixel:", hsv[100, 100])


# ============================================================
# 2. Lab
# ============================================================

lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

print("Lab pixel:", lab[100, 100])


# ============================================================
# 3. Per-channel Adjustment
# ============================================================

l, a, b = cv2.split(lab)

# Per-channel Adjustment: increase only L (Lightness) by 30
l = cv2.add(l, 30)

adjusted_lab = cv2.merge((l, a, b))

adjusted = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)


# ============================================================
# 4. Color Range Masking
# ============================================================

lower = np.array([165, 60, 40])
upper = np.array([179, 255, 255])

# Color Range Masking: inside = white (255), outside = black (0)
mask = cv2.inRange(hsv, lower, upper)

# White areas keep original colors; black areas become black
result = cv2.bitwise_and(image, image, mask=mask)


# ============================================================
# Show Results
# ============================================================

cv2.imshow("Original", image)
cv2.imshow("Adjusted Lightness", adjusted)
cv2.imshow("Mask", mask)
cv2.imshow("Selected Red Areas", result)

cv2.waitKey(0)
cv2.destroyAllWindows()