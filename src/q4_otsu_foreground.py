import cv2
import numpy as np
import matplotlib.pyplot as plt

# ================== LOAD IMAGE ==================
img = cv2.imread(
    r'C:\Users\HP\OneDrive\Documents\GitHub\ET3112_Assignment_01\images\looking_out.jpg'
)

if img is None:
    print("Error: Image not found!")
    exit()

# ================== CONVERT TO GRAYSCALE ==================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ================== OTSU THRESHOLDING ==================
# Otsu automatically finds the optimal threshold
threshold_value, binary_mask = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

print("Otsu threshold value:", threshold_value)

# ================== DISPLAY GRAYSCALE & MASK ==================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(binary_mask, cmap='gray')
plt.title('Otsu Binary Mask')
plt.axis('off')

plt.show()

# ================== FOREGROUND EXTRACTION ==================
foreground = cv2.bitwise_and(gray, gray, mask=binary_mask)

# ================== HISTOGRAM EQUALIZATION ON FOREGROUND ==================
hist, bins = np.histogram(foreground[foreground > 0], 256, [0, 256])
cdf = hist.cumsum()
cdf_normalized = cdf / cdf[-1]

equalized_fg = foreground.copy()
equalized_fg[foreground > 0] = np.floor(
    255 * cdf_normalized[foreground[foreground > 0]]
).astype(np.uint8)

# ================== DISPLAY FOREGROUND RESULTS ==================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(foreground, cmap='gray')
plt.title('Foreground (Masked)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(equalized_fg, cmap='gray')
plt.title('Equalized Foreground')
plt.axis('off')

plt.show()
