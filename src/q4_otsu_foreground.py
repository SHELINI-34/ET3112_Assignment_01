import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread(r'C:\Users\HP\OneDrive\Documents\GitHub\ET3112_Assignment_01\images\looking_out.jpg')
if img is None:
    print("Error: Image not found!")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Part (a): Otsu thresholding
threshold_value, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Part (b): Histogram equalization on foreground only
equalized_img = gray.copy()
equalized_full = cv2.equalizeHist(gray)
equalized_img[binary_mask == 255] = equalized_full[binary_mask == 255]

# Print results
print(f"Part (a) - Otsu Threshold Value: {threshold_value:.2f}")
print(f"Part (b) - Histogram equalization applied to foreground")

# Figure: Main results
plt.figure(figsize=(16, 8))

plt.subplot(2, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title('(a) Original Grayscale', fontsize=14, fontweight='bold')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(binary_mask, cmap='gray')
plt.title(f'(b) Binary Mask\nThreshold = {threshold_value:.2f}', fontsize=14, fontweight='bold')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(gray, cmap='gray')
plt.title('(c) Before Enhancement', fontsize=14, fontweight='bold')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(equalized_img, cmap='gray')
plt.title('(d) After Enhancement', fontsize=14, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.show()

# Histogram comparison
foreground_original = gray[binary_mask == 255]
foreground_equalized = equalized_img[binary_mask == 255]

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(foreground_original, bins=256, range=[0, 256], color='blue', alpha=0.7)
plt.title('Histogram - Before', fontsize=13, fontweight='bold')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(foreground_equalized, bins=256, range=[0, 256], color='blue', alpha=0.7)
plt.title('Histogram - After', fontsize=13, fontweight='bold')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Print hidden features
print("\nHidden features revealed:")
print("- Facial features and hair texture visible")
print("- Clothing details and fabric texture emerge")
print("- Doorway and wall details enhanced")
print("- Shadow regions show tonal variation")
print("- Overall improved contrast and depth")