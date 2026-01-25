import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img_path = r'C:\Users\HP\OneDrive\Documents\GitHub\ET3112_Assignment_01\images\looking_out.jpg'
img = cv2.imread(img_path)

if img is None:
    print("Error: Image not found!")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Part (a): Otsu thresholding ---
# We use THRESH_BINARY_INV because the woman/room are DARK (low values)
# and we want them to be the 'foreground' (white in the mask).
threshold_value, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# --- Part (b): Histogram equalization on foreground only ---
equalized_img = gray.copy()

# Extract only the foreground pixels
foreground_pixels = gray[binary_mask == 255]

# Equalize only those pixels
equalized_foreground = cv2.equalizeHist(foreground_pixels)

# Put the enhanced pixels back into the image
equalized_img[binary_mask == 255] = equalized_foreground.flatten()

# --- Visualization ---
plt.figure(figsize=(16, 8))

plt.subplot(1, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title('Original Grayscale')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(binary_mask, cmap='gray')
plt.title(f'Otsu Foreground Mask\nThreshold = {threshold_value}')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(equalized_img, cmap='gray')
plt.title('Enhanced Foreground')
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Reported Threshold Value: {threshold_value}")