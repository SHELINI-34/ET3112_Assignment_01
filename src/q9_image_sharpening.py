import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ================== 1. LOAD IMAGE ==================
# Using your specific path
img_path = r'C:\Users\HP\OneDrive\Documents\GitHub\ET3112_Assignment_01\images\taylor.jpg'
img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Image not found! Please check the file path.")
    exit()

# ================== 2. LAPLACIAN SHARPENING ==================

# Step A: Compute the Laplacian (Second Derivative)

laplacian = cv.Laplacian(img, cv.CV_64F)

# Step B: Sharpen by subtracting the Laplacian from the original

sharpened_lap = img - laplacian

# Step C: Post-processing

sharpened_lap = np.clip(sharpened_lap, 0, 255).astype(np.uint8)

# ================== 3. DISPLAY RESULTS ==================
plt.figure(figsize=(15, 5))

# Original Plot
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original (Smooth)')
plt.axis('off')

# Laplacian Plot (Showing the edges detected)
plt.subplot(1, 3, 2)
plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian (Edge Map)')
plt.axis('off')

# Final Sharpened Result
plt.subplot(1, 3, 3)
plt.imshow(sharpened_lap, cmap='gray')
plt.title('Final Sharpened Image')
plt.axis('off')

plt.tight_layout()
plt.show()

print("Sharpening process complete. Images displayed for comparison.")
