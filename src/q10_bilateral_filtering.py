import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def bilateral_filter_manual(img, diameter, sigma_s, sigma_r):
    h, w = img.shape
    radius = diameter // 2
    output = np.zeros_like(img, dtype=np.float32)
    
    # Add padding to handle edges
    padded_img = cv2.copyMakeBorder(img, radius, radius, radius, radius, cv2.BORDER_REFLECT)

    for i in range(h):
        for j in range(w):
            wp = 0
            filtered_value = 0
            i_pad, j_pad = i + radius, j + radius
            p_intensity = padded_img[i_pad, j_pad]

            for x in range(-radius, radius + 1):
                for y in range(-radius, radius + 1):
                    q_intensity = padded_img[i_pad + x, j_pad + y]
                    
                    # Spatial Weight
                    spatial_dist_sq = x**2 + y**2
                    spatial_weight = math.exp(-spatial_dist_sq / (2 * sigma_s**2))
                    
                    # Range Weight
                    intensity_diff_sq = (float(q_intensity) - float(p_intensity))**2
                    range_weight = math.exp(-intensity_diff_sq / (2 * sigma_r**2))
                    
                    weight = spatial_weight * range_weight
                    wp += weight
                    filtered_value += weight * q_intensity

            output[i, j] = filtered_value / wp

    return np.uint8(np.clip(output, 0, 255))

# ================== LOAD IMAGE ==================
img_path = r'C:\Users\HP\OneDrive\Documents\GitHub\ET3112_Assignment_01\images\taylor.jpg'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Image not found!")
    exit()

# (b) Gaussian Smoothing
gaussian = cv2.GaussianBlur(img, (7, 7), sigmaX=3)

# (c) Bilateral Filtering (OpenCV)
bilateral_cv = cv2.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50)

# (d) Bilateral Filtering (Manual)
bilateral_manual = bilateral_filter_manual(img, diameter=7, sigma_s=3, sigma_r=50)

# ================== DISPLAY RESULTS ==================
# 1. We use a larger figsize for more breathing room
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image", pad=15)
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(gaussian, cmap='gray')
plt.title("Gaussian Filtering)", pad=15)
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(bilateral_cv, cmap='gray')
plt.title("Bilateral Filter (OpenCV)", pad=15)
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(bilateral_manual, cmap='gray')
plt.title("Bilateral Filter (Manual)", pad=15)
plt.axis('off')

# 2. Adjust spacing manually: 
# wspace = width spacing between subplots
# hspace = height spacing between subplots
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.4)

plt.show()