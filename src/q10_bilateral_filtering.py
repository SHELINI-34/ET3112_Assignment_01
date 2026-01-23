import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def bilateral_filter_manual(img, diameter, sigma_s, sigma_r):
    h, w = img.shape
    radius = diameter // 2
    output = np.zeros_like(img, dtype=np.float32)

    for i in range(radius, h - radius):
        for j in range(radius, w - radius):
            wp = 0
            filtered_value = 0

            for x in range(-radius, radius + 1):
                for y in range(-radius, radius + 1):
                    spatial_weight = math.exp(-(x**2 + y**2) / (2 * sigma_s**2))
                    intensity_diff = img[i + x, j + y] - img[i, j]
                    range_weight = math.exp(-(intensity_diff**2) / (2 * sigma_r**2))

                    weight = spatial_weight * range_weight
                    wp += weight
                    filtered_value += weight * img[i + x, j + y]

            output[i, j] = filtered_value / wp

    return np.uint8(output)


img = cv2.imread(
    r'C:\Users\HP\OneDrive\Documents\GitHub\ET3112_Assignment_01\images\taylor.jpg',
    cv2.IMREAD_GRAYSCALE
)

if img is None:
    print("Image not found!")
    exit()

gaussian = cv2.GaussianBlur(img, (7, 7), sigmaX=2)

bilateral_cv = cv2.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50)

bilateral_manual = bilateral_filter_manual(
    img, diameter=7, sigma_s=3, sigma_r=50
)


plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(gaussian, cmap='gray')
plt.title("Gaussian Filtering")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(bilateral_cv, cmap='gray')
plt.title("Bilateral Filter (OpenCV)")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(bilateral_manual, cmap='gray')
plt.title("Bilateral Filter (Manual)")
plt.axis('off')

plt.show()
