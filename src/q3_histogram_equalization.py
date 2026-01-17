import cv2
import numpy as np
import matplotlib.pyplot as plt

# ================== LOAD IMAGE ==================
img = cv2.imread(
    r'C:\Users\HP\OneDrive\Documents\GitHub\ET3112_Assignment_01\images\runway.jpg',
    cv2.IMREAD_GRAYSCALE
)

if img is None:
    print("Error: Image not found!")
    exit()

# ================== HISTOGRAM EQUALIZATION FUNCTION ==================
def histogram_equalization(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]

    equalized = np.floor(255 * cdf_normalized[image]).astype(np.uint8)
    return equalized

# Apply histogram equalization
img_eq = histogram_equalization(img)

# ================== DISPLAY IMAGES ==================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_eq, cmap='gray')
plt.title('Histogram Equalized Image')
plt.axis('off')

plt.show()

# ================== HISTOGRAMS ==================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(img.flatten(), bins=256, range=(0, 255))
plt.title('Original Histogram')

plt.subplot(1, 2, 2)
plt.hist(img_eq.flatten(), bins=256, range=(0, 255))
plt.title('Equalized Histogram')

plt.show()
