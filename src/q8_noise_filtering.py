import cv2
import matplotlib.pyplot as plt

# ================== LOAD IMAGE ==================
img = cv2.imread(
    r'C:\Users\HP\OneDrive\Documents\GitHub\ET3112_Assignment_01\images\emma_salt_pepper.jpg',
    cv2.IMREAD_GRAYSCALE
)

if img is None:
    print("Image not found!")
    exit()

# ================== GAUSSIAN FILTER ==================
gaussian = cv2.GaussianBlur(img, (5, 5), sigmaX=1)

# ================== MEDIAN FILTER ==================
median = cv2.medianBlur(img, 5)

# ================== DISPLAY RESULTS ==================
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Noisy Image")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(gaussian, cmap='gray')
plt.title("Gaussian Filtering")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(median, cmap='gray')
plt.title("Median Filtering")
plt.axis('off')

plt.show()
