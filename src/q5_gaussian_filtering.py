import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ================== PART (a): 5x5 GAUSSIAN KERNEL ==================
sigma = 2
kernel_size = 5
k = kernel_size // 2

x, y = np.mgrid[-k:k+1, -k:k+1]
gaussian_kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

print("5x5 Normalized Gaussian Kernel:\n", gaussian_kernel)

# ================== PART (b): 51x51 GAUSSIAN 3D SURFACE ==================
size_3d = 51
k3 = size_3d // 2

x3, y3 = np.mgrid[-k3:k3+1, -k3:k3+1]
gaussian_3d = np.exp(-(x3**2 + y3**2) / (2 * sigma**2))
gaussian_3d = gaussian_3d / gaussian_3d.sum()

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x3, y3, gaussian_3d, cmap='viridis')
ax.set_title('3D Gaussian Kernel (51x51)')
plt.show()

# ================== LOAD IMAGE ==================
img = cv2.imread(
    r'C:\Users\HP\OneDrive\Documents\GitHub\ET3112_Assignment_01\images\runway.jpg',
    cv2.IMREAD_GRAYSCALE
)

if img is None:
    print("Error: Image not found!")
    exit()

# ================== PART (c): MANUAL GAUSSIAN SMOOTHING ==================
manual_blur = cv2.filter2D(img, -1, gaussian_kernel)

# ================== PART (d): OPENCV GAUSSIAN BLUR ==================
opencv_blur = cv2.GaussianBlur(img, (5, 5), sigma)

# ================== DISPLAY RESULTS ==================
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(manual_blur, cmap='gray')
plt.title('Manual Gaussian Smoothing')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(opencv_blur, cmap='gray')
plt.title('OpenCV GaussianBlur')
plt.axis('off')

plt.show()
