import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ================== PARAMETERS ==================
sigma = 2
kernel_size = 5
k = kernel_size // 2

# ================== PART (b): DoG KERNELS ==================
x, y = np.mgrid[-k:k+1, -k:k+1]

G = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))

Gx = -(x / sigma**2) * G
Gy = -(y / sigma**2) * G

# Normalize (sum of absolute values = 1)
Gx = Gx / np.sum(np.abs(Gx))
Gy = Gy / np.sum(np.abs(Gy))

print("Derivative of Gaussian Kernel (x-direction):\n", Gx)
print("Derivative of Gaussian Kernel (y-direction):\n", Gy)

# ================== PART (c): 3D VISUALIZATION ==================
size_3d = 51
k3 = size_3d // 2

x3, y3 = np.mgrid[-k3:k3+1, -k3:k3+1]
G3 = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x3**2 + y3**2) / (2 * sigma**2))
Gx3 = -(x3 / sigma**2) * G3

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x3, y3, Gx3, cmap='viridis')
ax.set_title('Derivative of Gaussian (x-direction)')
plt.show()

# ================== LOAD IMAGE ==================
img = cv2.imread(
    r'C:\Users\HP\OneDrive\Documents\GitHub\ET3112_Assignment_01\images\runway.jpg',
    cv2.IMREAD_GRAYSCALE
)

if img is None:
    print("Error: Image not found!")
    exit()

# ================== PART (d): IMAGE GRADIENTS (DoG) ==================
grad_x_dog = cv2.filter2D(img, -1, Gx)
grad_y_dog = cv2.filter2D(img, -1, Gy)

# ================== PART (e): SOBEL GRADIENTS ==================
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

sobel_x = np.uint8(np.absolute(sobel_x) / np.max(np.absolute(sobel_x)) * 255)
sobel_y = np.uint8(np.absolute(sobel_y) / np.max(np.absolute(sobel_y)) * 255)

# ================== DISPLAY RESULTS ==================
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(grad_x_dog, cmap='gray')
plt.title('DoG Gradient X')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(grad_y_dog, cmap='gray')
plt.title('DoG Gradient Y')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(sobel_x, cmap='gray')
plt.title('Sobel Gradient X')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(sobel_y, cmap='gray')
plt.title('Sobel Gradient Y')
plt.axis('off')

plt.show()
