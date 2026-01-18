import cv2
import numpy as np
import matplotlib.pyplot as plt


def zoom_nearest(image, scale):
    h, w = image.shape
    new_h = int(h * scale)
    new_w = int(w * scale)

    zoomed = np.zeros((new_h, new_w), dtype=image.dtype)

    for i in range(new_h):
        for j in range(new_w):
            x = int(i / scale)
            y = int(j / scale)
            zoomed[i, j] = image[min(x, h-1), min(y, w-1)]

    return zoomed

def zoom_bilinear(image, scale):
    h, w = image.shape
    new_h = int(h * scale)
    new_w = int(w * scale)

    zoomed = np.zeros((new_h, new_w), dtype=np.float32)

    for i in range(new_h):
        for j in range(new_w):
            x = i / scale
            y = j / scale

            x0 = int(np.floor(x))
            y0 = int(np.floor(y))
            x1 = min(x0 + 1, h - 1)
            y1 = min(y0 + 1, w - 1)

            dx = x - x0
            dy = y - y0

            zoomed[i, j] = (
                (1-dx)*(1-dy)*image[x0, y0] +
                dx*(1-dy)*image[x1, y0] +
                (1-dx)*dy*image[x0, y1] +
                dx*dy*image[x1, y1]
            )

    return zoomed.astype(np.uint8)


def normalized_ssd(img1, img2):
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    return np.sum(diff**2) / np.sum(img1.astype(np.float32)**2)


large = cv2.imread(
    r'C:\Users\HP\OneDrive\Documents\GitHub\ET3112_Assignment_01\images\im01.png',
    cv2.IMREAD_GRAYSCALE
)

small = cv2.imread(
    r'C:\Users\HP\OneDrive\Documents\GitHub\ET3112_Assignment_01\images\im01small.png',
    cv2.IMREAD_GRAYSCALE
)

scale = large.shape[0] / small.shape[0]

zoom_nn = zoom_nearest(small, scale)
zoom_bl = zoom_bilinear(small, scale)

ssd_nn = normalized_ssd(large, zoom_nn)
ssd_bl = normalized_ssd(large, zoom_bl)

print("SSD Nearest Neighbor:", ssd_nn)
print("SSD Bilinear:", ssd_bl)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(large, cmap='gray')
plt.title("Original Large Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(zoom_nn, cmap='gray')
plt.title("Nearest Neighbor Zoom")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(zoom_bl, cmap='gray')
plt.title("Bilinear Zoom")
plt.axis("off")

plt.show()

