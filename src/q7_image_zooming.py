import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Nearest Neighbor Interpolation
# -----------------------------
def zoom_nearest(img, scale):
    h, w = img.shape[:2]
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    zoomed = np.zeros((new_h, new_w, 3), dtype=img.dtype)

    for i in range(new_h):
        for j in range(new_w):
            x = min(int(i / scale), h - 1)
            y = min(int(j / scale), w - 1)
            zoomed[i, j] = img[x, y]

    return zoomed


# -----------------------------
# Bilinear Interpolation
# -----------------------------
def zoom_bilinear(img, scale):
    h, w = img.shape[:2]
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    zoomed = np.zeros((new_h, new_w, 3), dtype=np.float32)

    for i in range(new_h):
        for j in range(new_w):
            x = i / scale
            y = j / scale

            x0 = int(np.floor(x))
            x1 = min(x0 + 1, h - 1)
            y0 = int(np.floor(y))
            y1 = min(y0 + 1, w - 1)

            dx = x - x0
            dy = y - y0

            top = (1 - dy) * img[x0, y0] + dy * img[x0, y1]
            bottom = (1 - dy) * img[x1, y0] + dy * img[x1, y1]
            zoomed[i, j] = (1 - dx) * top + dx * bottom

    return np.uint8(zoomed)


# -----------------------------
# Normalized SSD
# -----------------------------
def compute_ssd(img1, img2):
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    return np.sum(diff ** 2) / img1.size


# -----------------------------
# IMAGE PAIRS
# -----------------------------
image_pairs = [
    ("im01.png", "im01small.png"),
    ("im02.png", "im02small.png"),
    ("im03.png", "im03small.png"),
]

# -----------------------------
# MAIN PROCESS
# -----------------------------
for idx, (orig_name, small_name) in enumerate(image_pairs, start=1):

    original = cv2.imread(f"../images/{orig_name}")
    small = cv2.imread(f"../images/{small_name}")

    if original is None or small is None:
        print(f"Error loading {orig_name} or {small_name}")
        continue

    # Compute scale factor
    scale = original.shape[0] / small.shape[0]

    # Zoom using both methods
    nearest = zoom_nearest(small, scale)
    bilinear = zoom_bilinear(small, scale)

    # Resize to EXACT original size (fixes SSD error)
    nearest_resized = cv2.resize(nearest, (original.shape[1], original.shape[0]))
    bilinear_resized = cv2.resize(bilinear, (original.shape[1], original.shape[0]))

    # Compute SSD
    ssd_nearest = compute_ssd(original, nearest_resized)
    ssd_bilinear = compute_ssd(original, bilinear_resized)

    print(f"\nResults for {orig_name}")
    print("SSD Nearest Neighbor:", ssd_nearest)
    print("SSD Bilinear:", ssd_bilinear)

    # -----------------------------
    # DISPLAY RESULTS
    # -----------------------------
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(nearest_resized, cv2.COLOR_BGR2RGB))
    plt.title("Nearest Neighbor")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(bilinear_resized, cv2.COLOR_BGR2RGB))
    plt.title("Bilinear")
    plt.axis('off')

    plt.suptitle(f"Q7 Image Zooming – Image Set {idx}")
    plt.show()
