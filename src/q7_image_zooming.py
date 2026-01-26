import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def zoom_image(image, s, method='nearest'):
    height, width = image.shape[:2]
    new_height, new_width = int(height * s), int(width * s)
    
    if method == 'nearest':
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    elif method == 'bilinear':
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

def compute_ssd(img1, img2):
    if img1.shape != img2.shape:
        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
    
    img1_norm = img1.astype(np.float32) / 255.0
    img2_norm = img2.astype(np.float32) / 255.0
    
    ssd = np.sum((img1_norm - img2_norm) ** 2)
    return ssd / (img1_norm.shape[0] * img1_norm.shape[1])

# --- Configuration for Grayscale images ---
image_pairs = [
    ('taylor.jpg', 'taylor_small.jpg'),      
    ('taylor.jpg', 'taylor_very_small.jpg'), 
    ('im01.png', 'im01small.png'),
    ('im02.png', 'im02small.png'),
    ('im03.png', 'im03small.png')
]

base_path = r'C:\Users\HP\OneDrive\Documents\GitHub\ET3112_Assignment_01\images'

print(f"{'Image Set':<35} | {'NN SSD':<12} | {'Bilinear SSD':<15}")
print("-" * 75)

for original_name, small_name in image_pairs:
    path_orig = os.path.join(base_path, original_name)
    path_small = os.path.join(base_path, small_name)
    
    # Read as GRAYSCALE
    orig_img = cv2.imread(path_orig, cv2.IMREAD_GRAYSCALE)
    small_img = cv2.imread(path_small, cv2.IMREAD_GRAYSCALE)
    
    if orig_img is None or small_img is None:
        continue

    # Calculate scale factor
    s = orig_img.shape[0] / small_img.shape[0]
    
    zoomed_nn = zoom_image(small_img, s, method='nearest')
    zoomed_bl = zoom_image(small_img, s, method='bilinear')
    
    # Compute SSD for the terminal table
    ssd_nn = compute_ssd(zoomed_nn, orig_img)
    ssd_bl = compute_ssd(zoomed_bl, orig_img)
    
    print(f"{small_name + ' to ' + original_name:<35} | {ssd_nn:<12.6f} | {ssd_bl:<15.6f}")

    # --- Display Results (Clean Titles) ---
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(orig_img, cmap='gray')
    plt.title(f"Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(zoomed_nn, cmap='gray')
    plt.title(f"Nearest-Neighbor Zoom")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(zoomed_bl, cmap='gray')
    plt.title(f"Bilinear Interpolation Zoom")
    plt.axis('off')

    plt.tight_layout()
    plt.show()