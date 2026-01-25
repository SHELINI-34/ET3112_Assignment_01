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

# --- Configuration ---
image_pairs = [
    ('im01.png', 'im01small.png'),
    ('im02.png', 'im02small.png'),
    ('im03.png', 'im03small.png')
]

base_path = '../images/'

print(f"{'Image Set':<15} | {'NN SSD':<12} | {'Bilinear SSD':<15}")
print("-" * 50)

for original_name, small_name in image_pairs:
    path_orig = os.path.join(base_path, original_name)
    path_small = os.path.join(base_path, small_name)
    
    orig_img = cv2.imread(path_orig)
    small_img = cv2.imread(path_small)
    
    if orig_img is None or small_img is None:
        continue

    # Scale factor
    s = orig_img.shape[0] / small_img.shape[0]
    
    # Process zooming
    zoomed_nn = zoom_image(small_img, s, method='nearest')
    zoomed_bl = zoom_image(small_img, s, method='bilinear')
    
    # Compute SSD (These will only print to terminal)
    ssd_nn = compute_ssd(zoomed_nn, orig_img)
    ssd_bl = compute_ssd(zoomed_bl, orig_img)
    
    # Print numerical results to Terminal
    print(f"{original_name:<15} | {ssd_nn:<12.6f} | {ssd_bl:<15.6f}")

    # --- Display for Manual Saving ---
    plt.figure(figsize=(18, 6))
    
    # Title is now clean (no numbers)
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Original ({original_name})")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(zoomed_nn, cv2.COLOR_BGR2RGB))
    plt.title("Nearest-Neighbor")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(zoomed_bl, cv2.COLOR_BGR2RGB))
    plt.title("Bilinear")
    plt.axis('off')

    plt.show() 

print("\nAll images processed. Numerical SSD values are listed in the terminal above.")
