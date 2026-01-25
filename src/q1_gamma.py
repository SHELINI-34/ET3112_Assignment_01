import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- PART A: LOAD AND PRE-PROCESS ---
# Make sure your image path is correct relative to your script location
image_path = '../images/runway.jpg'

if not os.path.exists(image_path):
    print(f"Error: File not found at {image_path}")
    exit()

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img_norm = img / 255.0  # Normalize to [0,1] for mathematical operations

# --- PART B: GAMMA TRANSFORMATIONS ---
def apply_gamma(image, gamma):
    # Formula: s = c * r^gamma (c=1 here)
    return np.power(image, gamma)

gamma_05 = apply_gamma(img_norm, 0.5) # Brightens shadows
gamma_20 = apply_gamma(img_norm, 2.0) # Darkens/increases contrast

# --- PART C: PIECEWISE CONTRAST STRETCHING ---
def contrast_stretch(image, r1, r2):
    # This creates a more aggressive stretch (Thresholding logic)
    stretched = np.zeros_like(image)
    
    # 1. Darken values below r1 to black
    stretched[image < r1] = 0
    
    # 2. Linearly stretch values between r1 and r2 to [0, 1]
    mask = (image >= r1) & (image <= r2)
    stretched[mask] = (image[mask] - r1) / (r2 - r1)
    
    # 3. Brighten values above r2 to white
    stretched[image > r2] = 1
    
    return stretched

contrast_res = contrast_stretch(img_norm, 0.2, 0.8)

# --- FINAL OUTPUT: CONSOLIDATED DISPLAY ---
plt.figure(figsize=(12, 8))

# Subplot 1: Original
plt.subplot(2, 2, 1)
plt.imshow(img_norm, cmap='gray')
plt.title('(a) Original Grayscale')
plt.axis('off')

# Subplot 2: Gamma 0.5
plt.subplot(2, 2, 2)
plt.imshow(gamma_05, cmap='gray')
plt.title('(b) Gamma Correction ($\gamma=0.5$)')
plt.axis('off')

# Subplot 3: Gamma 2.0
plt.subplot(2, 2, 3)
plt.imshow(gamma_20, cmap='gray')
plt.title('(c) Gamma Correction ($\gamma=2.0$)')
plt.axis('off')

# Subplot 4: Contrast Stretching
plt.subplot(2, 2, 4)
plt.imshow(contrast_res, cmap='gray')
plt.title('(d) Contrast Stretching ($r_1=0.2, r_2=0.8$)')
plt.axis('off')

plt.tight_layout()
plt.show()