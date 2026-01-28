import cv2
import numpy as np
import matplotlib.pyplot as plt

# ================== LOAD IMAGE ==================
img = cv2.imread(
    r'C:\Users\HP\OneDrive\Documents\GitHub\ET3112_Assignment_01\images\highlights_and_shadows.jpg'
)

if img is None:
    print("Error: Image not found!")
    exit()

# ================== BGR TO LAB ==================
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# Split LAB channels
L, a, b = cv2.split(lab)

# ================== GAMMA CORRECTION ON L ==================
L_norm = L / 255.0
gamma = 0.8   # Gamma value 

L_gamma = np.power(L_norm, gamma)

# Convert back to uint8
L_gamma_uint8 = (L_gamma * 255).astype(np.uint8)

# ================== MERGE & CONVERT BACK ==================
lab_gamma = cv2.merge((L_gamma_uint8, a, b))
img_gamma = cv2.cvtColor(lab_gamma, cv2.COLOR_LAB2BGR)

# Convert to RGB for plotting
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gamma_rgb = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2RGB)

# ================== DISPLAY IMAGES ==================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_gamma_rgb)
plt.title('Gamma Corrected (L Channel)')
plt.axis('off')

plt.show()

# ================== HISTOGRAMS ==================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(L.flatten(), bins=256, range=(0, 255))
plt.title('Histogram of Original L Channel')

plt.subplot(1, 2, 2)
plt.hist(L_gamma_uint8.flatten(), bins=256, range=(0, 255))
plt.title('Histogram of Gamma Corrected L Channel')

plt.show()
