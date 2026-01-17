import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read image in grayscale
img = cv2.imread('../images/runway.jpg', cv2.IMREAD_GRAYSCALE)

# Check if image loaded
if img is None:
    print("Error: Image not found!")
    exit()

# Step 2: Normalize image to range [0,1]
img_norm = img / 255.0

# Step 3: Gamma correction function
def gamma_correction(image, gamma):
    return np.power(image, gamma)

# Step 4: Apply gamma corrections
gamma_05 = gamma_correction(img_norm, 0.5)
gamma_2 = gamma_correction(img_norm, 2.0)

# Step 5: Display results
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(img_norm, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gamma_05, cmap='gray')
plt.title('Gamma = 0.5')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(gamma_2, cmap='gray')
plt.title('Gamma = 2')
plt.axis('off')

plt.show()

