import cv2
import matplotlib.pyplot as plt

# ================== LOAD IMAGE ==================
img = cv2.imread(
    r'C:\Users\HP\OneDrive\Documents\GitHub\ET3112_Assignment_01\images\taylor.jpg',
    cv2.IMREAD_GRAYSCALE
)

if img is None:
    print("Image not found!")
    exit()

# ================== LAPLACIAN SHARPENING ==================
laplacian = cv2.Laplacian(img, cv2.CV_64F)
sharpened = img - laplacian
sharpened = cv2.convertScaleAbs(sharpened)

# ================== DISPLAY RESULTS ==================
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(sharpened, cmap='gray')
plt.title("Sharpened Image")
plt.axis('off')

plt.show()
