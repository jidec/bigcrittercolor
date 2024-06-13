import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('D:/bcc/dfly_appr_expr/appr3/segments/INATRANDOM-4988619_segment.png')
#image = cv2.imread('D:/bcc/dfly_appr_expr/appr3/segments/INATRANDOM-4263361_segment.png')
#image = cv2.imread('D:/bcc/dfly_appr_expr/appr3/segments/INATRANDOM-5618336_segment.png')
image = cv2.imread('D:/bcc/dfly_appr_expr/appr3/segments/INATRANDOM-38377437_segment.png')

# Parameters for bilateral filter
params = [
    (5, 50, 50),
    (9, 75, 75),
    (9, 150, 150),
    (15, 75, 75),
    (15, 150, 150)
]

# Apply bilateral filter with different parameters
filtered_images = [cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace) for d, sigmaColor, sigmaSpace in params]

# Displaying the original and filtered images
images = [image] + filtered_images
titles = ['Original Image'] + [f'Bilateral Filter d={d}, sigmaColor={sc}, sigmaSpace={ss}' for d, sc, ss in params]

# Use matplotlib to display the images
plt.figure(figsize=(15, 12))

for i, (img, title) in enumerate(zip(images, titles)):
    plt.subplot(3, 2, i + 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.show()