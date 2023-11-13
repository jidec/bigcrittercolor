import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('D:/bcc/dfly_appr_expr/appr3/segments/INATRANDOM-4988619_segment.png')
image = cv2.imread('D:/bcc/dfly_appr_expr/appr3/segments/INATRANDOM-4263361_segment.png')
#image = cv2.imread('D:/bcc/dfly_appr_expr/appr3/segments/INATRANDOM-5618336_segment.png')
#image = cv2.imread('D:/bcc/dfly_appr_expr/appr3/segments/INATRANDOM-38377437_segment.png')

# Gaussian Blur
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

# Median Blur
median_blur = cv2.medianBlur(image, 5)

# Bilateral Filter
bilateral_filter = cv2.bilateralFilter(image, 9, 75, 75)

# Displaying the original and blurred images
images = [image, gaussian_blur, median_blur, bilateral_filter]
titles = ['Original Image', 'Gaussian Blur', 'Median Blur', 'Bilateral Filter']

# Use matplotlib to display the images
plt.figure(figsize=(12, 8))

for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()