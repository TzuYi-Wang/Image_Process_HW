import cv2
import numpy as np

# 讀取影像
img1 = cv2.imread("blurry_moon.tif", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("skeleton_orig.bmp", cv2.IMREAD_GRAYSCALE)

# 定義Laplacian operator
# Laplacian operator可以將一個像素的鄰域中的亮度值與該像素自身的亮度值進行差分
# ，以檢測出邊緣和細節的變化。在Laplacian operator中
# ，通常使用以下的3x3卷積核來對像素進行差分計算：
#  0  1  0
#  1 -4  1
#  0  1  0
laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
sharpened_img1 = cv2.filter2D(img1, -1, laplacian_kernel)
sharpened_img2 = cv2.filter2D(img2, -1, laplacian_kernel)

# 定義High-boost filtering
k = 1.5 # k is high-boost filter parameter
# Gaussian blur
blurred_img1 = cv2.GaussianBlur(img1, (3, 3), 0) 
sharpened_img1_2 = cv2.addWeighted(img1, k, blurred_img1, -k+1, 0)
# Gaussian blur
blurred_img2 = cv2.GaussianBlur(img2, (3, 3), 0) 
sharpened_img2_2 = cv2.addWeighted(img2, k, blurred_img2, -k+1, 0)

# 顯示原影像
cv2.imshow("Original Image 1", img1)
cv2.imshow("Original Image 2", img2)

# 顯示處理後的影像
cv2.imshow("Sharpened Image 1 (Laplacian)", sharpened_img1)
cv2.imshow("Sharpened Image 2 (Laplacian)", sharpened_img2)
cv2.imshow("Sharpened Image 1 (High-boost)", sharpened_img1_2)
cv2.imshow("Sharpened Image 2 (High-boost)", sharpened_img2_2)

cv2.waitKey(0)
cv2.destroyAllWindows()
