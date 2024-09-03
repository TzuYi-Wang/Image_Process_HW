import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('Lena.bmp', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('Peppers.bmp', cv2.IMREAD_GRAYSCALE)

# 放原圖 in figure 1
plt.subplot(321), plt.imshow(img1, cmap='gray')
plt.title('Original Image 1'), plt.xticks([]), plt.yticks([])
plt.subplot(322), plt.imshow(img2, cmap='gray')
plt.title('Original Image 2'), plt.xticks([]), plt.yticks([])

# histogram equalization函式
def hist_eq(img):
    # 計算影像histogram
    hist, bins = np.histogram(img.flatten(), 256, [0,256])
    # 計算累積分佈函式
    cdf = hist.cumsum()
    # normalize累積分佈函式
    cdf_normalized = cdf * hist.max() / cdf.max()
    # 計算histogram equalize後的像素值
    img_eq = np.interp(img.flatten(), bins[:-1], cdf_normalized).astype(np.uint8)
    img_eq = img_eq.reshape(img.shape)
    return img_eq

# 對每個圖進行global histogram equalization
img1_eq = hist_eq(img1)
img2_eq = hist_eq(img2)

# 放global histogram equalization的圖 in figure 1
plt.subplot(323), plt.imshow(img1_eq, cmap='gray')
plt.title('Global Histogram Equalization 1'), plt.xticks([]), plt.yticks([])
plt.subplot(324), plt.imshow(img2_eq, cmap='gray')
plt.title('Global Histogram Equalization 2'), plt.xticks([]), plt.yticks([])

# 影像分割函式
def divide_image(img):
    h, w = img.shape
    # 分成16個相等的blocks
    block_h = h // 4
    block_w = w // 4
    blocks = []
    for i in range(4):
        for j in range(4):
            block = img[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            blocks.append(block)
    return blocks

# local histogram equalization函式
def block_hist_eq(img_blocks):
    img_blocks_eq = []
    for block in img_blocks:
        block_eq = hist_eq(block)
        img_blocks_eq.append(block_eq)
    return img_blocks_eq

# 對每個影像進行local histogram equalization
img1_blocks = divide_image(img1)
img2_blocks = divide_image(img2)
img1_blocks_eq = block_hist_eq(img1_blocks)
img2_blocks_eq = block_hist_eq(img2_blocks)

# 將16個blocks還原為原始影像大小
img1_local_eq = np.vstack([np.hstack(img1_blocks_eq[i:i+4]) for i in range(0,16,4)])
img2_local_eq = np.vstack([np.hstack(img2_blocks_eq[i:i+4]) for i in range(0,16,4)])

# 放local histogram equalization的影像 in figure 1
plt.subplot(325), plt.imshow(img1_local_eq, cmap='gray')
plt.title('Local Histogram Equalization 1'), plt.xticks([]), plt.yticks([])
plt.subplot(326), plt.imshow(img2_local_eq, cmap='gray')
plt.title('Local Histogram Equalization 2'), plt.xticks([]), plt.yticks([])

# 放原圖的histogram in figure 2
plt.figure(figsize=(10, 5))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
plt.subplot(221), plt.imshow(img1, cmap='gray')
plt.title('Original Image 1'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.hist(img1.flatten(), 256, [0,256], color='r')
plt.title('Histogram 1'), plt.xlim([0,256])
plt.subplot(223), plt.imshow(img2, cmap='gray')
plt.title('Original Image 2'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.hist(img2.flatten(), 256, [0,256], color='r')
plt.title('Histogram 2'), plt.xlim([0,256])
plt.subplots_adjust(bottom=0.05)

# 放global approach後的圖的histogram in figure 3
plt.figure(figsize=(10, 5))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
plt.subplot(221), plt.imshow(img1_eq, cmap='gray')
plt.title('Global Histogram Equalization 1'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.hist(img1_eq.flatten(), 256, [0,256], color='r')
plt.title('Histogram 1'), plt.xlim([0,256])
plt.subplot(223), plt.imshow(img2_eq, cmap='gray')
plt.title('Global Histogram Equalization 2'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.hist(img2_eq.flatten(), 256, [0,256], color='r')
plt.title('Histogram 2'), plt.xlim([0,256])
plt.subplots_adjust(bottom=0.05)

# 放local approach後的圖的histogram in figure 4
plt.figure(figsize=(10, 5))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
plt.subplot(221), plt.imshow(img1_local_eq, cmap='gray')
plt.title('Local Histogram Equalization 1'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.hist(img1_local_eq.flatten(), 256, [0,256], color='r')
plt.title('Histogram 1'), plt.xlim([0,256])
plt.subplot(223), plt.imshow(img2_local_eq, cmap='gray')
plt.title('Local Histogram Equalization 2'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.hist(img2_local_eq.flatten(), 256, [0,256], color='r')
plt.title('Histogram 2'), plt.xlim([0,256])
plt.subplots_adjust(bottom=0.05)
# 顯示所有輸出結果(所有Figure)
plt.show()
