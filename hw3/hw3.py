import cv2
import numpy as np

def enhance_rgb(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    b, g, r = cv2.split(image)

    r_enhanced = cv2.equalizeHist(r)
    g_enhanced = cv2.equalizeHist(g)
    b_enhanced = cv2.equalizeHist(b)

    enhanced_image = cv2.merge([b_enhanced, g_enhanced, r_enhanced])

    # Convert the enhanced image back to the range of 0-255
    enhanced_image = (enhanced_image * 255).astype(np.uint8)

    return enhanced_image


def enhance_hsi(image):
    # Convert image from BGR to HSI color space
    hsi_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, i = cv2.split(hsi_image)

    i_enhanced = cv2.equalizeHist(i)

    enhanced_hsi_image = cv2.merge([h, s, i_enhanced])

    # Convert the enhanced HSI image back to the BGR color space
    enhanced_bgr_image = cv2.cvtColor(enhanced_hsi_image, cv2.COLOR_HSV2BGR)

    return enhanced_bgr_image


def enhance_lab(image):
    # Convert image from BGR to L*a*b* color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab_image)
    l_enhanced = cv2.equalizeHist(l)
    enhanced_lab_image = cv2.merge([l_enhanced, a, b])
    # Convert the enhanced L*a*b* image back to the BGR color space
    enhanced_bgr_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)

    return enhanced_bgr_image


# 載入四個彩色圖像
image1 = cv2.imread('aloe.jpg')
image2 = cv2.imread('church.jpg')
image3 = cv2.imread('house.jpg')
image4 = cv2.imread('kitchen.jpg')

# 在RGB色彩空間中進行圖像增強
enhanced_rgb_image1 = enhance_rgb(image1)
enhanced_rgb_image2 = enhance_rgb(image2)
enhanced_rgb_image3 = enhance_rgb(image3)
enhanced_rgb_image4 = enhance_rgb(image4)

# 在HSI色彩空間中進行圖像增強
enhanced_hsi_image1 = enhance_hsi(image1)
enhanced_hsi_image2 = enhance_hsi(image2)
enhanced_hsi_image3 = enhance_hsi(image3)
enhanced_hsi_image4 = enhance_hsi(image4)

# 在Lab*色彩空間中進行圖像增強
enhanced_lab_image1 = enhance_lab(image1)
enhanced_lab_image2 = enhance_lab(image2)
enhanced_lab_image3 = enhance_lab(image3)
enhanced_lab_image4 = enhance_lab(image4)

# 顯示或保存增強的圖像
cv2.imshow('Enhanced RGB Image 1', enhanced_rgb_image1)
cv2.imshow('Enhanced RGB Image 2', enhanced_rgb_image2)
cv2.imshow('Enhanced RGB Image 3', enhanced_rgb_image3)
cv2.imshow('Enhanced RGB Image 4', enhanced_rgb_image4)

cv2.imshow('Enhanced HSI Image 1', enhanced_hsi_image1)
cv2.imshow('Enhanced HSI Image 2', enhanced_hsi_image2)
cv2.imshow('Enhanced HSI Image 3', enhanced_hsi_image3)
cv2.imshow('Enhanced HSI Image 4', enhanced_hsi_image4)

cv2.imshow('Enhanced L*a*b* Image 1', enhanced_lab_image1)
cv2.imshow('Enhanced L*a*b* Image 2', enhanced_lab_image2)
cv2.imshow('Enhanced L*a*b* Image 3', enhanced_lab_image3)
cv2.imshow('Enhanced L*a*b* Image 4', enhanced_lab_image4)

cv2.waitKey(0)
cv2.destroyAllWindows()
