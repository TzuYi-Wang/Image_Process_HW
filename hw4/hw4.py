import cv2

# 定義執行Sobel邊緣檢測的函數
def edge_detection(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    # Calculate the magnitude of the gradient
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    # Normalize the magnitude to a range of 0-255
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    return magnitude

image1 = cv2.imread('baboon.png')
image2 = cv2.imread('peppers.png')
image3 = cv2.imread('pool.png')

edges1 = edge_detection(image1)
edges2 = edge_detection(image2)
edges3 = edge_detection(image3)

cv2.imshow('Image 1: baboon.png  Edges', edges1)
cv2.imshow('Image 2: peppers.png Edges', edges2)
cv2.imshow('Image 3: pool.png Edges', edges3)
cv2.waitKey(0)
cv2.destroyAllWindows()
