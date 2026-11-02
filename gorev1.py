import cv2
import numpy as np
image = cv2.imread('resim.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, threshold1=100, threshold2=200)

filtered_edges = cv2.medianBlur(edges, ksize=5)

img_hough = image.copy()      
img_hough_p = image.copy()    

#Standart Hough Dönüşümü 

lines = cv2.HoughLines(filtered_edges, rho=1, theta=np.pi/180, threshold=150)

if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        
        cv2.line(img_hough, (x1, y1), (x2, y2), (0, 0, 255), 2)

#Olasılıksal Hough Dönüşümü 

lines_p = cv2.HoughLinesP(filtered_edges, rho=1, theta=np.pi/180, threshold=100, 
                          minLineLength=50, maxLineGap=10)

if lines_p is not None:
    for line in lines_p:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_hough_p, (x1, y1), (x2, y2), (0, 255, 0), 2)

#Sonuçları Ekranda Gösterme
cv2.imshow('1. Kenarlar (Canny + Median)', filtered_edges)
cv2.imshow('2. Standart Hough Lines (Kırmızı)', img_hough)
cv2.imshow('3. Olasılıksal Hough LinesP (Yeşil)', img_hough_p)

cv2.waitKey(0)
cv2.destroyAllWindows()