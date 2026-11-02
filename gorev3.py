import cv2
import numpy as np

resim = cv2.imread("resim.jpg")
gri = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)

kenar = cv2.Canny(gri, 100, 200)

kenar_median = cv2.medianBlur(kenar, 5)

cv2.imwrite("c1_canny.png", kenar)
cv2.imwrite("c2_canny_median.png", kenar_median)

sonuc = resim.copy()
cemberler = cv2.HoughCircles(kenar_median, cv2.HOUGH_GRADIENT,
                             dp=1,         
                             minDist=30,    
                             param1=100,   
                             param2=20,     
                             minRadius=10,
                             maxRadius=60)

if cemberler is not None:
    cemberler = np.round(cemberler[0]).astype(int)
    for (x, y, r) in cemberler:
        cv2.circle(sonuc, (x, y), r, (0, 0, 255), 2)  
        cv2.circle(sonuc, (x, y), 2, (0, 255, 0), 3)   
    print("Bulunan cember sayisi:", len(cemberler))
else:
    print("Cember bulunamadi")

cv2.imwrite("c3_opencv_HoughCircles.png", sonuc)