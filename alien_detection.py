import cv2 as cv
import numpy as np

#read image
img = cv.imread('openCV/sample.jpg')
cv.imshow('Sample', img)

#find no. of led
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 225, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
# print("Number of LEDs = " + str(len(contours)))

if contours==2:
    print("alien_a")
elif contours==3:
     print("alien_b")
elif contours==4:
     print("alien_c")
else:
     print("alien_x")

#Drawing contours
cv.drawContours(img, contours, -1, (0, 255, 0), 2, lineType= cv.LINE_AA)

#Finding area & centroid
led_info = []

for contour in contours:
    area = cv.contourArea(contour)

    M = cv.moments(contour)
    if M["m00"]!= 0:
            cx = float(M["m10"]/M["m00"])
            cy = float(M["m01"]/M["m00"])
            led_info.append({'centroid': (cx,cy), 'area:': area })

for i, led in enumerate(led_info):
    print(f"LED {i+1}: Centroid {led['centroid']}")
    print(f"Contour Area: {area}")

#show no.of led 
x,y,w,h = cv.boundingRect(contour)
cv.putText(img, "+"+ str(len(contours)),(x-10,y-10),cv.FONT_HERSHEY_COMPLEX, .5, (0,255,0), 2)
cv.imshow('Put Text', img)
          
cv.waitKey(0)
cv.destroyAllWindows()