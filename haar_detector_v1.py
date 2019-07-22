import numpy as np
import cv2


cascade = cv2.CascadeClassifier('cascade.xml')


cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cracks = cascade.detectMultiScale(gray, 1.1, 5)

    for (x,y,w,h) in cracks:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
       
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()