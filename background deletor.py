import cv2
import numpy as np
file_name = "result.png"

src = cv2.imread(file_name, 1)
tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
_,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
b, g, r = cv2.split(src)
rgba = [b,g,r, alpha]
dst = cv2.merge(rgba,4)
cv2.imwrite("test.png", dst)

edges = cv2.Canny(dst, 100, 50)
cv2.imshow('Canny', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()