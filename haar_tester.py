import cv2
from math import sin, cos, radians

# camera =  cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier('cascade.xml')

settings = {
    'scaleFactor': 1.1, 
    'minNeighbors': 5, 
    'minSize': (20, 20)
}

def rotate_image(image, angle):
    if angle == 0: return image
    height, width = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 0.9)
    result = cv2.warpAffine(image, rot_mat, (width, height), flags=cv2.INTER_LINEAR)
    return result

def rotate_point(pos, img, angle):
    if angle == 0: return pos
    x = pos[0] - img.shape[1]*0.4
    y = pos[1] - img.shape[0]*0.4
    newx = x*cos(radians(angle)) + y*sin(radians(angle)) + img.shape[1]*0.4
    newy = -x*sin(radians(angle)) + y*cos(radians(angle)) + img.shape[0]*0.4
    return int(newx), int(newy), pos[2], pos[3]

img = cv2.imread('sample.jpg')

for angle in [0, -25, 25]:
    rimg = rotate_image(img, angle)
    detected = cascade.detectMultiScale(rimg, **settings)
    if len(detected):
        detected = [rotate_point(detected[-1], img, -angle)]
        break

# Make a copy as we don't want to draw on the original image:
for x, y, w, h in detected[-1:]:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)

cv2.imshow('facedetect', img)


cv2.waitKey(0)
cv2.destroyAllWindows()