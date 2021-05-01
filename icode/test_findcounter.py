import cv2

image = cv2.imread('007-3386-200.png')
counters, _ = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print(counters)

