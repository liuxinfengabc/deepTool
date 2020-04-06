# coding=utf-8
import cv2

img = cv2.imread("D:/DeepTool/DeepTool/deal-data-whole5/train/keyhole/jpg1-L1-249.jpg", 0)
print(img)
cv2.imwrite('C:/pic/1/5.jpg', img)
