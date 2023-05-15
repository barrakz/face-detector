import cv2

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('fs.jpg')

cv2.imshow('Face Detector App', img)
cv2.waitKey()

print("Code completed")