#importing libraries
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

#capturing live video
cap = cv2.VideoCapture(0)

#detects the hands in the video
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

#folder path to saved images
folder = "DATA/3"

counter = 0

#frame capturing and detecting the hands
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
 #creation of imgwhite and imgcrop if hands are detected
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

   # aspect ratio calculation and resizing the image
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

#displaying the images
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

#saving the images for training
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)