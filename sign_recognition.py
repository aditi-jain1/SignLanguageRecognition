import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import tensorflow as tf

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 50
imgSize = 300
counter = 0

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    #crop image
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]

        imgWhite = np.ones([imgSize, imgSize, 3], np.uint8)*255
        imgCrop = img[y- offset:y+h + offset, x-offset:x+w+offset]

        #imgWhite[0:imgCrop.shape[0], 0:imgCrop.shape[1]] = imgCrop

        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal)/2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)
            print(labels[index])
        
        if aspectRatio <= 1:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal)/2)
            imgWhite[hGap:hGap+ hCal,:] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)
            prediction_list = [str(labels[i]) + str(prediction[i]) for i in range(len(prediction))]
            print(prediction_list)

            print(labels[index])

        cv2.imshow("Cropped Image", imgCrop)
        cv2.imshow("White", imgWhite)


    x, y, c = img.shape
    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)

cap.release()

cv2.destroyAllWindows()