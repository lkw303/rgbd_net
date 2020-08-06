import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
IMG_SIZE = 384
weights_path = None
if len(sys.argv)>1:
    weights_path = sys.argv[1]
else:
    if weights_path is None:
        weights_path = "./trained_weights/rgb_weights/train_trainable_cropped_s384_1.h5"

model = tf.keras.models.load_model(weights_path)

def segment(img):
    original_size = (np.shape(img)[1],np.shape(img)[0])
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    img = img/255
    prediction = model.predict(np.array([img]))
    pred_img = tf.argmax(prediction, axis =-1)
    pred_img = pred_img[..., tf.newaxis][0]
    pred_img = tf.keras.preprocessing.image.array_to_img(pred_img)
    pred_img = cv2.resize(np.uint8(pred_img), original_size)

    return(pred_img)


#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("videos/driving_1.mp4")

while cap.isOpened():
    _, frame = cap.read()
    cv2.imshow("Frame", frame)
    img_segment = segment(frame)
    cv2.imshow("segment", img_segment)

    
    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()