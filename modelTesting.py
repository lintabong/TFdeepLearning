import os
import cv2
import time
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Activation, Dropout
from tensorflow.keras.models import load_model

# load the model
model = load_model('model.h5')

CATEGORIES = ['', '']
cap = cv2.VideoCapture(0)
while True:
  _, frame = cap.read()
  img = frame.copy()
  try:
      x, y, w, h = get_detection(frame)
      crop_img = img[y:y+h, x:x+w]
      crop_img = cv2.resize(crop_img, (100, 100))
      crop_img = np.expand_dims(crop_img, axis=0)

prediction = model.predict(crop_img)
index = np.argmax(prediction)
res = CATEGORIES[index]
if index == 0:
    color = (0, 0, 255)
else:
    color = (0, 255, 0)

cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
cv2.putText(frame, res, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, color, 2, cv2.LINE_AA)
except:
pass

cv2.imshow("frame", frame)
if cv2.waitKey(1) == ord('q'):
    break