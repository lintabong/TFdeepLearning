import cv2
import numpy
from tensorflow.keras.models import load_model

model = load_model("model.h5")
CATEGORIES = ['berdiri', 'jongkok']
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    img = frame.copy()
    try:
        x, y, w, h = cv2.get_detection(frame)
        crop_img = img[y:y+h, x:x+w]
        crop_img = cv2.resize(crop_img, (100, 100))
        crop_img = numpy.expand_dims(crop_img, axis=0)
        prediction = model.predict(crop_img)
        index = numpy.argmax(prediction)
        res = CATEGORIES[index]
        if index == 0:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, res, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,0.8, color, 2, cv2.LINE_AA)
    except:
        pass

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break

