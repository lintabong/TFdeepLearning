import cv2
import numpy
from tensorflow.keras.models import load_model

model = load_model("model.h5")
CATEGORIES = ['cat1', 'cat2']
cap = cv2.VideoCapture(0)
while True:
    img = cv2.imread("img0.jpeg")
    try:
        # x, y, w, h = cv2.get_detection(frame)
        # crop_img = img[y:y+h, x:x+w]
        # crop_img = cv2.resize(crop_img, (100, 100))
        # crop_img = numpy.expand_dims(crop_img, axis=0)
        prediction = model.predict(img)
        index = numpy.argmax(prediction)
        res = CATEGORIES[index]
        print(res)
    except:
        pass

    cv2.imshow("frame", img)
    if cv2.waitKey(1) == ord('q'):
        break

