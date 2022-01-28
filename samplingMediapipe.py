import cv2
from time import sleep
import os
import mediapipe as mp

myName = input('name = ')
myName = str(myName)
mypath = os.path.abspath(__file__)
dirName = os.path.dirname(str(mypath))
directories = os.listdir(dirName)

os.mkdir(dirName + '/' + myName)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cam = cv2.VideoCapture(0)
i = 0
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        ret ,img =cam.read()
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        if results.pose_landmarks:
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = image.shape
                myL = results.pose_landmarks.landmark

        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        img = cv2.flip(img,1)
        cv2.imshow("frame", img)


        i = i + 1
        cv2.imwrite(dirName + '/' + myName  + '/' + myName + str(i) + '.jpg', img)
        sleep(0.1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()
