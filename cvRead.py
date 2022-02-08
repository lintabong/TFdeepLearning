import cv2

cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.flip(img, 1)
    cv2.imshow("frame", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
