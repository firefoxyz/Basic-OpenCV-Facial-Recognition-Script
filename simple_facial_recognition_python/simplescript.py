# Written by firefoxyz

import pathlib
import cv2

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml" # The file used to detect what you want, you can also switch this to data/haarcascade_fullbody.xml to make it a full body tracker.

clf = cv2.CascadeClassifier(str(cascade_path))

camera = cv2.VideoCapture(0) # The ammount of cameras used, 0 = 1, you can increase this number to how many cameras you want to use, or you could replace this with a file to detect a file.

while True: # This is an infinite loop.
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=1, # Defines how many faces it can detect.
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 255), 2) # Draws a rectangle over the tracked face, uses BGR, so the default is yellow, you can change this.

    cv2.imshow("Faces", frame)
    if cv2.waitKey(1) == ord("q"): # Defines the key to press to stop the loop which as default is Q.
        break
    
camera.release()
cv2.destroyAllWindows()