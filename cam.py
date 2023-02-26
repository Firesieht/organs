import numpy as np
import cv2 as cv


def cam_start(cap):
    if cap.stopped == False:
        while True:
            if cap.stopped == True:
                break
            frame = cap.read()
            cv.imshow("original", frame)

            if cv.waitKey(1) == ord('q'):
                cap.stop()
                break
    cv.destroyAllWindows()