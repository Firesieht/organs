import cv2 as cv
import time 
from threading import Thread 
from recognise import processing
from cam import cam_start

class CamParallel :

    def __init__(self, stream_id=0):
        self.stream_id = stream_id 
        self.cam = cv.VideoCapture(self.stream_id)
        if self.cam.isOpened() is False :
            print("Нет вебкамеры")
            exit(0)
            
        self.grabbed , self.frame = self.cam.read()
        if self.grabbed is False :
            print('ошибка камеры')
            exit(0)

        self.stopped = True 
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True

    def start(self):
        self.stopped = False
        self.t.start() 

    def update(self):
        while True :
            if self.stopped is True :
                break
            self.grabbed , self.frame = self.cam.read()
            if self.grabbed is False :
                print('ошибка камеры')               
                self.stopped = True
                break 
        self.cam.release()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True 

def startSegmentation():
    webcam_stream = CamParallel(stream_id=0) 
    webcam_stream.start()
    cv.namedWindow("frame", cv.WINDOW_AUTOSIZE)


    p2 = Thread(target=cam_start, args=[webcam_stream])

    p2.start()

    num_frames_processed = 0 
    start = time.time()
    while True :
        if webcam_stream.stopped is True :
            break
        else :
            frame = webcam_stream.read() 

        mask, res_frame = processing(frame)

        out = cv.addWeighted(mask, 0.5, res_frame, 0.5, 0)
        

        num_frames_processed += 1 
        cv.imshow('frame' , cv.resize(out, (960,960)))


        key = cv.waitKey(1)
        if key == ord('q'):
            break
    end = time.time()
    webcam_stream.stop() 

    elapsed = end-start
    fps = num_frames_processed/elapsed 
    print("FPS: {} , Время: {} , Кадры: {}".format(fps, elapsed, num_frames_processed))
    cv.destroyAllWindows()


if __name__ == "__main__":
    startSegmentation()
