import cv2 
import time 
from recognise import processing

# opening video capture stream
vcap = cv2.VideoCapture(1)
if vcap.isOpened() is False :
    print("[Exiting]: Error accessing webcam stream.")
    exit(0)
fps_input_stream = int(vcap.get(5))
print("FPS of webcam hardware/input stream: {}".format(fps_input_stream))
grabbed, frame = vcap.read() # reading single frame for initialization/ hardware warm-up

# processing frames in input stream
num_frames_processed = 0 
start = time.time()
while True:
    grabbed, frame = vcap.read()
    if grabbed is False :
        print('[Exiting] No more frames to read')
        break

    cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)
    # adding a delay for simulating time taken for processing a frame 
    mask, res_frame = processing(frame)

    out = cv2.addWeighted(mask, 0.5, res_frame, 0.5, 0)
    num_frames_processed += 1
    cv2.imshow('frame', cv2.resize(out, (960, 960)))

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
end = time.time()

# printing time elapsed and fps 
elapsed = end-start
fps = num_frames_processed/elapsed 
print("FPS: {} , Elapsed Time: {} , Frames Processed: {}".format(fps, elapsed, num_frames_processed))

# releasing input stream , closing all windows 
vcap.release()
cv2.destroyAllWindows()