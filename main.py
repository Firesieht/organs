import onnxruntime
import numpy as np
import cv2 as cv
from time  import time
import subprocess as sp
import multiprocessing as mp
import numba


@numba.njit
def assemblyFrame(ort_outputs):
    colors =[
        [255, 0, 0],
        [255, 255, 0],
        [64, 255, 0],
        [0, 255, 255],
        [0, 64, 255],
        [255, 0, 128],
        [128, 0, 255],
        [128, 128, 128],
        [255, 128, 0],
        [0, 128, 255],
        [255, 255, 255],
        [0, 0, 0],
        [179, 130, 122],
        [222, 222, 222]
    ]   

    # res = np.zeros([256,256,3], dtype=numba.uint8)
    res = [[[0,0,0] for _ in range(256)] for _ in range(256)]

    i_iter = len(ort_outputs[0][0])
    k_iter = len(ort_outputs[0][0,0])
    n_iter = len(ort_outputs[0][0,0][0])

    for k in range(k_iter):
        for n in range(n_iter):
            colorMax = [0.0, 13.0]
            for i in range(i_iter):
                val = ort_outputs[0][0,i][k][n]
                if  val > colorMax[0]:
                    colorMax = [val, i]
            res[k][n] = colors[int(colorMax[1])]

    return np.array(res, dtype=np.uint8)


def getVideoDetails(file):
    cap = cv.VideoCapture(file)
    return int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
def videoMultiprocessing(groupNum):
    cap = cv.VideoCapture(file_name)

    if (cap.isOpened()== False): 
        print("Ошибка чтения файла")
        
    cap.set(cv.CAP_PROP_POS_FRAMES, jump * groupNum)

    fps = int(cap.get(cv.CAP_PROP_FPS))
    proc_frames = 0
    video_cod = cv.VideoWriter_fourcc(*'mp4v')
    video_output = cv.VideoWriter(str(groupNum)+'.mp4', video_cod, fps, (256, 256), isColor=True)

    while proc_frames < jump:
        _, frame = cap.read()
        frame = cv.resize(frame, (256, 256)) 
        frame = np.array([frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]], dtype=np.float32)
        ort_inputs = {ort_session.get_inputs()[0].name: np.array([frame])}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        t1 = time()
        res = assemblyFrame(ort_outputs)
        t2 = time()
        print(t2-t1)
        video_output.write(res)
        cv.imwrite("res1.png", res)
        proc_frames += 1

    cap.release()
    video_output.release()

def multi():
    p = mp.Pool(num_processes)
    p.map(videoMultiprocessing, range(num_processes))


ort_session = onnxruntime.InferenceSession("model.onnx")
num_processes = mp.cpu_count()
file_name = "./d.mov"
width, height, frame_count = getVideoDetails(file_name)
jump =  frame_count//num_processes


if __name__ == '__main__':
    t1 = time()
    multi()
    t2 = time()
    print("Общее время работы файла:", t2-t1)





