import onnxruntime
import numpy as np
import cv2 as cv
import numba
EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']

ort_session = onnxruntime.InferenceSession("model.onnx", providers=EP_list)

@numba.njit
def assemblyFrame(ort_outputs, frame):
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

            color = colors[int(colorMax[1])]

            res[k][n] = color

    return np.array(res, dtype=np.uint8), frame



    
def processing(frame):
    resized_frame = cv.resize(frame, (256, 256)) 
    frame = np.array([resized_frame[:, :, 0], resized_frame[:, :, 1], resized_frame[:, :, 2]], dtype=np.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: np.array([frame])}
    ort_outputs = ort_session.run(None, ort_inputs)
    res = assemblyFrame(ort_outputs, resized_frame)
    return res










