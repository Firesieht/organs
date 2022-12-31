from time import time
import numba
from numba import jit
import numpy as np

@jit(nopython=True)
def assemblyFrame():

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

    res = [[[0,0,0] for _ in range(256)] for _ in range(256)]
    # res = np.zeros([256,256,3])
    i_iter = 13
    k_iter = 256
    n_iter = 256

    for k in range(k_iter):
        for n in range(n_iter):
            colorMax = [0, 13]
            for i in range(i_iter):
                val = i
                if  val > colorMax[0]:
                    colorMax = [val, i]
            res[k][n] = colors[colorMax[1]]
    print(res)
    return res


t1 = time()
assemblyFrame()
t2 = time()

print(t2-t1)