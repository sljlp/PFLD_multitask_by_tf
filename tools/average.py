import os
from .. import config
from config import config
import numpy as np

def parseLmkLine(line):
    '''
    :param line:  name -2 106x2 lmk (\n) str
    :return: abs imgpath, shape of 106x2 np.array
    '''
    datas = line.strip().split()
    img_path = datas[0]+'.jpg'
    pts = [float(d) for d in datas[2:]]
    return img_path, np.array(pts).reshape([106,2])


# if __name__ == '__main__':
#     lmklines = open(config.sample_lmk_path,'r').readlines()
#     for line in lmklines:
#         imgpath, pts = parseLmkLine(line)
#
#
#         input('testting ...')
