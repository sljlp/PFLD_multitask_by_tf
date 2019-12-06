import cv2
import numpy as np
def getBBox(pts):
    '''
    :param pts: shape of 106x2 np.array
    :return: shape of (4,) np.array
    '''
    pts = pts.reshape([106,2])
    bbox = np.zeros((4,),dtype=float)
    bbox[:2] = np.min(pts,axis=0)
    bbox[2:] = np.max(pts,axis=0)
    return bbox

def drawRect(img,bbox,color=(0,0,255)):
    '''
    :param img: shape of (w,h,c) cv2.mat
    :param bbox: shape of (4,) np.array
    :return:
    '''

    bbox = bbox.reshape([4]).astype(int)
    img2 = img.copy()
    cv2.rectangle(img2,(bbox[0],bbox[1]),(bbox[2],bbox[3]),color,2)
    return img2



def drawPoints(img,pts,color=(0,255,255)):
    '''
    :param img: shape of (w,h,c) cv2.mat
    :param pts: shape of (X*2,)/(X,2) np.array(dtype=int)
    :param color: tuple (0,255,255)
    :return: img2 drew with pts
    '''
    # print (type(img))
    pts = np.asarray(pts).reshape([-1,2]).astype(int)
    # print pts
    img2 = img.copy()
    if type(color) == type([]) or type(color) == type(np.array([])):
        index=0
        for p in pts:
            # print p[1]
            c = color[index]
            cv2.circle(img2, (p[0],p[1]), 2, c, 1, -1)
            index += 1
    else:
        for p in pts:
            cv2.circle(img2, (p[0], p[1]), 2, color, 1, -1)

    return img2

def expandBox(box,factor):
    center = (box[0:2] + box[2:4])/2
    tedBox = box.copy()
    tedBox[0:2] = (box[0:2] - center) * (1+factor) + center
    tedBox[2:4] = (box[2:4] - center) * (1+factor) + center
    return tedBox

