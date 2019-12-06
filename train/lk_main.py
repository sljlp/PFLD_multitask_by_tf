import tensorflow as tf
from lk import lk
import numpy as np
import cv2

def draw(img,pts):
    img2 = img.copy()
    pts = np.array(pts).reshape([-1,2]).astype(int)
    for p in pts:
        cv2.circle(img2,tuple(p),1,(0,255,0),-1)
    return img2

if __name__ =='__main__':

    data = np.zeros((1, 10, 100, 100, 3), dtype=np.float64)
    cap = cv2.VideoCapture('/Users/pengliu/Desktop/rocface/rocface.mp4')
    for i in range(10):
        ret, img = cap.read()
        cv2.imshow('cap', img)
        data[0,i ] = cv2.resize(img,(100,100))
        # if i >= 90:
        #     data[i - 90] = cv2.resize(img, (100, 100))
        cv2.waitKey(50)
    cap.release()

    inputs = tf.placeholder(shape=(1,10,100,100,3),dtype=tf.float64)
    out = lk.lk_forward(None,inputs)
    sess = tf.Session()


    # img = cv2.imread('/Users/pengliu/Desktop/Bazel/cmp/sampleimg/AFW_70037463_1_2_AUG_20.jpg')
    # img = cv2.resize(img,(100,100))
    # data[0] = img
    # for i in range(1,10):
    #     data[i] =



    # print len(data)
    data = data.astype(np.float64) / 256

    predictions, next_batch, fBack_batch, back_batch, boxes = sess.run(out,feed_dict={inputs:data})

    data = data * 256.0
    data = data.astype(np.uint8)

    ranges = [[0,0],[0,1],[1,1],[1,0]]

    com = np.zeros((200, 200, 3), dtype=np.uint8)
    img_dir = '/Users/pengliu/Desktop/rocface/ml'
    imgidex = 1
    for d in zip(predictions[0], next_batch[0], fBack_batch[0], back_batch[0],data[0]):
        # im1 = im.copy()
        # im2 = im.copy()
        # im3 = im.copy()
        # im4 = im.copy()


        for i in range(4):
            start = ranges[i]
            com[start[0]*100:start[0]*100+100,start[1]*100:start[1]*100+100]= draw(d[4],d[i])
        cv2.imshow('com',com)
        cv2.imwrite(img_dir+'/img%04d.jpg' % (imgidex),com)
        imgidex += 1
    print (boxes)
        # cv2.waitKey()

    # im1 = draw(im,ps1)


