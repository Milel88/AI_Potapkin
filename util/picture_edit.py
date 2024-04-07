######################################
#
# файл с методом для работы с изображениями
#
#####################
import cv2 as cv
import numpy as np
class VideoEditor:
    @staticmethod
    def frame_edit(frame,scale):
        frame=cv.resize(frame,None ,fx=scale,fy=scale,interpolation=cv.INTER_AREA)
        return frame

    @staticmethod
    def mask_edit(frame,kernel,iteration):
        kernel=np.ones((kernel,kernel),np.uint8)
        #dilation=cv.dilate(frame,kernel,iterations=iteration)
        opening=cv.morphologyEx(frame,cv.MORPH_OPEN,kernel)
        return opening


    @staticmethod
    def find_ctr(frame,res):
        ret,thresh=cv.threshold(frame,170,255,0)
        im2,contours,hierarchy=cv.findContours(
            thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE
        )
        cv.drawContours(res,contours,-1,(0,255,0),3)