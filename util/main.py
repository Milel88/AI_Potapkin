######################################
#
# файл с главным кодом
#
#####################

import cv2 as cv
import numpy as np
from util.picture_edit import VideoEditor

cap=cv.VideoCapture("Traffic IP Camera video (1).mp4")

bg_subs = cv.createBackgroundSubtractorMOG2()
history = 300
learning_rate = 1.0 / history

while cap.isOpened:
    ret,frame=cap.read()

    if ret!=True:
        break
    frame=VideoEditor.frame_edit(frame,0.9)

    mask=bg_subs.apply(frame,learningRate=learning_rate)
    mask=VideoEditor.mask_edit(mask,5,1)

    out=frame.copy()

    #result=cv.bitwise_and(out,out,mask=mask)
    #result = cv.bitwise_and(out, out, mask=result)

    #cv.imshow("res",mask)
    ret,thresh1=cv.threshold(mask,50,255,cv.THRESH_BINARY)
    filter_im=VideoEditor.filter_im(thresh1,5)
    edited=VideoEditor.mask_edit(filter_im,5,1)
    #cv.imshow("res",thresh1)
    #cv.imshow("res",filter_im)
    #cv.imshow("res", edited)


    VideoEditor().find_ctr(edited,out)
    cv.imshow("input", out)


    #cv.imshow("input",frame)

    c=cv.waitKey(10)
    if c==27:
        break
cap.release()
cv.destroyAllWindows()
