######################################
#
# файл с главным кодом
#
#####################

import cv2 as cv
import numpy as np
from picture_editing import VideoEditor

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
    VideoEditor().find_ctr(mask,out)
    cv.imshow("input", out)


    #cv.imshow("input",frame)

    c=cv.waitKey(10)
    if c==27:
        break
cap.release()
cv.destroyAllWindows()
