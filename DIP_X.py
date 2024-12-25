"""Canny spliter with illumination robustness."""

import os
import numpy as np
import cv2
from functions import *

cam_path = 0
filename = ''
# 设置低通滤波的内核大小
gauss_brush_size = 7
gauss_brush_low = 0.25
gauss_brush_sigma = np.sqrt(- (gauss_brush_size-1)**2 / 8 / np.log(gauss_brush_low))


def imshow(winname, arr):
    global filename
    cv2.imshow(winname, arr)
    if filename:
        cv2.imwrite('DIP_X-' + os.path.splitext(filename)[0] + '-' + winname + '.png', arr)
    else:
        out_put.write(arr)


def img_iter(cam_path, head):
    global filename, out_put
    if isinstance(cam_path, int) or os.path.isfile(cam_path):
        cam = CAPTUREStruct(cam_path)
        out_put = VIDEOStruct('temp-%s.mp4' % time.strftime("%m-%d-%H-%M"), 'mp4v', cam.fps, cam.size)
        while cv2.waitKey(30) != ord('q'):
            b, c = cam.read()
            if not b:
                break
            cv2.imshow('raw', c)
            yield c

        out_put.close()
        cam.close()
    else:
        for file in os.listdir(cam_path):
            if os.path.splitext(file)[1] not in ('.jpg', '.png', '.webp', '.gif', '.jpeg'):
                continue
            if not file.startswith(head):
                continue
            print(file)
            filename = file
            c = cv2.imread(file)
            cv2.imshow('raw', c)
            yield c
            if cv2.waitKey(800) == ord('q'):
                break

    cv2.destroyAllWindows()



out_put: VIDEOStruct= None
clahe = cv2.createCLAHE(2.0, (int(640/480*6), 6))
orbs = {}
shifts = {}
images = iter(img_iter(cam_path, 'blue-'))

getROICenter = lambda roi: (roi[0]+roi[2]/2, roi[1]+roi[3]/2)
areaROI = lambda roi: roi[2]*roi[3]
def getROICross(*rois, xs=[], ys=[]):
    xs.clear()
    ys.clear()
    for roi in rois:
        xs.append(roi[0])
        xs.append(roi[0]+roi[2])
        ys.append(roi[1])
        ys.append(roi[1]+roi[3])
    xs.sort()
    ys.sort()
    l = len(rois)
    return (xs[l], ys[l], xs[l+1], ys[l+1])

transform_roi = lambda xyxy: (xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1])
transform_xyxy = lambda roi: np.int16((roi[0], roi[1], roi[2] + roi[0], roi[3] + roi[1]))

similarROI = lambda roi1, roi2: areaROI(transform_roi(getROICross(roi1, roi2))) / areaROI()

while cv2.waitKey(1) != ord('q'):
    c = next(images)
    gray = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY)
    gray = clahe.apply(gray, )
    rois = cv2.selectROIs('raw', c, )
    for roi in rois:
        imdst = gray[roi[1]: roi[1]+roi[3], roi[0]: roi[0]+roi[2]]
        # Find in Old objects
        for orb in orbs:
            Mask, good, corners = orb(imdst)
            if Mask is not None:
                break
        else:
            center = getROICenter(roi)
            for shift in shifts:
                if shift.box


    for i in range(10):  # simulate YOLO as lower speed
        imshow('Feature', orb.draw(c, Mask, good, corners))
        mask = cv2.Canny(c, threshold1, threshold2)
        imshow('canny', mask)

        c = cv2.GaussianBlur(c, (gauss_brush_size,gauss_brush_size), gauss_brush_sigma)
        cv2.imshow('raw', c)
        hist = shift(c)

