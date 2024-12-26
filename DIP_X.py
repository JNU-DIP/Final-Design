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
size = (640, 480)
clahe = cv2.createCLAHE(2.0, (int(size[0]/size[1]*6), 6))
orbs = {}
shifts = {}
images = iter(img_iter(cam_path, 'blue-'))
similar0 = 0.6
pads = (50, 40)
mask = np.zeros(size[::-1], dtype=np.uint8)

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

similarROI = lambda roi1, roi2: (cross := areaROI(transform_roi(getROICross(roi1, roi2)))) / max(1, areaROI(roi1) + areaROI(roi2) - cross)
inROI = lambda center, roi: roi[0] <= center[0] <= roi[0]+roi[2] and roi[1] <= center[1] <= roi[1]+roi[3]
cutROI = lambda img, roi: img[roi[1]: roi[1]+roi[3], roi[0]: roi[0]+roi[2]]

def grap_cutROI(image, roi, mask=None):
    # 定义背景模型和前景模型
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    # 应用 GrabCut 算法
    m,bg,fg=cv2.grabCut(image, mask, roi, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    return m

try:
    while True:
        c = next(images)
        gray = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY)
        gray = clahe.apply(gray, )
        hsv = cv2.cvtColor(c, cv2.COLOR_RGB2HSV)
        rois = cv2.selectROIs('raw', c, )
        ids = list(range(len(rois)))

        for Id, roi in zip(ids, rois):
            center = getROICenter(roi)
            # imdst = cutROI(gray, roi)
            mask.fill(0)
            grap_cutROI(c, roi, mask)
            mask = ((mask == 1) | (mask == 3)).view(np.uint8)
            np.putmask(mask, mask.view(bool), 255)
            roi_pad = padROI(roi, pads, size)
            gray_roi = cutROI(gray&mask, roi)
            cv2.imshow('gray roi', gray_roi)
            # cv2.waitKey(0)
            # Find in Old objects
            try:
                for ID, orb in orbs.items():
                    # if inROI(center, shift.box):
                    if similarROI(roi, orb.get_roi(size)) > similar0:
                        orb.set_dst(gray_roi)
                        orb.pos = center
                        Id = ID
                        break
                    # Mask, good, corners = orb(imdst)
                    # if Mask is not None:
                    #     break
                else:
                    if Id in orbs:
                        orbs[Id].set_dst(gray_roi)
                        orbs[Id].pos = center
                    else:
                            orb = FeatureStruct(gray_roi, method='ORB')
                            orb.pos = center
                            orbs[Id] = orb
            except ValueError:
                pass

            mask_roi = cutROI(mask, roi_pad)
            hsv_roi = cutROI(hsv, roi_pad)
            colors = ShiftStruct.scan_inRange(hsv_roi, mask_roi)
            m = ~mask_roi
            np.putmask(hsv_roi[:,:,0], m, 0)
            np.putmask(hsv_roi[:,:,1], m, 0)
            np.putmask(hsv_roi[:,:,2], m, 0)
            print(roi, roi_pad, colors)
            cv2.imshow('hsv', cv2.cvtColor(hsv_roi, cv2.COLOR_HSV2RGB))
            # cv2.waitKey(0)
            if colors[0] is not None and colors[1] is not None:
                for ID, shift in shifts.items():
                    if similarROI(roi, shift.get_roi(size)) > similar0:
                        shift.init_frame(hsv_roi, (roi[0] - roi_pad[0], roi[1] - roi_pad[1], roi[2], roi[3]), colors)
                        shift.pos = center
                        Id = ID
                        break
                else:
                    if Id in shifts:
                        shifts[Id].init_frame(hsv_roi, (roi[0] - roi_pad[0], roi[1] - roi_pad[1], roi[2], roi[3]), colors)
                        shifts[Id].pos = center
                    else:
                        shift = ShiftStruct(hsv_roi, (roi[0] - roi_pad[0], roi[1] - roi_pad[1], roi[2], roi[3]), colors, mode='mean')
                        shift.pos = center
                        shifts[Id] = shift
            else:
                pass


        for i in range(30):  # simulate YOLO as lower speed
            c = next(images)
            gray = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY)
            clahe.apply(gray, gray)
            for orb in orbs.values():
                roi_pad = padROI(orb.get_roi(size), pads, size)
                gray_roi = cutROI(gray, roi_pad)
                Mask, good, corner = orb(gray_roi)
                orb.draw(cutROI(c, roi_pad), None, good, corner)

            for shift in shifts.values():
                roi_pad = padROI(shift.get_roi(size), pads, size)
                shift(cutROI(c, roi_pad))
                xyxy_hsv = transform_xyxy(shift.get_roi(size))
                cv2.rectangle(c, xyxy_hsv[:2], xyxy_hsv[2:], 255, 2)

            cv2.imshow('det', c)
            out_put.write(c)

except StopIteration:
    pass
