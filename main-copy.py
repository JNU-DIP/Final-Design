"""Canny spliter with illumination robustness."""

import os
import random

import numpy as np
import cv2
from functions import *
from ultralytics import YOLO

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



class YOLOv8(YOLO):
    def __init__(self, model="./yolov8n.pt", **kwargs):
        super().__init__(model, **kwargs)
        self.objs = []
        self.verbose = kwargs.get('verbose', True)

    @staticmethod
    # 计算检测框的中心点，便于后续显示
    def calculate_center(box):
        shape = box.shape
        box.resize((shape[0], 2, 2))    # do not copy
        center = box.mean(axis=1, dtype=np.uint16)
        box.resize(shape)           # recovery
        return center

    def __call__(self, frame):
        # YOLO 跟踪
        results = self.track(frame, persist=True, tracker="bytetrack.yaml", max_det=8, verbose=self.verbose)[0]
        self.objs.clear()
        # print(results.boxes.id, results.boxes.data[:, 4])
        for detection in results.boxes.data:
            if len(detection) < 7:
                confidence, cls = detection[4:6]
                obj_id = -1
            else:
                # 获取检测信息，含ID
                obj_id, confidence, cls = detection[4:7]
            obj_id = int(obj_id)  # 转换ID为整数
            cls = int(cls)  # integer class
            self.objs.append((obj_id, cls, detection[:4]))
        return self.objs



out_put: VIDEOStruct= None
size = (640, 480)
clahe = cv2.createCLAHE(2.0, (int(size[0]/size[1]*6), 6))
orbs = {}
shifts = {}
images = iter(img_iter(cam_path, 'blue-'))
similar0 = 0.6
pads = (50, 40)
mask = np.zeros(size[::-1], dtype=np.uint8)
yolo = YOLOv8(verbose=False)

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
        hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
        gray = hsv[:,:,2] = clahe.apply(hsv[:,:,2], )
        cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, c)
        # rois = cv2.selectROIs('raw', c, )
        # ids = list(range(len(rois)))
        objs = yolo(c)
        ids = [obj[0] for obj in objs]
        rois = [transform_roi(obj[2]) for obj in objs]

        for Id, roi in zip(ids, rois):
            center = getROICenter(roi)
            # imdst = cutROI(gray, roi)
            mask.fill(0)
            grap_cutROI(c, roi, mask)
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
                        ids[ids.index(Id)] = ID
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
                        if Id == -1:
                            while 1:
                                Id = random.randint(0, 100)
                                if Id not in orbs: break
                        orb = FeatureStruct(gray_roi, method='ORB')
                        orb.pos = center
                        orbs[Id] = orb
            except ValueError:
                pass

            mask_roi = cutROI(mask, roi_pad)
            colors = ShiftStruct.scan_inRange(cutROI(hsv, roi_pad), mask_roi)
            m = ~mask_roi
            c_roi = cutROI(c, roi_pad).copy()
            np.putmask(c_roi[:,:,0], m, 0)
            np.putmask(c_roi[:,:,1], m, 0)
            np.putmask(c_roi[:,:,2], m, 0)
            print(roi, roi_pad, colors)
            cv2.imshow('hsv', c_roi)
            # cv2.waitKey(0)
            if colors[0] is not None and colors[1] is not None:
                for ID, shift in shifts.items():
                    if similarROI(roi, shift.get_roi(size)) > similar0:
                        shift.init_frame(c_roi, (roi[0] - roi_pad[0], roi[1] - roi_pad[1], roi[2], roi[3]), colors)
                        shift.pos = center
                        ids[ids.index(Id)] = ID
                        Id = ID
                        break
                else:
                    if Id in shifts:
                        shifts[Id].init_frame(c_roi, (roi[0] - roi_pad[0], roi[1] - roi_pad[1], roi[2], roi[3]), colors)
                        shifts[Id].pos = center
                    else:
                        if Id == -1:
                            while 1:
                                Id = random.randint(0, 100)
                                if Id not in shifts: break
                        shift = ShiftStruct(c_roi, (roi[0] - roi_pad[0], roi[1] - roi_pad[1], roi[2], roi[3]), colors, mode='mean')
                        shift.pos = center
                        shifts[Id] = shift
                        print(shift.box, 's')
            else:
                pass

        print(tuple(orbs), tuple(shifts))
        for i in range(30):  # simulate YOLO as lower speed
            c = next(images)
            hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
            gray = hsv[:,:,2] = clahe.apply(hsv[:,:,2], )
            cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, c)
            for orb in orbs.values():
                roi_pad = padROI(orb.get_roi(size), pads, size)
                gray_roi = cutROI(gray, roi_pad)
                try:
                    Mask, good, corner = orb(gray_roi)
                    if Mask:
                        orb.pos = ((orb.pos[0]-roi_pad[2]/2, orb.pos[1]-roi_pad[3]/2) + corner[4, 0]).astype(np.int16)
                        # print(roi_pad, orb.pos, 'o')
                        orb.draw(cutROI(c, roi_pad), None, None, corner)
                except cv2.error:
                    pass

            mask.fill(0)
            for shift in shifts.values():
                roi_pad = padROI(shift.get_roi(size), pads, size)
                imshow('shift', shift(cutROI(c, roi_pad)))
                shift.pos = (np.subtract(shift.pos, np.divide(roi_pad[2:], 2)) + shift.box[:2] + np.divide(shift.box[2:], 2)).astype(np.int16)
                # print(roi_pad, shift.pos, 's')
                mask_roi = cutROI(mask, roi_pad)
                mask_roi |= shift.src
                xyxy_hsv = transform_xyxy(shift.get_roi(size))
                cv2.rectangle(c, xyxy_hsv[:2], xyxy_hsv[2:], 255, 2)

            cv2.imshow('det', c)
            cv2.imshow('shift', mask)
            out_put.write(c)

except StopIteration:
    pass
