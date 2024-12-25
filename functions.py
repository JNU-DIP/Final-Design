import numpy as np
import cv2
import time


class CAPTUREStruct:
    def __init__(self, path: str or int or tuple or cv2.VideoCapture,
                 size: tuple = None, fps: int = None, fourcc: str = None):
        self.video = cv2.VideoCapture(path) if isinstance(path, (str, int)) else \
            cv2.VideoCapture(*path) if isinstance(path, (tuple, list)) else path
        assert self.video.isOpened()
        if size is not None:
            self.size = size
        if fps:
            self.fps = fps
        if fourcc:
            self.fourcc = fourcc

    @property
    def size(self):
        assert hasattr(self.video, 'get')
        return int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @size.setter
    def size(self, _size):
        assert hasattr(self.video, 'set')
        self.video.set(3, _size[0])
        self.video.set(4, _size[1])

    @property
    def fps(self):
        assert hasattr(self.video, 'get')
        return int(self.video.get(cv2.CAP_PROP_FPS))

    @fps.setter
    def fps(self, _fps):
        assert hasattr(self.video, 'set')
        self.video.set(5, _fps)

    @property
    def fourcc(self):
        assert hasattr(self.video, 'get')
        return int(self.video.get(cv2.CAP_PROP_FOURCC))

    @fourcc.setter
    def fourcc(self, _fourcc):
        assert hasattr(self.video, 'set')
        self.video.set(6, _fourcc)

    @property
    def expos(self):
        import cv2
        assert hasattr(self.video, 'get')
        return int(self.video.get(cv2.CAP_PROP_EXPOSURE))

    @expos.setter
    def expos(self, _exposure):
        assert hasattr(self.video, 'get')
        self.video.set(cv2.CAP_PROP_EXPOSURE, _exposure)

    @property
    def bright(self):
        assert hasattr(self.video, 'get')
        return int(self.video.get(cv2.CAP_PROP_BRIGHTNESS))

    @bright.setter
    def bright(self, _brightness):
        assert hasattr(self.video, 'get')
        self.video.set(cv2.CAP_PROP_BRIGHTNESS, _brightness)

    def read(self):
        try:
            assert self.video.isOpened()
            return self.video.read()
        except:
            self.close()
            raise

    def __enter__(self): return self

    def __exit__(self, tp, val, tb):
        self.close()

    def close(self):
        cv2.destroyAllWindows()
        if self.video is not None:
            self.video.release()
            self.video = None


class VIDEOStruct:
    FOURCC = (('I420', 'YUV'), ('PIM1', 'MPEG-1'), ('XVID', 'MPEG-4'), ('THEO', 'Ogg Vorbis'), ('FLV1', 'Flash'),
              ('AVC1', 'H264'), ('DIV3', 'MPEG-4.3'), ('DIVX', 'MPEG-4'), ('MP42', 'MPEG-4.2'), ('MJPG', 'motion-jpeg'),
              ('U263', 'H263'), ('I263', 'H263I'), ('MP4V', 'mp4'))

    def __init__(self, path, fourcc: str, fps: int, size: tuple):
        assert fourcc in map(lambda x: x[0].lower(), self.FOURCC)
        assert path.isascii()
        self.video = cv2.VideoWriter()
        self.video.open(path, (cv2.VideoWriter.fourcc
                               if not hasattr(cv2, 'VideoWriter_fourcc') else cv2.VideoWriter_fourcc)(*fourcc),
                        fps, size, True)
        # self.fourcc, self.size, self.fps = fourcc, size, fps

    @property
    def size(self):
        assert hasattr(self.video, 'get')
        return int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self):
        assert hasattr(self.video, 'get')
        return int(self.video.get(cv2.CAP_PROP_FPS))

    @property
    def fourcc(self):
        assert hasattr(self.video, 'get')
        return int(self.video.get(cv2.CAP_PROP_FOURCC))

    def write(self, frame):
        try:
            assert self.video.isOpened()
            self.video.write(frame)
        except:
            self.close()
            raise

    def close(self):
        if self.video:
            self.video.release()
            self.video = None

    def __enter__(self): return self

    def __exit__(self, tp, val, tb):
        self.close()


class ShiftStruct:
    def __init__(self, init_frame=None, box=None, colors=None, pad=100, counts=10, mode='Cam'):
        # set up the termination criteria, either 10 iteration or move by at least 1 pt
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, counts, 1)
        self.model = cv2.CamShift if mode == 'Cam' else cv2.meanShift
        self.colors = colors
        self.box = box
        self.padding = pad
        self.src = init_frame
        self.roi_hist = None
        if init_frame is not None:
            self.init_frame(init_frame)

    def init_frame(self, frame, box=None, colors=None):
        if box is None:
            box = self.box
        if colors is None:
            colors = self.colors
        if box is None or colors is None:
            raise ValueError("Must give a condition.")
        # set up the ROI for tracking
        roi = frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, colors[0], colors[1])
        self.roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)
        self.colors = colors
        if self.src is not None and self.src.shape == frame.shape[:2]:
            self.src.fill(0)
        else:
            self.src = np.zeros_like(frame, shape=frame.shape[:2])
        self.box = box
        return mask

    def __call__(self, frame, box=None, colors=None):
        if box is not None:
            self.init_frame(frame, box, colors)
        if colors is None:
            colors = self.colors
        timer = cv2.getTickCount()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1, self.src)
        np.putmask(self.src, ~cv2.inRange(hsv, colors[0], colors[1]), 0)
        ret, box = self.model(self.src, self.box, self.term_crit)
        self.box = box
        self.fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        return ret

    @staticmethod
    def scan_inRange(hsv, mask, rate=1.2, dead_band=(0., 0., 0.)):
        size, _size = np.histogram(mask, bins=2)[0]
        _mask = ~mask
        rate = 1 / rate
        hsvL = np.array([0, 0, 0], dtype=np.uint8)
        hsvH = np.array([180, 255, 255], dtype=np.uint8)
        hist_i = cv2.calcHist([hsv], [0], mask, [180], [0, 180])
        hist_i *= 1 / size
        hist_i = np.diff(hist_i, prepend=0)
        hist_b = cv2.calcHist([hsv], [0], _mask, [180], [0, 180])
        hist_b *= 1 / _size
        hist_b = np.diff(hist_b, prepend=0)
        Lic = Hic = hist_i.argmax()
        hist_r = hist_i * rate
        hist_r += dead_band[0]
        in_range = hist_i >= hist_b
        if not in_range[Hic]:
            return None, None
        while in_range[Hic]:
            if Hic == 179: break
            Hic += 1
        while in_range[Lic]:
            if Lic == 0: break
            Lic -= 1
        hsvH[0] = Hic
        hsvL[0] = Lic

        hist_i = cv2.calcHist([hsv], [1], mask, [256], [0, 256])
        hist_i *= 1 / size
        hist_i = np.diff(hist_i, prepend=0)
        hist_b = cv2.calcHist([hsv], [1], _mask, [256], [0, 256])
        hist_b *= 1 / _size
        hist_b = np.diff(hist_b, prepend=0)
        Lic = Hic = hist_i.argmax()
        hist_r = hist_i * rate
        hist_r += dead_band[1]
        in_range = hist_i >= hist_b
        if not in_range[Hic]:
            return None, None
        while in_range[Hic]:
            if Hic == 255: break
            Hic += 1
        while in_range[Lic]:
            if Lic <= 10: break
            Lic -= 1
        hsvH[1] = Hic
        hsvL[1] = Lic

        hist_i = cv2.calcHist([hsv], [2], mask, [256], [0, 255], hist=hist_i)
        hist_i *= 1 / size
        hist_i = np.diff(hist_i, prepend=0)
        hist_b = cv2.calcHist([hsv], [2], _mask, [256], [0, 255], hist=hist_b)
        hist_b *= 1 / _size
        hist_b = np.diff(hist_b, prepend=0)
        Lic = Hic = hist_i.argmax()
        np.multiply(hist_i, rate, out=hist_r)
        hist_r += dead_band[2]
        np.greater_equal(hist_i, hist_b, out=in_range)
        if not in_range[Hic]:
            return None, None
        while in_range[Hic]:
            if Hic == 255: break
            Hic += 1
        while in_range[Lic]:
            if Lic <= 20: break
            Lic -= 1
        hsvH[2] = Hic
        hsvL[2] = Lic
        return hsvL, hsvH

    def draw(self, frame):
        ret = self(frame)
        # Draw it on image
        if self.model is cv2.meanShift:
            frame = cv2.rectangle(frame, self.box[:2], (self.box[0]+self.box[2], self.box[1]+self.box[3]), 255, 2)
            cv2.putText(frame, 'meanShift', (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        else:
            pts = np.int0(cv2.boxPoints(ret))
            frame = cv2.polylines(frame, [pts], True, 255, 2)
            cv2.putText(frame, 'CamShift', (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.putText(frame, "FPS : " + str(int(self.fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.imshow('Shift', frame)


def find_corners(img, nums=25, min_confidence=0.1, distance=10):
    # https://docs.opencv.org/4.7.0/d4/d8c/tutorial_py_shi_tomasi.html
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert 0 <= min_confidence < 1
    corner = np.int0(cv2.goodFeaturesToTrack(img, nums, min_confidence, distance)).ravel()
    return corner

def triangle_area(p1, p2, p3):
    return 0.5 * abs((p2[0]-p1[0]) * (p3[1]-p1[1]) - (p2[1]-p1[1]) * (p3[0]-p1[0]))


class FeatureStruct:
    def __init__(self, dst_image, min_mum=10, low_rate=0.7, max_error=5.0,
                 confidence=0.98, search_checks=50, area_rate=0.1, method='SIFT'):
        # Base on cv2.findHomography; cv2.ORB_create
        # https://docs.opencv.org/4.7.0/d1/de0/tutorial_py_feature_homography.html
        # 使用ORB算法检测关键点和描述符
        if method == 'ORB':
            self.orb = cv2.ORB.create() if hasattr(cv2, 'ORB') else cv2.ORB_create()  # 图像特征提取
        # 创建匹配器并匹配两幅图像的描述符
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=12,  # 16
                                key_size=20,  # 12
                                multi_probe_level=2)  # 1
        else:
            self.orb = cv2.SIFT.create() if hasattr(cv2, 'STFT') else cv2.STFT_create()  # 有专利限制
            index_params = dict(algorithm=1, trees=5)

        search_params = dict(checks=search_checks)
        # https://docs.opencv.org/4.7.0/dc/dc3/tutorial_py_matcher.html
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        # self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.dst_image = dst_image
        if dst_image.ndim == 3:
            dst_image = cv2.cvtColor(dst_image, cv2.COLOR_BGR2GRAY)
        self.kp1, self.desc1 = self.orb.detectAndCompute(dst_image, None)
        if self.desc1 is None:
            raise ValueError('Can not detect dst image.')
        self.min_mum = min_mum
        self.low_rate = low_rate
        self.max_error = max_error
        self.confidence = confidence
        h, w = dst_image.shape  # src
        self.corner = np.float32([
            [0, 0],
            [0, h-1],
            [w-1, h-1],
            [w-1, 0],
            [w/2, h/2]
        ]).reshape(-1, 1, 2)
        self.matrix = np.eye(3, dtype=np.float32)
        self.area_rate = area_rate
        self.method = method

    def __call__(self, image):
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 找到两幅图像的特征
        self.kp2, self.desc2 = self.orb.detectAndCompute(image, None)
        if self.desc2 is None:
            # print('Can not detect.')
            return None, None, None
        # 特征匹配
        matches = self.matcher.knnMatch(self.desc1, self.desc2, k=2)
        # 仅使用足够好的匹配点
        good = []
        if self.method == 'ORB':
            for ms in matches:
                if not ms: continue     # ORB don't hand in one or more values in some case, special treatment.
                elif len(ms) == 1:
                    good.append(ms[0])
                elif len(ms) == 2 and ms[0].distance < self.low_rate*ms[1].distance:
                    good.append(ms[0])
        else:   # STFT
            for m, n in matches:
                if m.distance < self.low_rate*n.distance:
                    good.append(m)

        Mask = corners = None
        if len(good) > self.min_mum:
            src_pts = np.float32([self.kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([self.kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            # 计算变换矩阵
            # 匹配时可能存在一些错误，这可能会影响结果。为了解决这个问题，算法使用RANSAC或LEAST_MEDIAN（可以通过标志来决定）。
            # 因此，提供正确估计的良好匹配称为异常值，其余的匹配称为异常值。cv.findHomography（） 返回一个掩码，该掩码指定了内值点和异常点。
            self.matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.max_error,
                            maxIters=500, confidence=self.confidence)
            # https://docs.opencv.org/4.7.0/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780
            # H[2,2]=1
            # x_i" = (H[0,0] *x_i + H[0,1] *y_i + H[0,2]) / (H[2,0] *x_i + H[2,1] *y_i + H[2,2])
            # y_i" = (H[1,0] *x_i + H[1,1] *y_i + H[1,2]) / (H[2,0] *x_i + H[2,1] *y_i + H[2,2])
            # error = \sum{(x_i' - x_i")^2 + (y_i' - y_i")^2}
            # {0,0} -> {H[0,2]/H[2,2], H[1,2]/H[2,2]} -> {x0, y0}
            corners = cv2.perspectiveTransform(self.corner, self.matrix)
            nc = corners[:, 0, :]
            h, w = image.shape[:2]
            # print(nc[:4, :2].astype(np.int16).tolist())
            if nc[4, 0] <= 0 or nc[4, 1] <= 0:
                pass
            elif nc[4, 0] > w or nc[4, 1] > h:
                pass
            elif (nc[:4, 0] < -w/2).any() or (nc[:4, 1] < -h/2).any():
                pass
            elif (nc[:4, 0] > w+w/2).any() or (nc[:4, 1] > h+h/2).any():
                pass
            elif triangle_area(nc[0], nc[1], nc[2])+triangle_area(nc[0], nc[2], nc[3]) <= h*w*self.area_rate:
                pass
            else:
                Mask = mask.ravel().tolist()
        return Mask, good, corners

    def draw(self, image, Mask, good, corners):
        if Mask:
            # 应用透视变换
            nc = corners[:, 0, :]
            h, w = image.shape[:2]
            for i, p in enumerate(nc):
                cv2.circle(image, (int(p[0]), int(p[1])), 5+2*i, (255,255,0), 3)
            print(f'w:{nc[4,0]/w:.2f}',
                  f'h:{nc[4,1]/h:.2f}',
                  f'area: {triangle_area(nc[0], nc[1], nc[2]):.0f}+{triangle_area(nc[0], nc[2], nc[3]):.0f}')
            # Draw a rect when matched out it
            image = cv2.polylines(image, [np.int32(corners[:4])], True, 255, 3, cv2.LINE_AA)
        else:
            print("Not enough matches are found - {}/{}".format(len(good), self.min_mum))

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=Mask,  # draw only inliers
                           flags=2)
        img = cv2.drawMatches(self.dst_image, self.kp1, image, self.kp2, good, None, **draw_params)
        return img


class Patrol:
    def __init__(self, thresh1, thresh2, rate):
        self.low_thresh, self.high_thresh, self.rate = thresh1, thresh2, rate
        pass

    def __call__(self, img):
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img, (5, 5), 0.14)
        _, thresh = cv2.threshold(blur, self.low_thresh, self.high_thresh, cv2.THRESH_BINARY_INV)
        # Erode and dilate to remove accidental line detections
        mask = cv2.erode(thresh, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Find the contours of the frame
        contours, hierarchy = cv2.findContours(mask, 1, cv2.CHAIN_APPROX_NONE)
        # Find the biggest contour (if detected)
        if len(contours) <= 1:
            return -1
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < img.shape[0]*img.shape[1] * self.rate:
            return 0
        M = cv2.moments(c)
        # 空间矩 重心、质心
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # https://www.cnblogs.com/IllidanStormrage/p/16225144.html
        # 凸包轮廓
        hull = cv2.convexHull(c)
        # 计算轮廓的近似梯形形状，获得方向
        # arc = cv2.arcLength(c, True)    # 周长
        # trapezoid = cv2.approxPolyDP(c, arc*0.05, True)
        rect = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.int16)    # 近似矩形

        # 直线拟合 hough https://www.cnblogs.com/Gaowaly/p/18327735
        pass

        a, b, c, d = np.float32((rect[1, 1] - rect[0, 1], rect[3, 1] - rect[0, 1], rect[1, 0] - rect[0, 0], rect[3, 0] - rect[0, 0]))[:, None]
        l01 = np.sqrt(a * a + c * c)    # 矩形边长1
        l30 = np.sqrt(b * b + d * d)    # 矩形边长2
        s1 = a / l01    # 矩形方向1
        s2 = b / l30    # 矩形方向2
        c1 = c / l01
        c2 = d / l30
        return rect, (contours, hull), (cx, cy), (l01, l30, s1, s2, c1, c2)

    def draw(self, img):
        rect, (contours, hull), (cx, cy), (l01, l30, s1, s2, c1, c2) = self(img)
        # draw out center point
        cv2.circle(img, (cx, cy), 7, (255, 0, 0), 1)

        cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
        cv2.drawContours(img, [hull], -1, (0, 0, 255), 1)
        # cv2.drawContours(crop_img, [trapezoid], -1, (255, 0, 0), 1)
        cv2.polylines(img, [np.int32(rect[:, None])], True, (255, 0, 0), 3, cv2.LINE_AA)
        p1 = np.int32(((-c1 * l01, -s1 * l01), (c1 * l01, s1 * l01)))[:, :, 0] + (cx, cy)
        p2 = np.int32(((-c2 * l30, -s2 * l30), (c2 * l30, s2 * l30)))[:, :, 0] + (cx, cy)
        cv2.line(img, p1[0], p1[1], (155, 255, 0), 2)
        cv2.line(img, p2[0], p2[1], (155, 255, 0), 2)
        print(cx, cy, int(l01), int(l30))
