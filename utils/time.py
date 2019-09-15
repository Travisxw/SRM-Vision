import cv2 as cv

def count_time(fn):
    def cv_count_time(*args,**kw):
        start = cv.getTickCount()
        pred_center = fn(*args,**kw)
        end = cv.getTickCount()
        during= (end - start) / cv.getTickFrequency()
        print("time spend: {:.2f} ms".format(during * 1000))
        return pred_center
    return cv_count_time
