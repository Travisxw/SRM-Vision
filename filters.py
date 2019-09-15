import cv2 as cv
import numpy as np


class KalmanFilter():
    def __init__(self,delta_time = 0.1,a = 20000,center = [0,0]):
        self._center = center
        self.kalman = cv.KalmanFilter(4,2,0)# 状态空间4D 分别是x y vx vy，测量空间2D 分别是 x y
        self.kalman.transitionMatrix = np.array([[1,0,delta_time,0],[0,1,0,delta_time],[0,0,1,0],[0,0,0,1]],dtype = np.float32)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],dtype = np.float32)
        self.kalman.statePre = np.array([[self._center[0]],[self._center[1]],[0],[0]],dtype = np.float32)
        self.kalman.statePost = np.array([[self._center[0]],[self._center[1]],[0],[0]],dtype = np.float32)
        self.kalman.processNoiseCov = a * np.array([
            [0.25*delta_time**4,0,0.5*delta_time**3,0],
            [0,0.25*delta_time**4,0,0.5*delta_time**3],
            [0.5*delta_time**3,0,delta_time**2,0],
            [0,0.5*delta_time**3,0,delta_time**2]],dtype = np.float32)
        # 当成匀变速直线运动计算的kalman滤波参数
    def correct(self,center):
        s = np.array([[np.float32(center[0])],[np.float32(center[1])]])
        self.kalman.correct(s)
    def predict(self):
        return self.kalman.predict()