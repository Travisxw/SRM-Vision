import serial
import numpy as np
import cv2 as cv
class RoboSerial():
    def __init__(self,com = "/dev/ttyUSB0",port = 9600,imgW = 1440,imgH = 740):#Linux com1
        self.serial = serial.Serial(com,port,timeout=0.5)
        self.imgH,self.imgW = imgH,imgW
        self.cloud_value = None
    def calc_value(self,detected_xy):
        assert detected_xy is not None
        #de_xy = detected_xy - [self.imgW/2.,self.imgH/2.]
        de_xy = detected_xy
        
        de_xy[1] = self.imgH-de_xy[1]
        #de_xy = de_xy + offsets
        # print(de_xy)
        if de_xy[0] < 0:
            de_xy[0] = de_xy[0]
        else:
            de_xy[0] = de_xy[0]
        if de_xy[1] < 0:
            de_xy[1] = de_xy[1]
        else:
            de_xy[1] = de_xy[1]
        
        #    signal_x = ord('c')
        #    de_xy[0] = -de_xy[0]
        #else:
        #    signal_x = ord("x")
        #if de_xy[1] < 0:
        #    signal_y = ord("c")
        #    de_xy[1] = -de_xy[1]
        #else:
        #    signal_y = ord("x")
        signal_x = ord('+')
        signal_y = ord('+')
        de_x,de_y = de_xy
        de_x = int(de_x)
        de_y = int(de_y)
        
        cloud_value=bytearray()
        cloud_value.append(ord('&'))#(38)
        cloud_value.append(ord('%'))#(37)
        cloud_value.append(de_x>>8&0xff)
        cloud_value.append(de_x&0xff)
        cloud_value.append(signal_x)
        cloud_value.append(de_y>>8&0xff)
        cloud_value.append(de_y&0xff)
        cloud_value.append(signal_y)
        sum = 0
        for i in range(len(cloud_value)):
            sum = sum + cloud_value[i]
        cloud_value.append(sum&0xff)
        self.cloud_value = cloud_value

    def send(self,pred_center):
        assert self.serial is not None
        if pred_center is None:
            return
        self.calc_value(pred_center)
        print(pred_center)
        self.serial.write(self.cloud_value)
    def read(self):
        return self.serial.read()

if __name__ == "__main__":
    import time
    ser = RoboSerial()
    while True:
        data = ser.read()
        time.sleep(0.1)
        if len(data):
            print(data[0])
