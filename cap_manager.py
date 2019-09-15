# -- coding: utf-8 --
import sys
import copy
import cv2 as cv
from ctypes import *
sys.path.append("./MvImport")
import numpy as np
import random
from MvImport.MvCameraControl_class import *
class CapManager():
    def __init__(self,cfg = None,video_path = "",debug = False):
        self.cfg = cfg
        self.debug = debug
        print(cfg)
        # print(video_path)
        self.fps = None
        SDKVersion = MvCamera.MV_CC_GetSDKVersion()
        print("SDKVersion[0x%x]" % SDKVersion)
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        # ch:枚举设备 | en:Enum device
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            print ("enum devices fail! ret[0x%x]" % ret)
            sys.exit()

        if deviceList.nDeviceNum == 0:
            print ("find no device!")
            sys.exit()

        print ("Find %d devices!" % deviceList.nDeviceNum)
        mvcc_dev_info = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
        strModeName = ""
        for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
            if per == 0:
                break
            strModeName = strModeName + chr(per)
        print ("device model name: %s" % strModeName)

        strSerialNumber = ""
        for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
            if per == 0:
                break
            strSerialNumber = strSerialNumber + chr(per)
        print ("user serial number: %s" % strSerialNumber)

        # ch:创建相机实例 | en:Creat Camera Object
        self.cam = MvCamera()
        
        # ch:选择设备并创建句柄| en:Select device and create handle
        stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents

        ret = self.cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            print ("create handle fail! ret[0x%x]" % ret)
            sys.exit()

        # ch:打开设备 | en:Open device
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print ("open device fail! ret[0x%x]" % ret)
            sys.exit()

        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            print ("set trigger mode fail! ret[0x%x]" % ret)
            sys.exit()

        # ch:获取数据包大小 | en:Get payload size
        stParam =  MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
        
        ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            print ("get payload size fail! ret[0x%x]" % ret)
            sys.exit()
        nPayloadSize = stParam.nCurValue

        # ch:开始取流 | en:Start grab image
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            print ("start grabbing fail! ret[0x%x]" % ret)
            sys.exit()
        self.nPayloadSize = nPayloadSize
        self.data_buf = (c_ubyte * nPayloadSize)()


        #self.stFrameInfo = MV_FRAME_OUT_INFO_EX()
        #memset(byref(self.stFrameInfo), 0, sizeof(self.stFrameInfo))
        if cfg is not None and cfg['isRecord']:
            self.videoWriter = cv.VideoWriter(cfg["recordPath"] + "/" + self.give_me_filename(),cv.VideoWriter_fourcc('X', 'V', 'I', 'D'),30.0,(740,1080),True)
        

    def save_video(self,frame):
        self.videoWriter.write(frame)
    def __getitem__(self,_):
        if self.debug:
            start = cv.getTickCount()
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
        ret = self.cam.MV_CC_GetOneFrameTimeout(byref(self.data_buf), self.nPayloadSize, stFrameInfo, 1000)
        
        if ret==0:
            #help(self.data_buf)
            frame = np.array(self.data_buf,dtype = np.uint8).reshape(740,1440,1)
            self.frame = cv.cvtColor(frame,cv.COLOR_BayerBG2BGR)
            if self.cfg is not None and self.cfg["isRecord"]:
                self.save_video(self.frame)
            if self.debug:
                end = cv.getTickCount()
                print((end - start)/cv.getTickFrequency()*1000,"ms")
 
        
            return True,self.frame
        print("no frame")
        return None,None
    def __del__(self):
        '''
        if self.cfg["isVideo"]:
            self.videoWriter.release()
        self.cap.release()
        '''
        ret = self.cam.MV_CC_StopGrabbing()
        if ret != 0:
            print ("stop grabbing fail! ret[0x%x]" % ret)
            del self.data_buf
            sys.exit()

        # ch:关闭设备 | Close device
        ret = self.cam.MV_CC_CloseDevice()
        if ret != 0:
            print ("close deivce fail! ret[0x%x]" % ret)
            del self.data_buf
            sys.exit()

        # ch:销毁句柄 | Destroy handle
        ret = self.cam.MV_CC_DestroyHandle()
        if ret != 0:
            print ("destroy handle fail! ret[0x%x]" % ret)
            del self.data_buf
            sys.exit()

        del self.data_buf
    def give_me_filename(self):
        return "{:6d}.avi".format(random.randint(0,999999))
    


if __name__ == "__main__":
    cap = CapManager(debug = True)
    for ret,frame in cap:
        b,g,r = cv.split(frame)
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        ret,binary = cv.threshold(r,200,255,cv.THRESH_BINARY)
        ret,binary2 = cv.threshold(gray,140,255,cv.THRESH_BINARY)
        
        #ret,binary2 = cv.threshold(r,140,255,cv.THRESH_BINARY)
        #binary= cv.bitwise_and(binary1,binary2)
        _,contours_color,_ = cv.findContours(binary,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        _,contours_gray,_ = cv.findContours(binary2,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours_gray:
            area = cv.contourArea(contour)
            if area < 20 or area > 1e5:
                continue
            for contour_c in contours_color:
                
                if cv.pointPolygonTest(contour_c,tuple(contour[0][0]),False) > 0:
                    (x,y),(w,h),theta = cv.minAreaRect(contour)
                    cv.rectangle(frame,(int(x),int(y)),(int(x+w),int(h+y)),(0,255,255),2)                

        cv.imshow("frame",frame)
        #cv.imshow("frame2",binary2)
        key = cv.waitKey(1)
        if key == ord('q'):
            break
        
        
    cv.destroyAllWindows()
