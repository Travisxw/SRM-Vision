import cv2 as cv
import copy
import numpy as np
from config import RoboConfig
import random
from make_feature_map import Feature_extractor
import glob
from filters import KalmanFilter
from utils.time import count_time
#import torch.nn as nn
#import torch
#from torchvision import transforms
#from mobilenet import model as md

#from PIL import Image
#import itertool
files = glob.glob("./images/*.jpg")
imgs = []
for file in files:
    img = cv.imread(file)
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    img = cv.resize(img,(32,40))
    imgs.append(img)
M = imgs
f = Feature_extractor(M)

class ArmorDetector():
    def __init__(self,configer,mode = "red",debug = False):
        self.configer = configer
        self.cfg = configer.getcfg()[mode + "_config"]
        self.mode = mode
        self.debug = debug
        self.positive = 0
        self.nn = 0
        self._test = False
        self.x_center = 0
        self.show = None
        self.detected_type = "stable"
        self.old_center = None
        self.r = 300
        self.left = 0
        self.kalman = KalmanFilter()
        if self.debug:
            cv.namedWindow("trackbar")
            cv.createTrackbar("thre1","trackbar",self.cfg["thre1"],360,self.trackbar_change)
            #cv.createTrackbar("thre2","trackbar",self.cfg["thre2"],360,self.trackbar_change)
            cv.createTrackbar("ratio_min","trackbar",self.cfg["ratio_min"],10,self.trackbar_change)
            cv.createTrackbar("ratio_max","trackbar",self.cfg["ratio_max"],10,self.trackbar_change)
            cv.namedWindow("number")
            cv.createTrackbar("hmin","number",0,180,self.trackbar_change)
            cv.createTrackbar("hmax","number",116,180,self.trackbar_change)
            cv.createTrackbar("smin","number",0,360,self.trackbar_change)
            cv.createTrackbar("smax","number",225,360,self.trackbar_change)
            cv.createTrackbar("vmin","number",10,360,self.trackbar_change)
            cv.createTrackbar("vmax","number",53,360,self.trackbar_change)
            
            #cv.createTrackbar("ss","trackbar",0,1000,self.trackbar_change)
            #cv.createTrackbar("offset_x","trackbar",self.cfg["offset_x"],1000,self.trackbar_change)
            #cv.createTrackbar("offset_y","trackbar",self.cfg["offset_y"],1000,self.trackbar_change)

            #cv.createButton("isRecord",self.trackbar_change,NULL,QT_CHECKBOX,0);
    def trackbar_change(self,o):
        thre1 = cv.getTrackbarPos("thre1","trackbar")
        #thre2 = cv.getTrackbarPos("thre2","trackbar")
        #offset_x = cv.getTrackbarPos("offset_x","trackbar")
        #offset_y = cv.getTrackbarPos("offset_y","trackbar")
        ratio_min = cv.getTrackbarPos("ratio_min","trackbar")
        ratio_max = cv.getTrackbarPos("ratio_max","trackbar")
        '''
        setattr(self,"cfg",{
            "thre1":thre1,
            "thre2":thre2,
            "offset_x":offset_x,
            "offset_y":offset_y,
            "ratio_min":ratio_min,
            "ratio_max":ratio_max
        })
        '''
        setattr(self,"cfg",{
            "thre1":thre1,
            "ratio_min":ratio_min,
            "ratio_max":ratio_max
        })
        '''
        setattr(self,"cfg",{
            "hmin":hmin,
            "hmax":hmax,
            "smin":smin,
            "smax":smax,
            "vmin":vmin,
            "vmax":vmax
        })
        '''
        self.configer.setColorConfig(mode = self.mode,color_cfg = self.cfg)
        self.configer.dump()
    def __del__(self):
        self.configer.setColorConfig(mode = self.mode,color_cfg = self.cfg)
    def old_area_find(self):
        
        h,w = self.frame.shape[:2]
        if self.old_center is not None:
            left = max(0,self.old_center[0] - self.r)
            right = min(w,self.old_center[0] + self.r)
            self.left = left
            if right > left:
                self.frame = self.frame[:,int(left):int(right)]
    
    def preprocess(self):
        '''
        这里没有使用opencv 的 split通道分离原因有下：
        1. 通道分离得到三个通道多余一个通道浪费时间
        2. opencv自带的split函数调用时间不稳定，容易出现很大的时间波动（底层实现有关）
        3. 使用numpy直接按索引获取时间稳定、时间消耗远小于opencv的split
        '''
        if self.mode == "red":
            r = self.frame[:,:,2:]
            ret,binary = cv.threshold(r,200,255,cv.THRESH_BINARY)
        else:
            b = self.frame[:,:,:1]
            ret,binary = cv.threshold(b,225,255,cv.THRESH_BINARY)
        #ret,binary_light = cv.threshold(b,140,255,cv.THRESH_BINARY)
        #binary = cv.inRange(hsv,np.array([self.cfg["hmin"],self.cfg["smin"],self.cfg["vmin"]]),np.array([self.cfg["hmax"],self.cfg["smax"],self.cfg["vmax"]]))
        
        #element = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        
        #binary = cv.erode(binary,element,iterations =1)
        
        #h,w = binary.shape[:2]
        
        _,contours,_ = cv.findContours(binary,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        return binary,contours
    
    def getLights(self,binary,contours,debug = False):
        out_lights = []
        if not len(contours):
            return out_lights
        
        for contour in contours:
            area = cv.contourArea(contour)
            if area < 10:
                continue
            
            
            (x,y),(w,h),theta = cv.minAreaRect(contour)
            rect = cv.minAreaRect(contour)
            if w>h:
                theta = theta + 90
                w,h = h,w
            if theta > 150 and theta < 200:
                theta = theta - 180
            if theta > 330:
                theta = theta - 360
            if theta > -30 and theta < 30:
                points = cv.boxPoints(rect)
                bbox = cv.boundingRect(points)
                
                
                if h and w and w/h > 0.85:
                    continue
                
                if h< 5 and w<5:
                    pass
                
                if self.debug:
                    if self.old_center is not None:
                        cv.rectangle(self.show,(bbox[0] + int(self.left),bbox[1]),(bbox[0] + bbox[2] + int(self.left),bbox[1] + bbox[3]),(0,255,255),2)
                    else:
                        cv.rectangle(self.show,(bbox[0],bbox[1]),(bbox[0] + bbox[2],bbox[1] + bbox[3]),(0,255,255),2)
                out_lights.append((points,theta))
                if debug:
                    cv.imwrite("./images2/bar_{:8d}.jpg".format(random.randint(0,99999999)),self.frame[bbox[1]:bbox[1] + bbox[3],bbox[0]:bbox[0] + bbox[2]])

        out_lights.sort(key = lambda x:x[0][0][0] + x[0][1][0] + x[0][2][0] + x[0][3][0])
        return out_lights
    
    def detect(self,lights,debug = False):
        center = None
        armor = None 
        armors = []
        height = 0
        de_min_k = 9999
        last_number = None
        if self.debug:
            ratio_min = cv.getTrackbarPos("ratio_min","trackbar") * 0.1
            ratio_max = cv.getTrackbarPos("ratio_max","trackbar") * 0.1
            hmin = cv.getTrackbarPos("hmin","number")
            hmax = cv.getTrackbarPos("hmax","number")
            smin = cv.getTrackbarPos("smin","number")
            smax = cv.getTrackbarPos("smax","number")
            vmin = cv.getTrackbarPos("vmin","number")
            vmax = cv.getTrackbarPos("vmax","number")
        else:
            ratio_min = self.cfg["ratio_min"]* 0.1
            ratio_max = self.cfg["ratio_max"]* 0.1
            hmin,hmax,smin,smax,vmin,vmax = 0,116,0,200,10,53
        b = None
        results = []
        if lights is None or len(lights)< 2:
            return None,None
        for i in range(0,len(lights)): # i
            for j in range(i,len(lights)):
                if i==j:
                    continue
                light1 = sorted(lights[i][0],key = lambda x:(x[0]**2 + x[1]**2))
                light2 = sorted(lights[j][0],key = lambda x:(x[0]**2 + x[1]**2))
                theta1,theta2 = lights[i][1],lights[j][1]
                
                if abs(theta1 - theta2) > 30:
                    continue
                if theta1 >2 and theta2 < -2:
                    continue
                am = (tuple(light1[0]),tuple(light2[-1]))
                #amh,amw = am[1][1] - am[0][1], am[1][0] - am[0][0]
                amh = max(light1[-1][1] - light1[0][1],light2[-1][1] - light2[0][1])
                amw = am[1][0] - am[0][0]
                if amw/amh>4:
                    continue
                x_1 = (light1[0][0] + light1[3][0])/2
                y_1 = (light1[0][1] + light1[3][1])/2
                x_2 = (light2[0][0] + light2[3][0])/2
                y_2 = (light2[0][1] + light2[3][1])/2
                
                de_x = abs(x_1 - x_2) + 1e-9
                de_y = abs(y_1 - y_2)
                
                if de_y > 30:
                    continue

                kk = de_y / de_x
                
                if kk>0.7:
                    continue
                rx = abs(am[0][0] - am[1][0])
                ry = abs(am[0][1] - am[1][1])
                
                center = [(am[0][0] + am[1][0])/2.,(am[0][1] + am[1][1])/2.]
                
                left = max(0,int(center[0] -rx*0.3))
                top = max(0,int(center[1] -ry*1.5))
                right = min(self.size[0],int(center[0] +rx*0.3))
                bottom = min(self.size[1],int(center[1] +ry*1.5))
                

                roi = self.frame[top:bottom,left:right]
                hh,ww = roi.shape[:2]
                

                
                if hh<5 or ww==0:
                    continue
                #print(ww,hh,ww/hh)
                
                
                roi = cv.cvtColor(roi,cv.COLOR_BGR2HSV)
                binary = cv.inRange(roi,np.array([hmin,smin,vmin]),np.array([hmax,smax,vmax]))
                # 调高曝光看中间数字，统计threshold之后white点比例，满足一定比例则为要求装甲板。
                #cv.imshow("roi",binary)
                element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

                binary = cv.dilate(binary,element,iterations =1)
                
                
                #print("the number is:",number,"the score is",score)
                
                aaa = np.sum(binary>0)/(ww*hh)

                if aaa< ratio_min or aaa > ratio_max:
                    continue
                
                k_1 = (light1[0][0] - light1[3][0]) / (light1[0][1] - light1[3][1] + 1e-5)
                k_2 = (light2[0][0] - light2[3][0]) / (light2[0][1] - light2[3][1] + 1e-5)
                #print(abs(k_1 - k_2))
                if abs(k_1 - k_2) > 0.5:
                    continue
                feature = f(cv.resize(binary,(32,40)))
                result,score = f.detect(feature)
                number = files[result].split("/")[-1].split("_")[0]
                if int (number)==0:
                    continue
                if int(number) == 9:
                    if self.debug:
                        temp = (tuple(light1[0]),tuple(light2[-1]))
                        if self.old_center is not None:
                            cv.rectangle(self.show,(int(temp[0][0] + self.left),temp[0][1]),(int(temp[1][0] + self.left),temp[1][1]),(0,22,123),2)
                            cv.putText(self.show,"pass 4".format(number),(int(temp[0][0] + self.left),temp[0][1]),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
                        else:
                            cv.rectangle(self.show,*temp,(0,123,123),2)
                            cv.putText(self.show,"pass 4".format(number),(int(temp[0][0]),temp[0][1]),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
                    continue
                else:
                    print("Got number:{}".format(number))
                area_1 = cv.contourArea(lights[i][0]) + 1e-5
                area_2 = cv.contourArea(lights[j][0]) + 1e-5
                #print(0.025*abs(k_1 - k_2),0.975*(1./area_1 + 1./area_2))
                #print(0.015 * abs(k_1 - k_2),2.1*(1./area_1 + 1./area_2))
                de_k = 0.015 * abs(k_1 - k_2)  + 2.1*(1./area_1 + 1./area_2)
                if de_k < de_min_k:
                    de_min_k = de_k
                    armor = (tuple(light1[0]),tuple(light2[-1]))
                    height = amh
                    #armors.append((tuple(light1[0]),tuple(light2[-1])))
                    last_number = number
                    if self.debug:
                        b = binary
                    
        if armor is not None:
            #armor = sorted(armors,key = lambda x: x[0][1] + x[1][1])[-1]
            #bh,bw = b.shape[:2]
            #if score < 120:
            #    #print(score)
            #    number = files[result].split("/")[-1].split("_")[0]
            #    print(number)
            #else:
            #    number = None
            #    #print(number)
            if self.debug:
                
                
                
                if self.old_center is not None:
                    cv.rectangle(self.show,(int(armor[0][0] + self.left),armor[0][1]),(int(armor[1][0] + self.left),armor[1][1]),(0,0,255),2)
                    cv.putText(self.show,"got {}".format(last_number),(int(armor[0][0] + self.left),armor[0][1]),cv.FONT_HERSHEY_COMPLEX,1,(255,0,255),1)
                else:
                    cv.rectangle(self.show,*armor,(0,0,255),2)
                    cv.putText(self.show,"got {}".format(last_number),(int(armor[0][0]),armor[0][1]),cv.FONT_HERSHEY_COMPLEX,1,(255,0,255),1)
                
                    #cv.putText(self.show,"the number is {}".format(number),(50,50),cv.FONT_HERSHEY_COMPLEX,2,(0,0,255))
                key = cv.waitKey(1)
                if key == ord('n'):
                    cv.imwrite("images/{}.jpg".format(random.randint(0,99999)),b)
                cv.imshow("b",b)
                #cv.imwrite("images/{}.jpg".format(random.randint(0,99999)),b)
            center = [(armor[0][0] + armor[1][0])/2.,(armor[0][1] + armor[1][1])/2.]
            #center = (np.array(armor[0]) + np.array(armor[1])) / 2.
            #height = armor[1][1] - armor[0][1]
            
            #cv.circle(self.show,tuple(center),9,(255,255,123),-1)
        else:
            return None,None
        if debug:
            try:
                im2write = self.raw[int(armor[0][1]):int(armor[1][1]),int(armor[0][0]):int(armor[1][0])]
                imh,imw = im2write.shape[:2]
                if imh * imw > 2:
                    cv.imwrite("./armors/{:6d}.jpg".format(random.randint(0,999999)),im2write)
            except Exception as e:
               print(e)
        if height:
            return center,height
        else:
            return None,0

    def draw_box(self,frame,points,color = (0,255,255)) :
        for i in range(4):
            #print(points[i])
            cv.line(frame,tuple(points[i%4]),tuple(points[(i+1)%4]),color,2)
        return frame
    
    def predict_value(self,center,height):
        
        self.old_center = center
        if center is None:
            return None
        
        center[0] = center[0] - 10
        if self.detected_type == 'rotating':
            cv.circle(self.show,(int(center[0]),int(center[1])),5,(0,255,255),-1)
            
            self.kalman.correct(np.array(center))
            center[0] = float(self.kalman.predict()[0])
        center[1] = center[1]  - 100
        center[1] = center[1] - 1000./height
        return center
    def set_mode(self,mode = "stable"):
        self.detected_type = mode
    def test(self,mode):
        if mode == 'start':
            self._test = True
            self.nn = 1
        elif mode == "end":
            print(self.positive/ self.nn)
            self.positive = 0
            self._test = False
    @count_time
    def __call__(self,frame):
        #start = cv.getTickCount()
        if self.debug:
            self.show = copy.deepcopy(frame)
        self.frame = frame
        h,w = self.frame.shape[:2]
        
        self.old_area_find()
        # 动态roi 5.0~6ms
        # 不动态roi 5.8~6ms
        # 大概省1ms
        self.size = self.frame.shape[:2][::-1]
        
        binary,contours = self.preprocess()
        lights = self.getLights(binary,contours)
        center,height = self.detect(lights,debug = False)
        
        if self.old_center is not None and center is not None:
            center[0] = center[0] + self.left
        
        pred_center = self.predict_value(center,height)
        
        if pred_center is not None:
            pred_center = pred_center[:2]
            if self._test:
                self.positive = self.positive + 1
            if self.debug:
                cv.circle(self.show,(int(pred_center[0]),int(pred_center[1])),5,(0,255,0),-1)
        if self.debug:
            #ss = cv.getTrackbarPos("ss","trackbar")
            #cv.line(self.show,(0,h - ss),(w,h - ss),(0,0,255),2)
            cv.circle(self.show,(w//2,h//2),7,(0,255,255),-1)
            cv.imshow("show",cv.resize(self.show,(w//2,h//2)))
            cv.imshow("binary",cv.resize(binary,(w//2,h//2)))
        
        self.nn = self.nn + 1
        return pred_center
        '''
        center = self.recgnize(lights)
        pred_center = self.predict_value(center)
        
        
        
        b = cv.resize(self.show,(w//2,h//2))  
        cv.imshow("rect",b)
        return pred_center
        '''


