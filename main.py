from detect import ArmorDetector
from config import RoboConfig
import os
import cv2 as cv
import numpy as np
from utils.myserial import RoboSerial
from cap_manager import CapManager
import threading
import argparse
import logging
#logging.basicConfig(filename="./log/access.log",format="%(asctime)s - %(name)s  - %(levelname)s - %(module)s %(message)s",datefmt="%Y-%m-%d %H:%M:%S %p",level=10)
key = 0
parser = argparse.ArgumentParser()
parser.add_argument("-d","--debug",default = False,help="debug mode",action = "store_true")
parser.add_argument("--mode",default = "red")
args = parser.parse_args()
def serial_task(serial,detector):
    while True:
        word = serial.read()
        if len(word):
            if word[0] == 10:
                detector.set_mode("stable")
                logging.info("stable")
            elif word[0] == 20:
                detector.set_mode("rotating")
                logging.info("rotating")
if __name__ == "__main__":
    serial = RoboSerial()
    configer = RoboConfig()
    if not os.path.exists( "./config.cfg"):
        configer.reset()
        configer.dump()
    else:
        configer.load()
    mode = args.mode

    debug = args.debug
    
    try:
        detector = ArmorDetector(configer,mode = mode,debug = debug)
    except Exception as e:
        os.remove("./config.cfg")
        configer.reset()
        configer.dump()
        detector = ArmorDetector(configer,mode = mode,debug = debug)
        logging.error("detector error")
    
    t1 = threading.Thread(target = serial_task,args=(serial,detector,))
    t1.setDaemon(True) # 主线程一挂t1也会挂 ～～～ ctrl c
    t1.start()
    stop = False
    while not stop:
        cap = CapManager(configer.getcfg(),"./blue.mp4")
        try:
            for idx,_ in enumerate(cap):
                '''
                detector.set_mode
                '''
                #cv.imshow("frame",frame)
                if debug:
                    key = cv.waitKey(1)
                    if key == ord('s'):
                        detector.test("start")
                    if key == ord('d'):
                        detector.test("end")
                    if key == ord('g'):
                        configer.dump()
                    if key == ord('q'):
                        del cap  
                        stop = True
                        break
                ret,frame = _
                
                if not ret or frame is None:
                    cv.waitKey(100)
                    logging.warning("another try to open the cap!")
                    break
                pred_center = detector(frame)
                if pred_center is not None:
                    if pred_center[0] < 0 or pred_center[0] > 1440 or pred_center[1] <0 or pred_center[1] > 740:
                        serial.send([720.,200.])
                    else:
                        serial.send(pred_center)
                else:
                    serial.send([720.,200.])
        except KeyboardInterrupt:
            del cap
            stop = True
            logging.warning("keyboard ctrl c stop!")
            break        
            
    
        #else:
        #    serial.send(np.array([720.,540.]))
        
        
        
        

# time spend: 16.168384 ms
    
