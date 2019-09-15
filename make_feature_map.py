'''
@ xueoaru
   2019
'''


import numpy as np
import cv2 as cv
from numpy.linalg import eig
class Feature_extractor():
    def __init__(self,training_M):
        self.imgW = 32
        self.imgH = 40
        self.M = np.array(training_M).reshape((-1,self.imgW*self.imgH)) # (m,D)
        self.t = self.M - np.mean(self.M,axis = 0) # (m,D)
        C = self.t.dot(self.t.T) # (m,m)
        vals,vecs = eig(C) # 特征值，特征向量 (m,) (m,m)
        indices = np.argsort(vals)
        U = vecs[indices[:-11:-1],:] #(10,m)
        self.U = U.dot(self.t) # (10,D)
        
    def __call__(self,x):
        x = x.reshape((-1,self.imgW*self.imgH)) # (1,D)
        Dx = x - np.mean(self.M,axis = 0) # (1,D)
        return self.U.dot(Dx.T) # (m,1)
    def detect(self,feature):
        df = self.U.dot(self.t.T) - feature #(m,m)
        df = np.sum(df**2,axis = 0)
        result = np.sqrt(df)/1e5
        
        idx = np.argmin(result)
        score = result[idx]
        return idx,score
        #print(np.argmin(result))
        #print(result)


if __name__ == "__main__":
    import glob
    files = glob.glob("./images/*.jpg")
    imgs = []
    for file in files:
        img = cv.imread(file)
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img = cv.resize(img,(32,40))
        imgs.append(img)
    M = imgs
        #M = [[[1,2,3],[4,5,6],[7,8,9]],[[1,3,3],[4,4,6],[7,7,9]]]
    f = Feature_extractor(M)
    print(f.U)
    feature_img = f.U[1,:].reshape((40,32))
    cv.imshow("r",feature_img)
    cv.waitKey(0)
    #feature = f(imgs[1])
    #print(files[1])
    #print(files)
    #f.detect(feature)