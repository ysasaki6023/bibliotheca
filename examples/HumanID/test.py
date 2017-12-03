# -*- coding: utf-8 -*-
import os,sys,argparse,re,string,logging,random,csv,glob,cv2,math,h5py
import numpy as np
import pandas as pd

from analysisGAP import net

n = net(nBatch=1,learnRate=0,zdim = 128,isTraining=False,saveFolder=None)
n.buildModel()
n.reloadModel("save/test2_vgg16_zdim128_avgPooling/weights.177.hdf5")

#fileList = ["iwano2","iwano3","tanaka1","tanaka2","woman1"] + ["0012_01","0012_03"] + ["0045_02","0045_04"] + ["0026_02","0026_03"]
fileList = ["iwano2","iwano3","tanaka1","tanaka2","woman1"]
imgList = []
pidList = []

for fname in fileList:
    print fname
    img = cv2.imread("img/%s.png"%fname)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float64)

    # 入力は60x160pixで固定
    img = cv2.resize(img,(60,160),interpolation=cv2.INTER_CUBIC)

    # 人物ID (Person ID)を計算する
    pid = n.eval(img)

    imgList.append(img)
    pidList.append(pid)

# Person IDは128次元ベクトル。この間の距離が0.5以下だと同一人物
for f1,p1 in zip(fileList,pidList):
    for f2,p2 in zip(fileList,pidList):
        if f1==f2:continue
        print "(%10s,%10s): is same person? ="%(f1,f2),n.isSamePerson(p1,p2,threshold=0.5)
