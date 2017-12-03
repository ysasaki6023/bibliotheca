# -*- coding: utf-8 -*-
import os,sys,argparse,re,string,logging,random,csv,glob,cv2,math,h5py
import numpy as np
import pandas as pd
from keras.layers import Input, Dense, LSTM, merge, Lambda, GRU, Dot, Reshape, Concatenate, Flatten, Dropout, Bidirectional, TimeDistributed, Activation, Conv3D, MaxPooling3D, GlobalMaxPooling2D, MaxPooling1D, GlobalAveragePooling1D, BatchNormalization, Conv2D, MaxPooling2D, GlobalMaxPooling1D, GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.constraints import min_max_norm
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l2
import keras.backend as K
import matplotlib
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
from keras.backend import tensorflow_backend
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from imgaug import augmenters as iaa

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

#####################################
class net(object):
    def __init__(self,isTraining=True,nBatch=1,zdim=128,doFineTune=False,learnRate=1e-5,saveFolder="save"):
        self.nColor = 3
        self.fps = 30
        self.doFineTune = doFineTune

        self.sizeX = 60
        self.sizeY = 160

        self.nBatch  = nBatch
        self.zdim = zdim
        self.learnRate = learnRate
        self.saveFolder = saveFolder
        self.isTraining = isTraining

        if isTraining:
            self.fileList = {"train":{},"test":{}}
            for f in glob.glob("data/*.png"):
                pid, fid = os.path.basename(f).replace(".png","").split("_")
                pid, fid = int(pid), int(fid)
                if pid%10==0: mode="test"
                else:         mode="train"
                if not pid in self.fileList[mode]: self.fileList[mode][pid]=[]
                self.fileList[mode][pid].append(f)

            print self.fileList

            self.imgAugSeq = iaa.Sequential([iaa.GaussianBlur((0.,1.5)),
                                             iaa.ContrastNormalization((0.5,1.5)),
                                             iaa.Multiply((0.75,1.25)),
                                             iaa.PerspectiveTransform(scale=(0.,0.075))
                                             ])
        return

    def loadOnePair(self,mode="all",samePair=True):
        """
        samePair: 同じ種別のペアにするか否か
        """
        fileList = self.fileList[mode]
        pid1 = random.choice(fileList.keys())
        if samePair: pid2 = pid1
        else:
            while True:
                pid2 = random.choice(fileList.keys())
                if not pid2==pid1: break

        fname1 = random.choice(fileList[pid1])
        fname2 = random.choice(fileList[pid2])

        img1 = cv2.imread(fname1)
        img2 = cv2.imread(fname2)
        
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

        img1 = cv2.resize(img1,(self.sizeX,self.sizeY),interpolation=cv2.INTER_CUBIC)
        img2 = cv2.resize(img2,(self.sizeX,self.sizeY),interpolation=cv2.INTER_CUBIC)

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        # pre-process
        img1 = preprocess_input( np.expand_dims(img1,axis=0) )[0]
        img2 = preprocess_input( np.expand_dims(img2,axis=0) )[0]

        return img1,img2
        
    def loadBatch(self,mode="all",samePairFrac=0.5):
        batch_inX1  = np.zeros( (self.nBatch, self.sizeY, self.sizeX, self.nColor) )
        batch_inX2  = np.zeros( (self.nBatch, self.sizeY, self.sizeX, self.nColor) )
        batch_label = np.zeros( (self.nBatch) )

        for iBatch in range(self.nBatch):
            if iBatch<(samePairFrac*self.nBatch): label, flag = 1., True
            else:                                 label, flag = 0., False
            img1, img2 = self.loadOnePair(mode=mode,samePair=flag)
            batch_inX1[iBatch]  = img1
            batch_inX2[iBatch]  = img2
            batch_label[iBatch] = label

        # augumentation?
        batch_inX1 = self.imgAugSeq.augment_images(batch_inX1)
        batch_inX2 = self.imgAugSeq.augment_images(batch_inX2)

        return {"inX1":batch_inX1,"inX2":batch_inX2},batch_label

    def yieldOne(self,mode="all"):
        while True:
            yield self.loadBatch(mode)

    def reloadModel(self,fPath):
        self.model.load_weights(fPath)
        print "model loaded from %s"%fPath
        return

    def loadModel(self,fPath):
        self.model = load_model(fPath)
        print "model loaded from %s"%fPath

    def buildModel(self):
        input_shape = (self.sizeY, self.sizeX, self.nColor)
        inImg1 = Input(shape=input_shape,name="inX1")
        inImg2 = Input(shape=input_shape,name="inX2")

        def stream(input_shape,output_dim=128):

            vgg = VGG16(weights="imagenet",include_top=False)
            if not self.doFineTune:
                for layer in vgg.layers:
                    layer.trainable = False
            myvgg = Model(inputs=vgg.input, outputs=vgg.get_layer('block5_pool').output)

            inX = Input(shape=input_shape)
            h = myvgg(inX)
            #h = GlobalAveragePooling2D()(h)
            h = Flatten()(h)
            h = Dense(output_dim*4,activation="relu")  (h)
            h = Dropout(0.5)(h)
            h = Dense(output_dim*2,activation="relu")  (h)
            h = Dense(output_dim  ,activation="linear")(h)
            h = Activation("tanh")(h)
            h = Lambda(lambda x: K.l2_normalize(x,axis=-1))(h)
            return Model(inputs=inX,outputs=h)

        s1 = stream(input_shape=input_shape,output_dim=self.zdim)(inImg1)
        s2 = stream(input_shape=input_shape,output_dim=self.zdim)(inImg2)

        def euclidean_distance(vects):
            x, y = vects
            return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

        dist = Lambda(euclidean_distance)([s1,s2])
        model = Model(inputs=[inImg1,inImg2], outputs=dist)
        eval_model = Model(inputs=inImg1, outputs=s1)

        def contrastive_loss(y_true, y_pred):
            margin = 1
            return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

        def accuracy(y_true, y_pred):
            margin = 1
            return K.mean(K.equal(y_true, K.cast(y_pred < (0.5*margin), y_true.dtype)))

        model.compile(loss=contrastive_loss,optimizer=Adam(self.learnRate),metrics=[accuracy])

        model.summary()
        self.model = model
        self.eval_model = eval_model
        self.graph = tf.get_default_graph()

        return

    def train(self):
        cp_cb = ModelCheckpoint(filepath = self.saveFolder+"/weights.{epoch:02d}.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto')
        tb_cb = TensorBoard(log_dir=self.saveFolder, histogram_freq=1)

        self.model.fit_generator(generator=self.yieldOne("train"),
                                epochs=100000000,
                                callbacks=[cp_cb,tb_cb],
                                validation_data = self.yieldOne("test"),
                                validation_steps = 20,
                                steps_per_epoch=500,
                                use_multiprocessing=True,
                                workers=3)

    def eval(self,img):
        assert img.shape == (self.sizeY, self.sizeX, self.nColor)
        img = preprocess_input( np.expand_dims(img,axis=0) )[0]
        res = self.eval_model.predict(np.expand_dims(img,axis=0))[0]
        return res

    def isSamePerson(self,vec1,vec2,threshold=0.5):
        dist = np.linalg.norm(vec2-vec1)
        return dist
        if dist<threshold: return True
        else: return False


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nBatch" ,"-b",dest="nBatch",type=int,default=256)
    parser.add_argument("--zdim" ,"-z",dest="zdim",type=int,default=128)
    parser.add_argument("--learnRate",dest="learnRate"  ,type=float,default=1e-4)
    parser.add_argument("--doFineTune","-f",dest="doFineTune"  ,action="store_true")
    parser.add_argument("--reload","-r",dest="reload"  ,type=str,default=None)
    parser.add_argument("--saveFolder","-s",dest="saveFolder"  ,type=str,default="save")
    parser.add_argument("--test","-t",dest="test"  ,type=str,default=None)
    parser.add_argument("--testSave",dest="testSave"  ,type=str,default=None)
    args = parser.parse_args()

    if args.test:
        assert args.reload, "please set model to use"
        n = net(nBatch=1,
                learnRate=0,
                zdim = args.zdim,
                isTraining=False,
                saveFolder=None)
        n.buildModel()
        if args.reload:
            n.reloadModel(args.reload)
        n.test(args.test,args.testSave)
    else:
        n = net(nBatch=args.nBatch,
                learnRate=args.learnRate,
                zdim = args.zdim,
                doFineTune = args.doFineTune,
                saveFolder=args.saveFolder)
        n.buildModel()
        if args.reload:
            n.reloadModel(args.reload)
        n.train()
