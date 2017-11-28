# -*- coding: utf-8 -*-
import os,sys
import urllib.request
import expandBib

class base(object):
    def __init__(self,userID,userPass):
        self.userID   = userID
        self.userPass = userPass
        self.cacheFolderName = ".cache"
        self.downloadURLBase = "yandy.bf1.jp"
        return

    def get(self,objectName,objectID):
        identifier = objectName+"_"+objectID

        # check cache file. If not, download
        checkPath = os.path.join(self.cacheFolderName,identifier+".zip")
        if not os.path.exists(checkPath):
            url = self.downloadURLBase+"?identifier=%s"%identifier
            obj = urllib.request.urlopen(url)
            local = open(checkPath, 'wb')
            local.write(obj.read())
            obj.close()
    # これ参考にロバストに:
    #https://github.com/fchollet/keras/blob/master/keras/utils/data_utils.py
