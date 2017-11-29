# -*- coding: utf-8 -*-
import bibliotheca.base

if __name__=="__main__":
    bMgr = bibliotheca.bibManager(userID="myName",userPass="myPass")
    vggClass = bMgr.get(objectName="VGG16",objectID="XXXXX-XXXXX")
    model = vggClass(x)
