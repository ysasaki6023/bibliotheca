# -*- coding: utf-8 -*-
import os,sys
import urllib.request
import expandBib
import os,sys
from Crypto.Cipher import AES
import hashlib
import base64

############################################################
## Utility functions
def get_encrypt_data(raw_data, key, iv):
    raw_data_base64 = base64.b64encode(raw_data)
    # 16byte
    if len(raw_data_base64) % 16 != 0:
        raw_data_base64_16byte = raw_data_base64
        for i in range(16 - (len(raw_data_base64) % 16)):
            raw_data_base64_16byte += "_"
    else:
        raw_data_base64_16byte = raw_data_base64
    secret_key = hashlib.sha256(key).digest()
    iv = hashlib.md5(iv).digest()
    crypto = AES.new(secret_key, AES.MODE_CBC, iv)
    cipher_data = crypto.encrypt(raw_data_base64_16byte)
    cipher_data_base64 = base64.b64encode(cipher_data)
    return cipher_data_base64

#####
def get_decrypt_data(cipher_data_base64, key, iv):
    cipher_data = base64.b64decode(cipher_data_base64)
    secret_key = hashlib.sha256(key).digest()
    iv = hashlib.md5(iv).digest()
    crypto = AES.new(secret_key, AES.MODE_CBC, iv)
    raw_data_base64_16byte = crypto.decrypt(cipher_data)
    raw_data_base64 = raw_data_base64_16byte.split("_")[0]
    raw_data = base64.b64decode(raw_data_base64)
    return raw_data

############################################################
## Main class
class base(object):
    ##### Initializer
    def __init__(self,userID,userPass):
        self.userID   = userID
        self.userPass = userPass
        self.downloadURLBase = "yandy.bf1.jp"
        return

    ##### download file and execute
    def get(self,objectName,objectID):
        # 1. build cache folder if it does not exist
        cachedir = os.path.expanduser(os.path.join('~', '.bibliotheca'))
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)

        # 2. identify the file and check if download is required
        identifier = objectName+"_"+objectID
        fpath       = os.path.join(cachedir,identifier+".tar.gz")
        untar_fpath = os.path.join(cachedir,identifier)

        download = False
        untar    = False
        if os.path.exists(untar_fpath):
            # Folder found. Here we assume the contents are correct
            download = False
        else:
            # Folder not found. We need to check if the tar.gz file have been downloaded
            if os.path.exists(fpath):
                # File found; verify integrity if a hash was provided.
                untar = True
            else:
                download = True
                untar = True

        # 3. download if required
        if download:
            # 3.1 Set download path
            urlBase = "http://XXXX/"
            origin = urlBase + identifier

            # 3.2 Download setup
            class ProgressTracker(object):
                # Maintain progbar for the lifetime of download.
                # This design was chosen for Python 2.7 compatibility.
                progbar = None

            def dl_progress(count, block_size, total_size):
                if ProgressTracker.progbar is None:
                    if total_size is -1:
                        total_size = None
                    ProgressTracker.progbar = Progbar(total_size)
                else:
                    ProgressTracker.progbar.update(count * block_size)

            # 3.3 Actual download
            print('Downloading data from', )
            error_msg = 'URL fetch failure on {}: {} -- {}'
            try:
                try:
                    urlretrieve(origin, fpath, dl_progress)
                except URLError as e:
                    raise Exception(error_msg.format(origin, e.errno, e.reason))
                except HTTPError as e:
                    raise Exception(error_msg.format(origin, e.code, e.msg))
            except (Exception, KeyboardInterrupt) as e:
                if os.path.exists(fpath):
                    os.remove(fpath)
                raise
            ProgressTracker.progbar = None

        # 4. untar if required
        if untar:
            untar_fpath = fpath.replace(".tar.gz","")
            if not os.path.exists(untar_fpath):
                with tarfile.open(fpath) as archive:
                    try:
                        archive.extractall(fpath)
                    except (tarfile.TarError, RuntimeError,KeyboardInterrupt):
                        if os.path.exists(untar_fpath):
                            if os.path.isfile(untar_fpath):
                                os.remove(untar_fpath)
                            else:
                                shutil.rmtree(untar_fpath)
                    raise

        # 5. melt obj file
        objPath = os.path.join(untar_fpath,"bibObj.py")
        objData = open(objPath,"b").read()
        objExec = exec(melt(bibBase=self,bibObj=None,data=objData,comment=None))
        objClass = makeGKClass(objExec,bibMgr=self)

        return objClass

############################################################
# 一応、super classへアクセスする手段は残されているが、、、
def makeGKClass(bibObject,bibMgr):
    class GKClass(bibObj):
        def __init__(self,*arg,**kwdarg):
            self.bibMgr = bibMgr
            # ToDo: report to the server
            return super(bibObj, self).__init__(*arg,**kwdarg)
        def __call__(self,*arg,**kwdarg):
            # ToDo: report to the server
            return super(bibObj, self).__call__(*arg,**kwdarg)
    return GKClass(bibObject)


############################################################
# melt()は、２つの機能を持つ (1) サーバーへ何の関数を何回実行したかを伝える (2) 暗号化されたデータを解凍する
# dataがNoneのときには、単に情報を伝えるだけ
# 一応、書き換えられないようにバイナリで提供する (未実装)
def melt(bibBase,bibObj,data=None,comment=None):
    # bibBase,bibObjの情報を適宜参照して、送信する情報を決定する
    return data
