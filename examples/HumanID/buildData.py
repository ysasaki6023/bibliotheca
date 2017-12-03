import os,sys,glob,shutil

# First please download data from here: http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html

if not os.path.exists("data"):
    os.makedirs("data")

# CUHK1
last_pid = -1
path = "original/campus" # this needs to be modified according to your setup
for f in os.listdir(path):
    k = f.replace(".png","")
    pid, fid = int(k[:4]), int(k[4:])
    if pid>last_pid: last_pid = pid
    newf = "%04d_%02d.png"%(pid,fid)
    #os.symlink(os.path.join(path,f),os.path.join("data",newf))
    shutil.copyfile(os.path.join(path,f),os.path.join("data",newf))

print last_pid

sys.exit(-1)

# CUHK2
path = "original/Dataset" # this needs to be modified according to your setup
for p in ["P1","P2","P3","P4","P5"]:
    c = "cam1"
    pidList = {}
    for f in os.listdir(os.path.join(path,p,c)):
        if not f.endswith(".png"): continue
        k = f.replace(".png","")
        print k
        pid, fid = k.split("_")

        fList = glob.glob(os.path.join(path,p,"cam*","%s_*.png"%pid))
        last_pid += 1
        for idx,ff in enumerate(fList):
            newf = "%04d_%02d.png"%(last_pid,idx)
            os.symlink(ff,os.path.join("data",newf))
