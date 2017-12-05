import os, glob, cv2
import numpy as np
import tensorflow as tf

""" 
    Generate Dataset (X,Y) from Folder Path.
    Each subfolder is regarded to contain images per each class. 
"""
def get_dataset(_loadpath='data/',_rszshape=(28,28),_imgext='png',_VERBOSE=True):
    flist  = sorted(os.listdir(_loadpath))
    nclass = len(flist)
    """ 1. Compute the total number of images """
    n_total  = 0
    for fidx,fn in enumerate(flist): # For all folders
        plidst = sorted(glob.glob(_loadpath+fn+'/*.'+_imgext))
        if _VERBOSE:
            print ("[%d/%d] [%04d] images" %(fidx,nclass,len(plidst)))
        n_total = n_total + len(plidst)
    if _VERBOSE:
        print ("Total [%d] images." % (n_total))
    """ 2.  Load Data """
    X = np.zeros((n_total,_rszshape[0]*_rszshape[1]))
    Y = np.zeros((n_total,nclass))
    imgcnt = 0
    for fidx,fn in enumerate(flist): # For all folders
        plidst = sorted(glob.glob(_loadpath+fn+'/*.png'))
        for pn in plidst: # For all images per folder   
            img_raw = cv2.imread(pn, cv2.IMREAD_GRAYSCALE)
            img_rsz = cv2.resize(img_raw,_rszshape)
            img_vec = img_rsz.reshape((1,-1))
            """ Concatenate input and output to X and Y """
            X[imgcnt:imgcnt+1,:] = img_vec
            Y[imgcnt:imgcnt+1,:] = np.eye(nclass, nclass)[fidx:fidx+1,:]
            imgcnt = imgcnt + 1
    if _VERBOSE:
        print ('Done.')
    """ 3. Ramdom Shuffle with Fixed Random Seed """
    np.random.seed(seed=0)
    randidx = np.random.randint(imgcnt,size=imgcnt)
    X = X[randidx,:]
    Y = Y[randidx,:]
    return X, Y, imgcnt

"""
    Open session
"""
def gpusession():
    config = tf.ConfigProto(); 
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    return sess



