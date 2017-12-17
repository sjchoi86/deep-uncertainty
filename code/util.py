import os, glob, cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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
    Open GPU Session
"""
def gpusession():
    config = tf.ConfigProto(); 
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    return sess

"""
    Plot Grid Images
"""
def plot_grid_imgs(_imgs,_nr=1,_nc=10,_imgshp=[28,28],_figsize=(15,2),_title=''):
    nr,nc = _nr,_nc
    fig = plt.figure(figsize=_figsize)
    fig.suptitle(_title, size=15)
    gs  = gridspec.GridSpec(nr,nc)
    gs.update(wspace=0.05, hspace=0.05)
    for i, img in enumerate(_imgs):
        if np.issubdtype(img.reshape(-1)[0], int):
            img = np.clip(img,0,255)
        else:
            img = np.clip(img,0.0,1.0)
        
        ax = plt.subplot(gs[i]); plt.axis('off')
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_aspect('equal')
        plt.imshow(img.reshape(_imgshp[0],_imgshp[1]),cmap='Greys_r',interpolation='none')
        plt.clim(0.0, 1.0)
    plt.show()
"""
    Normalizer 
"""
class normalizer(object):
    def __init__(self,_name=''):
        self.name = _name
        print ("[%s] Instantiated" % (self.name))
    
"""
    Generate complex data for tesing mixture density network 
"""
def get_data4mdn(_xmin=-5.,_xmax=5.,_bias=0,_nsample=1e3):
    XMIN,XMAX = _xmin,_xmax
    NSAMPLE   = _nsample
    _x1 = np.float32(np.random.uniform(XMIN,XMAX,((int)(NSAMPLE/2),1)))
    _r1 = np.array([np.random.normal(scale=np.abs(i)) for i in _x1])
    _y1 = np.float32((_x1**2)+_r1*1.0) 
    _z1 = np.float32((_x1**2)-_bias+_r1*1.0) 
    _x2 = np.float32(np.random.uniform(XMIN,XMAX,((int)(NSAMPLE/2),1)))
    _r2 = np.array([np.random.normal(scale=np.abs(i)) for i in _x2])
    _y2 = np.float32(-(_x2**2)+_r2*1.0)
    _z2 = np.float32(-(_x2**2)+_bias+_r2*1.0)
    _xdata = np.concatenate((_x1,_x2),axis=0)
    _ydata = np.concatenate((_y1,_y2),axis=0)
    _zdata = np.concatenate((_z1,_z2),axis=0)
    return _xdata,_ydata,_zdata
    



        
        
        
        
        
        
        
        
        
        
        
        
        
        
       
