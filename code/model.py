import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.keras as keras
import numpy as np
import matplotlib.pyplot as plt

"""
    Basic Convolutional Neural Network with tf.slim
"""
class basic_cnn(object):
    def __init__(self,_name='CNN',
                 _xtrain=np.zeros((100,1)),_ytrain=np.zeros((100,1)),
                 _xtest=np.zeros((100,1)),_ytest=np.zeros((100,1)),
                 _xval=np.zeros((100,1)),_yval=np.zeros((100,1)),
                 _xshp=np.array([28,28]),_nclass=10,_labels=None,
                 _sess=None,_batch_size=64,_lr=1e-3):
        self.name   = _name
        self.xtrain,self.ytrain = _xtrain,_ytrain
        self.xtest,self.ytest = _xtest,_ytest
        self.xval,self.yval = _xval,_yval
        self.xshp   = _xshp
        self.nclass = _nclass
        self.labels = _labels
        self.xdim   = self.xshp[0]*self.xshp[1]
        self.sess   = _sess
        self.ntotal = self.xtrain.shape[0]
        self.batch_size = _batch_size
        self.lr     = _lr
        """ Build Model """ 
        self.build_model()
        """ Build Graph """ 
        self.build_graph()
        """ Initialize """
        self.init_weight()
        print ("[%s] Instantiated" % (self.name))
        print (" Input size:[%s] #class:[%d]" % (self.xshp,self.nclass))
        print ("Trainable Variables")
        for i in range(len(self.t_vars)):
            w_name  = self.t_vars[i].name
            w_shape = self.t_vars[i].get_shape().as_list()
            print (" [%d] Name:[%s] Shape:[%s]" % (i,w_name,w_shape))
        print ("Global Variables")
        for i in range(len(self.g_vars)):
            w_name  = self.g_vars[i].name
            w_shape = self.g_vars[i].get_shape().as_list()
            print (" [%d] Name:[%s] Shape:[%s]" % (i,w_name,w_shape))
    """
        Construct CNN
    """
    def build_model(self):
        self.x = tf.placeholder(dtype=tf.float32,shape=[None,self.xdim])
        self.y = tf.placeholder(dtype=tf.float32,shape=[None,self.nclass])
        self.is_training = tf.placeholder(dtype=tf.bool,shape=[])
        self.x_rshp = tf.reshape(self.x, [-1,self.xshp[0],self.xshp[1],1])
        bnparam = {'is_training':self.is_training,'decay':0.9,'updates_collections':None}
        self.winit = tf.truncated_normal_initializer(stddev=0.01)
        with tf.variable_scope('W') as scope:
            self.net = slim.conv2d(self.x_rshp,64,[3,3],padding='SAME',
                                   activation_fn=tf.nn.relu,
                                   weights_initializer=self.winit,
                                   normalizer_fn=slim.batch_norm,
                                   normalizer_params=bnparam,
                                   scope='conv1')
            self.net = slim.max_pool2d(self.net, [2, 2], scope='pool1')
            self.net = slim.conv2d(self.net,128,[3,3],padding='SAME',
                                   activation_fn=tf.nn.relu,
                                   weights_initializer=self.winit,
                                   normalizer_fn=slim.batch_norm,
                                   normalizer_params=bnparam,
                                   scope='conv2')
            self.net = slim.max_pool2d(self.net, [2, 2], scope='pool2')
            self.net = slim.conv2d(self.net,256,[3,3],padding='SAME',
                                   activation_fn=tf.nn.relu,
                                   weights_initializer=self.winit,
                                   normalizer_fn=slim.batch_norm,
                                   normalizer_params=bnparam,
                                   scope='conv3')
            self.net = slim.max_pool2d(self.net, [2, 2], scope='pool3')
            self.net = slim.flatten(self.net, scope='flatten3')
            self.net = slim.fully_connected(self.net, 1024,
                            activation_fn       = tf.nn.relu,
                            weights_initializer = self.winit,
                            normalizer_fn       = slim.batch_norm,
                            normalizer_params   = bnparam,
                            scope='fc3')
            self.net = slim.dropout(self.net, keep_prob=0.7,
                            is_training=self.is_training, scope='dropout4')  
            self.out = slim.fully_connected(self.net,self.nclass, 
                            activation_fn=None, normalizer_fn=None, scope='out5')
        # GET VARIABLES
        t_vars = tf.trainable_variables()
        self.t_vars = [var for var in t_vars if 'W/' in var.name]
        g_vars = tf.global_variables()
        self.g_vars = [var for var in g_vars if 'W/' in var.name]
    """
        Build Graph
    """
    def build_graph(self):
        self.loss = tf.losses.softmax_cross_entropy(self.y,self.out)
        self.corr = tf.equal(tf.argmax(self.y,1),tf.argmax(self.out,1))
        self.accr = tf.reduce_mean(tf.cast(self.corr,tf.float32))
        self.step = tf.Variable(0)
        self.lr   = tf.train.exponential_decay(
                        learning_rate=self.lr,  
                        global_step=self.step*self.batch_size, 
                        decay_steps=self.ntotal,       
                        decay_rate=0.95,              
                        staircase=False) 
        # LR = learning_rate*decay_rate^(global_step/decay_steps)
        self.optm = tf.train.AdamOptimizer(self.lr).minimize(self.loss,global_step=self.step)
    """
        Initialize Weight
    """
    def init_weight(self):
        self.sess.run(self.step.assign(0))
        self.sess.run(tf.global_variables_initializer())
        self.step_val = self.sess.run(self.step)
    
    """
        Update
    """
    def update(self):
        """ Mini-batch training """
        self.step_val = self.sess.run(self.step)
        offset = (self.step_val*self.batch_size)%(self.ntotal)
        batch_x = self.xtrain[offset:(offset+self.batch_size),:]
        batch_y = self.ytrain[offset:(offset+self.batch_size),:]
        oper_update=[self.optm,self.loss,self.accr]
        feed_update={self.x:batch_x,self.y:batch_y,self.is_training:True}
        _,self.loss_val,self.train_accr=self.sess.run(oper_update,feed_dict=feed_update)
    """
        Check Data
    """
    def check_data(self,_xdata,_ydata,_nunit=128,_shuffle=True):
        _ndata = _xdata.shape[0]
        _niter = (_ndata // _nunit) + 1
        _loss_total,_corr_total = 0.0,0.0
        missidxlist = []
        for _iter in range(_niter):
            _xbatch = _xdata[_iter*_nunit:(_iter+1)*_nunit,:]
            _ybatch = _ydata[_iter*_nunit:(_iter+1)*_nunit,:]
            oper_check  = [self.loss,self.corr]
            feed_check  = {self.x:_xbatch,self.y:_ybatch,self.is_training:False}
            _loss,_corr = self.sess.run(oper_check,feed_dict=feed_check)
            _loss_total = _loss_total+_loss
            _corr_total = _corr_total+np.sum(_corr)
            """ Find misclassified images """
            missidx = _iter*_nunit+np.where(_corr==False)[0]
            missidxlist = np.concatenate((missidxlist,missidx))
        _loss_avg = _loss_total/float(_niter)
        _accr_avg = _corr_total/float(_ndata)
        """ Random shuffle and cast to int """
        if _shuffle:
            np.random.shuffle(missidxlist)
        missidxlist = missidxlist.astype(int)
        return _loss_avg,_accr_avg,missidxlist
    """
        Get Label
    """
    def get_label(self,_xdata,_nunit=128):
        _nunit = 256
        _ndata = _xdata.shape[0]
        _niter = (_ndata // _nunit) + 1
        predlist = []
        for _iter in range(_niter):
            _xbatch = _xdata[(int)(_iter*_nunit):(int)(_iter+1)*_nunit,:]
            _out = self.sess.run(self.out,feed_dict={self.x:_xbatch,self.is_training:False})
            predlist = np.concatenate((predlist,np.argmax(_out,axis=1)))
            predlist = predlist.astype(int)
        predlabel = self.labels[predlist]
        return predlabel,predlist
    """
        Print Current Status
    """
    def print_status(self):
        test_loss,test_accr,_   = self.check_data(_xdata=self.xtest,_ydata=self.ytest)
        train_loss,train_accr,_ = self.check_data(_xdata=self.xtrain,_ydata=self.ytrain)
        print ("[iter:%04d]TrainLoss:[%.2e],TrainAccr:[%.2f%%],TestAccr:[%.2f%%]"%
               (self.step_val,train_loss,train_accr*100.,test_accr*100.))
    """
        Plot Misclassified Images
    """
    def plot_misclassified(self,_xdata=None,_ydata=None,_nplot=5):
        if _xdata==None: 
            _xdata = self.xtest
            _ydata = self.ytest
        _,_,test_missidx = self.check_data(_xdata=_xdata,_ydata=_ydata)
        selmissidx = test_missidx[:min(_nplot,len(test_missidx))].astype(int)
        f,axarr = plt.subplots(1,_nplot,figsize=(18,8))
        for idx,imgidx in enumerate(selmissidx):
            currimg=np.reshape(_xdata[imgidx:imgidx+1,:],self.xshp)
            truelabel=self.labels[np.argmax(_ydata[imgidx:imgidx+1,:])]
            predlabel,predlist=self.get_label(_xdata=_xdata[imgidx:imgidx+1,:])
            axarr[idx].imshow(currimg,cmap=plt.get_cmap('gray'))
            axarr[idx].set_title('[%d]T:%s(P:%s)'%(imgidx,truelabel,predlabel[0]),fontsize=15)
        plt.show()
    """
        Save
    """
    def save(self,_savename=None):
        """ Save name """
        if _savename==None:
            _savename='net/net_cnn.npz'
        """ Get global variables """
        g_wnames,g_wvals = [],[]
        for i in range(len(self.g_vars)):
            curr_wname = self.g_vars[i].name
            curr_wvar  = [v for v in tf.global_variables() if v.name==curr_wname][0]
            curr_wval  = self.sess.run(curr_wvar)
            g_wnames.append(curr_wname)
            g_wvals.append(curr_wval)
        """ Save """
        np.savez(_savename,g_wnames=g_wnames,g_wvals=g_wvals)
        print ("[%s] Saved. Size is [%.1f]MB" % 
               (_savename,os.path.getsize(_savename)/1000./1000.))
    """
        Restore
    """
    def restore(self,_loadname=None):
        if _loadname==None:
            _loadname='net/net_cnn.npz'
        l = np.load(_loadname)
        g_wnames = l['g_wnames']
        g_wvals  = l['g_wvals']
        for widx,wname in enumerate(g_wnames):
            curr_wvar  = [v for v in tf.global_variables() if v.name==wname][0]
            self.sess.run(tf.assign(curr_wvar,g_wvals[widx]))
        print ("Weight restored from [%s]" % (_loadname))


        
        
        
""" 
    Basic DCVAE
"""
def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1. + leak)
        f2 = 0.5 * (1. - leak)
    return f1 * x + f2 * tf.abs(x)
class basic_dcvae(object):
    def __init__(self,_name='DCVAE',_xdim=np.array([28,28,1]),_zdim=128,
                 _filtersize=[5,5],_nfilter=32,_actv=lrelu,
                 _lr=0.001,_beta1=0.9,_beta2=0.999,_epsilon=1e-8):
        """ Parse inputs """
        self.name = _name
        if len(_xdim)==2: _xdim=np.concatenate([_xdim,np.array([1])])
        self.xdim       = _xdim
        self.xdim_flat  = np.prod(self.xdim)
        self.zdim       = _zdim
        self.filtersize = _filtersize
        self.nfilter    = _nfilter
        self.actv       = _actv
        self.lr         = _lr
        self.beta1      = _beta1
        self.beta2      = _beta2
        self.epsilon    = _epsilon
        """ Build model """
        self.build_model()
        """ Build graph """
        self.build_graph()
        print ("[%s] Instantiated" % (self.name))
        print (" xdim:[%s] zdim:[%d] nfilter:[%d] filtersize:[%s]"%
               (self.xdim,self.zdim,self.nfilter,self.filtersize))
        """ Print """
        print ("Trainable Variables")
        for i in range(len(self.t_vars)):
            w_name  = self.t_vars[i].name
            w_shape = self.t_vars[i].get_shape().as_list()
            print (" [%d] Name:[%s] Shape:[%s]" % (i,w_name,w_shape))
        print ("Global Variables")
        for i in range(len(self.g_vars)):
            w_name  = self.g_vars[i].name
            w_shape = self.g_vars[i].get_shape().as_list()
            print (" [%d] Name:[%s] Shape:[%s]" % (i,w_name,w_shape))
            
    """
        Encoder
    """
    def encoder(self,_x,_name='encoder',_reuse=False):
        with tf.variable_scope(_name, reuse=_reuse) as scope:
            with slim.arg_scope([slim.conv2d], stride=2, activation_fn=self.actv,
                                weights_initializer=self.conv_init,biases_initializer=self.bias_init,
                                normalizer_fn=slim.batch_norm,normalizer_params=self.bn_params):
                _net = slim.conv2d(_x,self.nfilter*1,self.filtersize,
                                       normalizer_fn=None,normalizer_params=None,scope='conv0')
                _net = slim.conv2d(_net,self.nfilter*2,self.filtersize,scope='conv1') 
                _net = slim.conv2d(_net,self.nfilter*4,self.filtersize,scope='conv2')
                _net = slim.conv2d(_net,self.nfilter*8,self.filtersize,scope='conv3')
                _zmu = slim.fully_connected(slim.flatten(_net),self.zdim,
                                            weights_initializer=self.fully_init,
                                            biases_initializer=self.bias_init,
                                            activation_fn=None,
                                            scope='zmu')
                _zlogvar = slim.fully_connected(slim.flatten(_net),self.zdim,
                                                weights_initializer=self.fully_init,
                                                biases_initializer=self.bias_init,
                                                activation_fn=None,
                                                scope='zlogvar')
        return _zmu, _zlogvar
    """
        Decoder
    """
    def decoder(self,_z,_name='decoder',_reuse=False):
        s16_0,s16_1 = int(self.xdim[0]/16),int(self.xdim[1]/16)
        with tf.variable_scope(_name, reuse=_reuse) as scope:
            with slim.arg_scope([slim.conv2d_transpose],stride=2,activation_fn=self.actv,
                                weights_initializer=self.conv_init, biases_initializer=self.bias_init,
                                normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params):
                _net = slim.fully_connected(_z,self.nfilter*8*s16_0*s16_1, activation_fn=None,
                                       weights_initializer=self.fully_init,
                                       biases_initializer=self.bias_init,
                                       scope='lin0')
                _net = tf.reshape(_net, [-1,s16_0,s16_1,self.nfilter*8]) 
                _net = slim.batch_norm(_net, activation_fn=tf.nn.relu, scope='lin0_bn', **self.bn_params)
                _net = slim.conv2d_transpose(_net,self.nfilter*4,self.filtersize,scope='up1')
                _net = slim.conv2d_transpose(_net,self.nfilter*2,self.filtersize,scope='up2')
                _net = slim.conv2d_transpose(_net,self.nfilter*1,self.filtersize,scope='up3')
                _net = slim.conv2d_transpose(_net,self.xdim[2],self.filtersize,
                                             activation_fn=None,
                                             scope='up4')
        return _net
    """
        Build Model
    """
    def build_model(self):
        self.x = tf.placeholder(tf.float32,shape=[None,self.xdim_flat])
        self.z = tf.placeholder(tf.float32,shape=[None,self.zdim])
        self.is_training = tf.placeholder(dtype=tf.bool,shape=[])
        self.kl_weight   = tf.placeholder(dtype=tf.float32,shape=[])
        self.conv_init   = tf.truncated_normal_initializer(stddev=0.02)
        self.fully_init  = tf.random_normal_initializer(stddev=0.02)
        self.bias_init   = tf.constant_initializer(0.)
        self.bn_init     = {'beta': tf.constant_initializer(0.),
                           'gamma': tf.random_normal_initializer(1., 0.02)}
        self.bn_params   = {'is_training':self.is_training,'decay':0.9,'epsilon':1e-5,
                           'param_initializers':self.bn_init,'updates_collections': None}
        """
            Reshape to image shape
        """
        self.x_reshape = tf.reshape(self.x,shape=[-1,self.xdim[0],self.xdim[1],self.xdim[2]])
        """
            Encoder
        """
        self.zmu,self.zlogvar = self.encoder(self.x_reshape,_name='encoder',_reuse=False)
        """
            Sample z
        """
        self.eps     = tf.random_normal(shape=tf.shape(self.zmu),mean=0,stddev=1,dtype=tf.float32)
        self.zsample = self.zmu + tf.sqrt(tf.exp(self.zlogvar/2.))*self.eps
        """
            Decoder with sampled z (for training)
        """
        self.xrecon_train = self.decoder(self.zsample,_name='decoder',_reuse=False)
        """
            Decoder with given z (for testing)
        """
        self.xrecon_test = self.decoder(self.z,_name='decoder',_reuse=True)
        """ Get Variables """
        _t_vars = tf.trainable_variables()
        self.t_vars = [var for var in _t_vars if ('encoder/' in var.name) or ('decoder/' in var.name)]
        _g_vars = tf.global_variables()
        self.g_vars = [var for var in _g_vars if ('encoder/' in var.name) or ('decoder/' in var.name)]
    """
        Build Graph
    """
    def build_graph(self):
        self.l1_loss     = 1./2.*tf.norm(self.xrecon_train-self.x_reshape,ord=1,axis=1)
        self.l2_loss     = 1./2.*tf.norm(self.xrecon_train-self.x_reshape,ord=2,axis=1)
        self.recon_loss  = tf.reduce_mean(self.l1_loss)
        self.kl_loss_raw = 0.5*tf.reduce_sum(tf.exp(self.zlogvar)+self.zmu**2-1.-self.zlogvar,1)
        self.kl_loss     = tf.reduce_mean(self.kl_loss_raw)
        self.total_loss  = self.recon_loss + self.kl_weight*self.kl_loss
        self.solver = tf.train.AdamOptimizer(
            learning_rate=self.lr,beta1=self.beta1,beta2=self.beta2,epsilon=self.epsilon)\
            .minimize(self.total_loss)
    """
        Save
    """
    def save(self,_sess,_savename=None):
        """ Save name """
        if _savename==None:
            _savename='net/net_dcvae.npz'
        """ Get global variables """
        g_wnames,g_wvals = [],[]
        for i in range(len(self.g_vars)):
            curr_wname = self.g_vars[i].name
            curr_wvar  = [v for v in tf.global_variables() if v.name==curr_wname][0]
            curr_wval  = _sess.run(curr_wvar)
            g_wnames.append(curr_wname)
            g_wvals.append(curr_wval)
        """ Save """
        np.savez(_savename,g_wnames=g_wnames,g_wvals=g_wvals)
        print ("[%s] Saved. Size is [%.1f]MB" % 
               (_savename,os.path.getsize(_savename)/1000./1000.))
    """
        Restore
    """
    def restore(self,_sess,_loadname=None):
        if _loadname==None:
            _loadname='net/net_dcvae.npz'
        l = np.load(_loadname)
        g_wnames = l['g_wnames']
        g_wvals  = l['g_wvals']
        for widx,wname in enumerate(g_wnames):
            curr_wvar  = [v for v in tf.global_variables() if v.name==wname][0]
            _sess.run(tf.assign(curr_wvar,g_wvals[widx]))
        print ("Weight restored from [%s]" % (_loadname))

    
        














































# EOF