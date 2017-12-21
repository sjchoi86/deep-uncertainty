import tensorflow as tf
import tensorflow.contrib.slim as slim

tfrni = tf.random_normal_initializer
tftni = tf.truncated_normal_initializer
tfci  = tf.constant_initializer

def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1. + leak)
        f2 = 0.5 * (1. - leak)
    return f1 * x + f2 * tf.abs(x)

def batch_norm(_x,_name="batch_norm"):
    return tf.contrib.layers.batch_norm(_x,decay=0.9,updates_collections=None,epsilon=1e-5,scale=True,scope=_name)

def instance_norm(_input,_name="instance_norm"):
    with tf.variable_scope(_name):
        depth = _input.get_shape()[3]
        scale = tf.get_variable("scale",[depth],initializer=tfrni(1.0,0.02,dtype=tf.float32))
        offset = tf.get_variable("offset",[depth],initializer=tfci(0.0))
        mean,variance = tf.nn.moments(_input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (_input-mean)*inv
        return scale*normalized + offset
    
def conv2d(_input,_output_dim,_ks=4,_s=2,_stddev=0.02,_padding='SAME',_name="conv2d",_actv=None):
    with tf.variable_scope(_name):
        return slim.conv2d(_input,_output_dim,_ks,_s, padding=_padding,activation_fn=_actv,
                            weights_initializer=tftni(stddev=_stddev),
                            biases_initializer=None)
    
def deconv2d(_input,_output_dim,_ks=4,_s=2,_stddev=0.02,_name="deconv2d",_actv=None):
    with tf.variable_scope(_name):
        return slim.conv2d_transpose(_input,_output_dim,_ks,_s,padding='SAME',activation_fn=_actv,
                                    weights_initializer=tftni(stddev=_stddev),
                                    biases_initializer=None)
    
def linear(_input,_output_dim,_name='Linear',_stddev=0.02,_bias=0.0, with_w=False):
    with tf.variable_scope(_name):
        _matrix = tf.get_variable("Matrix", [_input.get_shape()[-1],_output_dim],tf.float32,
                                 tfrni(stddev=_stddev))
        _bias = tf.get_variable("bias",[_output_dim],initializer=tfci(_bias))
        if with_w:
            return tf.matmul(_input,_matrix)+_bias, _matrix, _bias
        else:
            return tf.matmul(_input,_matrix)+_bias
        
        