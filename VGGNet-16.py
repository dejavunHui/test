import tensorflow as tf
import math
import time
from datetime import datetime

def conv_op(input_tensor,name,kh,kw,n_out,dh,dw,p):
    n_in=input_tensor.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel=tf.get_variable(scope+'weigths',
        shape=[kh,kw,n_in,n_out],
        dtype=tf.float32,initializer=tf.contrib.layers.xavier_initalizer_conv2d())
        conv=tf.nn.conv2d(input_tensor,kernel,strides=[1,dh,dw,1],padding='SAME')
        bias_init_val=tf.constant(0.0,shape=[n_out],dtype=tf.float32)
        biases=tf.Variable(bias_init_val,name='biases')
        z=tf.nn.bias_add(conv,biases)
        activation=tf.nn.relu(z,name=scope)
        p+=[kernel,biases]
        return activation
def fc_op(input_tensor,name,n_out,p):
    n_in=input_tensor.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel=tf.get_variable(scope+'weights',
        shape=[n_in,n_out],dtype=tf.float32,initializer=tf.contrib.layer.xavier_initializer())
        bias=tf.constant(0.0,dtype=tf.float32,shape=[n_out])
        biases=tf.Variable(bias,name='biases')
        activation=tf.nn.relu(tf.matmul(input_tensor,kernel)+biases)
        p+=[kernel,biases]
        return activation
        
def mpool_op(input_tensor,name,kh,kw,dh,dw):
    return tf.nn.max_pool(input_tensor,ksize=[1,kh,kw,1],strides=[1,dh,dw,1],padding='SAME',name=name)

def inference_op(input_tensor,keep_prob):
    p=[]
    conv1_1=conv_op(input_tensor,'conv1_1',3,3,64,1,1,p)
    conv1_2=conv_op(conv1_1,'conv1_2',3,3,64,1,1,p)
    pool1=mpool_op(conv1_2,'pool1',2,2,2,2)
    conv2_1=conv_op(pool1,'conv2_1',3,3,128,1,1,p)
    conv2_2=conv_op(conv2_1,'conv2_2',3,3,128,1,1,p)
    pool2=mpool_op(conv2_2,'pool2',2,2,2,2)
    conv3_1=conv_op(pool2,'conv3_1',3,3,256,1,1,p)
    conv3_2=conv_op(conv3_1,'conv3_1',3,3,256,1,1,p)
    pool3=mpool_op(conv3_2,'pool3',2,2,2,2)
    conv4_1=conv_op(pool3,'conv4_1',3,3,512,1,1,p)
