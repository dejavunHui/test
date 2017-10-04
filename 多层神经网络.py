# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 16:54:52 2017

@author: lenovo
"""

import tensorflow as tf
from numpy.random import RandomState

#进行权重的l2正则化
def get_weights(shape,l2):
    var=tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    tf.add_to_collection(
                         'losses',tf.contrib.layers.l2_regularizer(l2)(var))
    return var
    
x=tf.placeholder(tf.float32,shape=(None,2))
y_=tf.placeholder(tf.float32,shape=(None,1))
batch_size=8
#每层神经网络的神经结点个数
layer_dimension=[2,10,10,10,1]
#神经网络层数
n_layers=len(layer_dimension)
#前向传播的最大深度的神经结点
cur_layer=x
#当前层神经节点个数
in_dimension=layer_dimension[0]
#搭建神经网络
for i in range(1,n_layers):
    out_dimension=layer_dimension[1]#下一层神经结点的个数
    weight=get_weights([in_dimension,out_dimension],0.001)
    bias=tf.Variable(tf.constant(0.1,shape=[out_dimension]))
    cur_layer=tf.nn.relu(tf.matmul(cur_layer,weight)+bias)
    in_dimension=layer_dimension[i]

mse_loss=tf.reduce_mean(tf.square(y_-cur_layer))
tf.add_to_collection('losser',mse_loss)
loss=tf.add_n(tf.get_collection('losses'))

global_step=tf.Variable(0)
learning_rate=tf.train.exponential_decay(
                                         0.3,global_step,100,0.096,staircase=True)

train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

dataset_size=128
rdm=RandomState(1)
X=rdm.rand(dataset_size,2)
Y=[[int(x1+x2<1)] for (x1,x2)in X]
with tf.Session() as s:
    s.run(tf.initialize_all_variables())
    steps=5000
    for i in range(steps):
        start=(i*batch_size)%dataset_size
        end=min(start+batch_size,dataset_size)
        s.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i%1000==0:
            now_loss=s.run(loss,feed_dict={x:X,y_:Y})
            print("经过%s次训练，损失函数为:%g"%(i,now_loss))
    print(weight)
    