# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 17:50:55 2017

@author: lenovo
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

input_node=784
output_node=10
batch_size=100
layer_node=500
learning_rate_base=0.3
learning_rate_decay=0.99
regularization_rate=0.0001#正则化系数
training_steps=30000#训练轮数
moving_averages_decay=0.99#滑动平均衰减率
#定义前向传播函数
def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
    if avg_class==None:
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
        return tf.matmul(layer1,weights2)+biases2
    else:
        layer1=tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1))+avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2)
        
#改写inference函数
def inference_(input_tensor,reuse=False):
    with tf.variable_scope('layer1',reuse=reuse):
        weights1=tf.get_variable('weights1',[input_node,layer_node],
                                 initializer=tf.truncated_normal_initializer(stddev=1.0))
        biases1=tf.get_variable('biases1',[layer_node],
                                initializer=tf.constant_initializer(0.1))
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
    with tf.variable_scope('layer2',reuse=reuse):
        weights2=tf.get_variable('weights2',[layer_node,output_node],
                                 initializer=tf.truncated_normal_initializer(stddev=1.0))
        biases2=tf.get_variable('biases2',[output_node],
                                initializer=tf.constant_initializer(0.1))
        layer2=tf.matmul(layer1,weights2)+biases2
    return layer2
        
def train(mnist):
    x=tf.placeholder(tf.float32,[None,input_node],name='x-input')
    y_=tf.placeholder(tf.float32,[None,output_node],name='y_input')
    #生成隐藏层参数
    weights1=tf.Variable(
                         tf.truncated_normal([input_node,layer_node],stddev=1))
    biases1=tf.Variable(tf.constant(0.1,shape=[layer_node]))
    #生成输出层的参数
    weights2=tf.Variable(
                         tf.truncated_normal([layer_node,output_node],stddev=1))
    biases2=tf.Variable(tf.constant(0.1,shape=[output_node]))
    y=inference(x,None,weights1,biases1,weights2,biases2)
    global_step=tf.Variable(0,trainable=False)
    variable_averages=tf.train.ExponentialMovingAverage(moving_averages_decay,global_step)
    variables_averages_op=variable_averages.apply(tf.trainable_variables())
    average_y=inference(x,variable_averages,weights1,biases1,weights2,biases2)
    #生成交叉熵
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=tf.argmax(y_,1))
    #计算当前batch的交叉熵平均值
    cross_entopy_mean=tf.reduce_mean(cross_entropy)
    #计算l2正则化损失函数
    regularizer=tf.contrib.layers.l2_regularizer(regularization_rate)
    regularization=regularizer(weights1)+regularizer(weights2)
    loss=regularization+cross_entopy_mean
    learning_rate=tf.train.exponential_decay(
                                             learning_rate_base,#基础学习率
                                             global_step,#当前迭代的轮数
                                             mnist.train.num_examples/batch_size,#过完所有训练次数所需要的迭代次数
                                             learning_rate_decay)#每执行上一次数迭代后学习率乘以此值
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    train_op=tf.group([train_step,variables_averages_op])
    correct_prediction=tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    with tf.Session() as s:
        s.run(tf.initialize_all_variables())
        validata_feed={x:mnist.validation.images,
                       y_:mnist.valodation.labels}
        test_feed=dict(x=mnist.test.images,
                       y=mnist.test.lables)
        for i in range(training_steps):
            if i%1000==0:
                validata_acc=s.run(accuracy,feed_dict=validata_feed)
                print("在%s次训练之后，验证集的正确率为:%g"%(i,validata_acc))
            xs,ys=mnist.train.next_batch(batch_size)
            s.run(train_op,feed_dict={x:xs,y_:ys})
            test_acc=s.run(accuracy,feed_dict=test_feed)
            print("在%s次训练之后，验证集的正确率为:%g"%(i,test_acc))
            pass
        pass
    pass
def main(argv=None):
    mnist=input_data.read_data_sets('/temp/data',one_hot=True)
    train( mnist)
    
if __name__=='__main__':
    tf.app.run()
    