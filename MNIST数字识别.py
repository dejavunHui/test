# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:27:54 2017

@author: lenovo
"""

#MNIST手写体数字识别
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

#获取训练数据及预测数据
#mnist=input_data.read_data_sets("/path/to/MNIST_data/",one_hot=True)
#print("Training data size:",mnist.train.num_examples)
#print("Validating data size:",mnist.validation.num_examples)
#print("Testing data size:",mnist.test.num_examples)
#print("Example training data:",mnist.train.images[0])
#print("Example training data lable:",mnist.train.labels[0])
#batch_size=100
#xs,ys=mnist.train.next_batch(batch_size)
#print("X shape:",xs.shape)
#print("Y shape:",ys.shape)

class MNIST_model(object):
    def __init__(self,input_node=784,output_node=10,layer_node=500,\
                 batch_size=100,learning_rate_base=0.8,learning_rate_decay=0.99,\
                 regularization=0.0001,training_steps=30000,moving_average_decay=0.99):
        self.input_node=input_node
        self.output_node=output_node
        self.layer_node=layer_node
        self.batch_size=batch_size
        self.learning_rate_base=learning_rate_base
        self.learning_rate_decay=learning_rate_decay
        self.regularization=regularization
        self.training_step=training_steps
        self.moving_average_decay=moving_average_decay
        pass
    def inference(self,input_tensor,avg_class,weights1,biases1,weights2,biases2):
        if avg_class==None:
            layer1=tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
            return tf.matmul(layer1,weights2)+biases2
        else:
            layer1=tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1))\
                              +avg_class.average(biases1))
            return tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2)
        pass
    def fit(self,mnist):
        x=tf.placeholder(tf.float32,[None,self.input_node],name='x-input')
        y_=tf.placeholder(tf.float32,shape=(None,self.output_node),name='y-input')
        #生成隐藏层的参数
        weights1=tf.Variable(tf.truncated_normal([self.input_node,self.layer_node],stddev=1))
        biases1=tf.Variable(tf.constant(0.1,shape=[self.layer_node]))
        #生成输出层的参数
        weights2=tf.Variable(tf.truncated_normal([self.layer_node,self.output_node],stddev=1))
        biases2=tf.Variable(tf.constant(0.1,shape=[self.output_node]))
        y=self.inference(x,None,weights1,biases1,weights2,biases2)
        global_step=tf.Variable(0,trainable=False)
        variable_average=tf.train.ExponentialMovingAverage(self.moving_average_decay,global_step)
        variable_average_op=variable_average.apply(tf.trainable_variables())
        average_y=self.inference(x,variable_average,weights1,biases1,weights2,biases2)
        cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
        cross_entropy_mean=tf.reduce_mean(cross_entropy)
        regularizer=tf.contrib.layers.l2_regularizer(self.regularization)
        regularization=regularizer(weights1)+regularizer(weights2)
        loss=cross_entropy_mean+regularization
        learning_rate=tf.train.exponential_decay(self.learning_rate_base,global_step,\
                                                 mnist.train.num_examples/self.batch_size,\
                                                 self.learning_rate_decay)
        train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
        #tf.control_dependencies和tf.group机制，用来进行多次的参数更新以及滑动平均值的更新
        #下面两行程序等价于train_op=tf.group(train_step,variables_averages_op)
        with tf.control_dependencies([train_step,variable_average_op]):
            train_op=tf.no_op(name='train')
            
        correct_prediction=tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        with tf.Session() as s:
            s.run(tf.initialize_all_variables())
            validate_feed={x:mnist.validation.images,
                           y_:mnist.validation.labels}
            test_feed={x:mnist.test.images,
                       y_:mnist.test.labels}
            for i in range(self.training_step):
                if i%1000==0:
                    validate_acc=s.run(accuracy,feed_dict=validate_feed)
                    print("After %s training step(s),validation accuracy"
                          "using average model is %g "%(i,validate_acc))
                xs,ys=mnist.train.next_batch(self.batch_size)
                s.run(train_op,feed_dict={x:xs,y_:ys})
            test_acc=s.run(accuracy,feed_dict=test_feed)
            print("After %d training step(s),test accuracy using average"
                  "model is %g "%(self.training_step,test_acc))
            pass
        pass
    def main(self,argv=None):
        mnist=input_data.read_data_sets("/path/to/MNIST_data/",one_hot=True)
        self.fit(mnist)
        
mnist=MNIST_model()
mnist.main()