# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 09:31:22 2017

@author: lenovo
"""
#滑动平均模型的代码实验

#tf.train.ExponentialMovingAverage(decay,update)
#shadow_variable=shadow_variable*decay+(1-decay)*variable
#decay=min{decay,(1+num_undate)/(10+num_undate)}
import tensorflow as tf

#定义计算滑动平均的变量，初始值为零，实数类型
v1=tf.Variable(0,dtype=tf.float32)
#神经网络的迭代轮数，动态控制神经网络的衰减率
step=tf.Variable(0,trainable=False)
#实例化一个滑动平均类
ema=tf.train.ExponentialMovingAverage(0.99,step)
#定义一个更新滑动平均的操作,每次执行操作，列表中的变量都会更新
mintain_averages_op=ema.apply([v1])
with tf.Session() as s:
    s.run(tf.initialize_all_variables())
    print(s.run([v1,ema.average(v1)]))
    #更新v1的值到5
    s.run(tf.assign(v1,5))
    s.run(mintain_averages_op)
    print(s.run([v1,ema.average(v1)]))
    #更新step到1000
    s.run(tf.assign(step,1000))
    s.run(tf.assign(v1,10))
    s.run(mintain_averages_op)
    print(s.run([v1,ema.average(v1)]))
    s.run(mintain_averages_op)
    print(s.run([v1,ema.average(v1)]))