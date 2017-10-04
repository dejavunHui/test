# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 17:16:04 2017

@author: lenovo
"""

import tensorflow as tf

queue=tf.FIFOQueue(100,'float')
#定义入队操作
enqueue_op=queue.enqueue([tf.random_normal([1])])
#表示了需要启动5个线程
qr=tf.train.QueueRunner(queue,[enqueue_op]*5)
tf.train.add_queue_runner(qr)
out_tensor=queue.dequeue()
with tf.Session() as s:
    coord=tf.train.Coordinator()
    #使用tf.train.QueueRunner()是需要明确调用tf.train.start_queue_runners来启动
    #所有线程
    threads=tf.train.start_queue_runners(sess=s,coord=coord)
    for _ in range(10):print(s.run(out_tensor)[0])
    coord.request_stop()
    coord.join(threads)
    