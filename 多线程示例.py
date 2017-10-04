# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 17:01:38 2017

@author: lenovo
"""

import tensorflow as tf
import numpy as np
import threading
import time

def MyLoop(coord,worker_id):
    while not coord.should_stop():
        if np.random.rand()<0.1:
            print("Stoping from id:%d\n"%worker_id)
            #通知其他线程停止
            coord.request_stop()
        else:
            print("Working on id:%d\n"%worker_id)
            time.sleep(1)

#声明一个Coordinator类来协同多个线程
coord=tf.train.Coordinator()
threads=[
         threading.Thread(target=MyLoop,args=(coord,i,)) for i in range(10)]
for t in threads:t.start()
coord.join(threads)