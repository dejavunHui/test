# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:24:43 2017

@author: lenovo
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
mnist=input_data.read_data_sets('/tem/data',dtype=tf.uint8,one_hot=True)
images=mnist.train.images
labels=mnist.train.labels
pixels=images.shape[1]
num_examples=mnist.train.num_examples
filename='C:/learnpython/TFRecord/output.tfrecords'
writer=tf.python_io.TFRecordWriter(filename)
for index in range(num_examples):
    image_raw=images[index].tostring()
    example=tf.train.Example(features=tf.train.Features(feature={
            'pixels':_int64_feature(pixels),
            'label':_int64_feature(np.argmax(labels[index])),
            'image_raw':_bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
writer.close()

#读取TFRecord文件样例
reader=tf.TFRecordReader()
#创建一个队列来维护输入文件列表
filename_queue=tf.train.string_input_producer(['C:/learnpython/TFRecord/output.tfrecords'])
#从文件中读取一个样例，也可以使用read_up_to函数一次性读取多个样例
_,serialized_examples=reader.read(filename_queue)
#解析读入的一个样例，如果需要解析多个样例，可以用parse_example函数
features=tf.parse_single_example(
                                 serialized_examples,
                                 features={
                                          'image_raw':tf.FixedLenSequenceFeature([],tf.string),
                                        'pixels':tf.FixedLenSequenceFeature([],tf.int64),
                                        'label':tf.FixedLenSequenceFeature([],tf.int64)})
#tf.decode_raw()可以将字符串解析成图像对应的像素数组
images=tf.decode_raw(features['image_raw'],tf.uint8)
labels=tf.cast(features['label'],tf.int32)
pixels=tf.cast(features['pixels'],tf.int32)
with tf.Session() as s:
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=s,coord=coord)
    for i in range(10):
        image,label,pixels=s.run(images,labels,pixels)