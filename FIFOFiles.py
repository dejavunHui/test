# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 19:42:04 2017

@author: lenovo
"""

import tensorflow as tf
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    
num_shards=2
instances_per_shard=2
for i in range(num_shards):
    filename=("./data.tfrecords-%.5d-of-%.5d"%(i,num_shards))
    writer=tf.python_io.TFRecordWriter(filename)
    for j in range(instances_per_shard):
        example=tf.train.Example(features=tf.train.Features(feature={
                            'i':_int64_feature(i),
                            'j':_int64_feature(j)}))
        writer.write(example.SerializeToString())
    writer.close()
    
files=tf.train.match_filenames_once("./data.tf*")
filename_queue=tf.train.string_input_producer(files,shuffle=False,num_epochs=2)
reader=tf.TFRecordReader()
_,serialized_example=reader.read(filename_queue)
features=tf.parse_single_example(
                                 serialized_example,
                                 features={
                                           'i':tf.FixedLenFeature([],tf.int64),
                                            'j':tf.FixedLenFeature([],tf.int64)})
with tf.Session() as s:
    s.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
    print(s.run(files))
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=s,coord=coord)
    for i in range(6):
        print(s.run([features['i'],features['j']]))
    coord.request_stop()
    coord.join(threads)
