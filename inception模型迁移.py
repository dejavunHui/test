# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:10:50 2017

@author: lenovo
"""

import tensorflow as tf
import glob
import os.path
import random
import numpy as np
from tensorflow.python.platform import gfile

#定义inception-v3模型瓶颈层的结点个数
bottleneck_tensor_size=2048
bottleneck_tensor_naame='pool_3/_reshape:0'
jpeg_data_tensor_name='DecodeJpeg/contents:0'
model_dir='C:/Users/lenovo/Downloads/inception_dec_2015'
model_file='tensorflow_inception_graph.pb'

#存放特征向量的文件的地址
cache_dir='C:/Users/lenovo/Downloads/bottleneck'
input_data='C:/Users/lenovo/Downloads/flower_photos'
#验证数据百分比
validation_percentage=10
#测试数据百分比
test_percentage=10
#定义神经网络的设置
learning_rate=0.01
steps=4000
batch=100
def create_image_lists(test_percentage,validation_percentage):
    result={}
    sub_dirs=[x[0] for x in os.walk(input_data)]
    is_root_dir=True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir=False
            continue
        extensions=['jpg','jpeg','JPG','JPEG']
        file_list=[]
        dir_name=os.path.basename(sub_dir)
        for extension in extensions:
            file_glob=os.path.join(input_data,dir_name,'*.'+extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue
        #获取类别名称
        label_name=dir_name.lower()
        training_images=[]
        testing_images=[]
        validation_images=[]
        for file_name in file_list:
            base_name=os.path.basename(file_name)
            chance=np.random.randint(100)
            if chance<validation_percentage:
                validation_images.append(base_name)
            elif chance<(test_percentage+validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
                
        result[label_name]={
            'dir':dir_name,
            'training':training_images,
            'validation':validation_images,
            'testing':testing_images,
            }
    return result
    
def get_image_path(image_lists,image_dir,label_name,index,category):
    #参数含义：所有图片信息，存放的根目录，类别名称，需要获取的图片的编号
    #，需要获取的图片所在集合（训练集，验证集，测试集）
    label_lists=image_lists[label_name]
    category_list=label_lists[category]
    mod_index=index%len(category_list)
    base_name=category_list[mod_index]
    sub_dir=label_lists['dir']
    final_path=os.path.join(image_dir,sub_dir,base_name)
    return final_path
    
#获取特征向量地址
def get_bottleneck_path(image_lists,label_name,index,category):
    return get_image_path(image_lists,cache_dir,label_name,index,category)+'.txt'

#加载inception模型训练图片，得到特征向量
def run_bottleneck_on_image(sess,image_data,image_data_tensor,bottleneck_tensor):
    bottleneck_values=sess.run(bottleneck_tensor,
                               {image_data_tensor:image_data})
    bottleneck_values=np.squeeze(bottleneck_values)
    return bottleneck_values
#获取处理后的特征，先试图寻找已经保存下来的向量，如果没有，
#就会计算这个向量然后保存进文件
def get_or_create_bottleneck(
                             sess,image_lists,label_name,index,
                             category,jpeg_data_tensor,bottleneck_tensor):
    #获取一张图片对应的特征向量文件的路径
    label_lists=image_lists[label_name]
    sub_dir=label_lists['dir']
    sub_dir_path=os.path.join(cache_dir,sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    bottleneck_path=get_bottleneck_path(image_lists,label_name,index,category)
    #特征向量文件不存在，则通过模型来计算向量并将向量保存进文件
    if not os.path.exists(bottleneck_path):
        image_path=get_image_path(
                                  image_lists,input_data,label_name,index,category)
        #获取图片内容
        image_data=gfile.FastGFile(image_path,'rb').read()
        #通过inception-v3模型计算特征向量
        bottleneck_values=run_bottleneck_on_image(sess,image_data,jpeg_data_tensor,bottleneck_tensor)
        #保存
        bottleneck_string=','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path,'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        #直接从文件中获取图片的特征向量
        with open(bottleneck_path,'r') as bottleneck_file:
            bottleneck_string=bottleneck_file.read()
        bottleneck_values=[float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values
    
#随机获取一个batch的图片作为训练数据
def get_random_cached_bottlenecks(
                                  sess,n_classes,image_lists,how_many,
                                  category,jpeg_data_tensor,bottleneck_tensor):
    bottlenecks=[]
    ground_truths=[]
    for _ in range(how_many):
        label_index=random.randrange(n_classes)
        label_name=list(image_lists.keys())[label_index]
        image_index=random.randrange(65536)
        bottleneck=get_or_create_bottleneck(sess,image_lists,label_name,image_index,
                                            category,jpeg_data_tensor,bottleneck_tensor)
        ground_truth=np.zeros(n_classes,dtype=np.float32)
        ground_truth=np.zeros(n_classes,dtype=np.float32)
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks,ground_truths
    
#获取全部的测试数据
def get_test_bottlenecks(sess,image_lists,n_classes,jpeg_data_tensor,bottleneck_tensor):
    bottlenecks=[]
    ground_truths=[]
    label_name_list=list(image_lists.keys())
    for label_index,label_name in enumerate(label_name_list):
        category='testing'
        for index ,unused_base_name in enumerate(
                                                 image_lists[label_name][category]):
            bottleneck=get_or_create_bottleneck(sess,image_lists,label_name,index,
                                                category,jpeg_data_tensor,bottleneck_tensor)
            ground_truth=np.zeros(n_classes,dtype=np.float32)
            ground_truth[label_index]=1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks,ground_truths
    
def main(_):
    #读取所有图片
    image_lists=create_image_lists(test_percentage,validation_percentage)
    n_classes=len(image_lists.keys())
    #读取模型
    with gfile.FastGFile(os.path.join(model_dir,model_file),'rb') as f:
        graph_def=tf.GraphDef()
        graph_def.ParseFromString(r.read())
    bottleneck_tensor,jpeg_data_tensor=tf.import_graph_def(
                                                           graph_def,
                                                           return_elements=[bottleneck_tensor_naame,jpeg_data_tensor_name])
    pass
