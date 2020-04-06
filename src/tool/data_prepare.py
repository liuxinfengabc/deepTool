#coding:utf-8
from keras.preprocessing.image import ImageDataGenerator
import os
def generate_batch_data(path):
    datagen = ImageDataGenerator(
            rotation_range=40,                                #  旋转范围
            width_shift_range=0.2,                            #  宽度调整范围
            height_shift_range=0.2,                           #  高度调整范围
            rescale=1./255,                                   #  尺度调整范围
            shear_range=0.2,                                   #  弯曲调整范围
            zoom_range=0.2,                                   #  缩放调整范围
            horizontal_flip=True,                              #  水平调整范围
            #brightness_range=0.3,                             #  亮度调整范围
            featurewise_center=True,                           #  是否特征居中
            featurewise_std_normalization=True,                #  特征是否归一化
            zca_whitening=True)                                #  是否使用 ZCA白化
    #        fill_mode='nearest')                              #  填充模式(图片大小不够时)
    return datagen.flow_from_directory(
            path,
            target_size=(224, 224),
            color_mode='rgb',
            batch_size=32,
            shuffle=False)

fileNum = 0

def get_steps_per_epoch(path): # 每个epoch所需要的步数
    fileNum = len([file for dir in os.listdir(path) for file in os.listdir(path+'/'+dir)])  #文件数量
    batch_size = 32 #每个batch 16个文件
    steps_per_epoch=int(fileNum/batch_size)  #每个轮回需要的步数（批次数） ，例如  128imges,则一个轮回需要128/16=8batch
    return steps_per_epoch