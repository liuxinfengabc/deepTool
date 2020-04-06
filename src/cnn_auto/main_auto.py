#coding:utf-8
import sys
import os
import tensorflow as tf
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2" #,1,2,3"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8 # 每个GPU现存上届控制在60%以内
session = tf.Session(config=config)
from src.cnn_auto.cnn_train_auto  import *
def main_auto():


      '''
      train_id  训练结果的存放地址
      'VGG16'---->10000
      'VGG19'---->20000
      'InceptionV3'---->30000
      'Xception'----->40000
      'InceptionResNetV2'----->50000
      'DenseNet201'----->60000
      'ResNet50'------->70000
       '''
      #models = [ 'VGG16', 'VGG19', 'InceptionV3', 'Xception', 'InceptionResNetV2', 'DenseNet201','ResNet50'] #所有的训练模型
      models = ['InceptionV3']
      train_id = 40000

      #优化器和损失函数
      # optimizers = ['sgd','adam','Adagrad','Adadelta','Adamax','Nadam','RMSprop']
      optimizers = ['Adagrad']
      # loss_fuction = ['binary_crossentropy']
      loss_fuction = ['categorical_crossentropy', 'binary_crossentropy']

      train_path =      '../../deal-data-whole2/train'                  #训练路径，需要清楚里面的数据数量，数据组成？

      validation_path=  "../../deal-data-whole2/validation"        # 2020.02.22.zfz修改：由于新训练的精度降低，重新改用whole2训练，生成process3
      #validation_path = "../../deal-data-whole5/validation"   # 验证路径，whole2 的数据图像不完整，用其进行验证具有泛化性。 by liuxf

      # train_process_path='../../deal-data-whole6/train_process1'   #训练结果存放的路径
      train_process_path = '../../deal-data-whole2/train_process2'  # 训练结果存放的路径

      nb_classes = 2
      batch_size = 64
      spatial_epochs = 50
      temporal_epochs = 10

      image_size = 224
      flag = True
      timesteps_TIM = 3
      tensorboard = 1
      #循环的层数是 model * optimazers * optimazers_参数变化范围 * loss_function
      for models_index in models:

          print(models_index)
          train_id_bak=train_id  #备份
          for optimzers_index in optimizers:

              print(optimzers_index)

              for loss_fuction_index in loss_fuction:

                  print(loss_fuction_index)

                  train(models_index,
                        optimzers_index,
                        loss_fuction_index,
                        train_path,
                        validation_path,
                        train_process_path,
                        nb_classes,
                        batch_size,
                        spatial_epochs,
                        temporal_epochs,
                        train_id,
                        image_size,
                        flag,
                        timesteps_TIM,
                        tensorboard)
                  train_id = train_id + 1
          train_id = train_id_bak + 10000
main_auto()




