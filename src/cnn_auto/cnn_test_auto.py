# coding:utf-8
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
from src.tool.data_prepare import *
from keras.callbacks import ModelCheckpoint
from keras import callbacks
from src.tool.models import *
from src.tool.models import *
from src.tool.find_all_hdf5 import *
from keras.callbacks import EarlyStopping
from src.tool.tool import *
from keras.utils import multi_gpu_model
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau
import pandas as pd
import xlwt
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

#进行配置，每个GPU使用60%上限现存
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3" # 使用编号为1，2号的GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7 # 每个GPU现存上届控制在60%以内
session = tf.Session(config=config)

# 设置session
KTF.set_session(session )

def test(data_path,generate_test, model_path):
    test_step = get_steps_per_epoch(data_path+'test/')

    #用于训练
    epochs = 200
    validation_steps=get_steps_per_epoch(data_path+'validation/')
    # example  all= 22011
    #batch size = 16 batch size 在tool/data_prepare下
    #steps_per_epoch = all / batch size=22011/16=1376
    steps_per_epoch=get_steps_per_epoch(data_path+'train/')
    '''-------------------------------------------------------------------------------------
       -first step: read image from $data_path ,and trian <<<------>>>lable,                    -
       -and preduce train_datas,train_lables,val_datas,val_lables,test_datas,test_lables   -
       - train : val:test == 6:2:2                                                         -
       -------------------------------------------------------------------------------------
    '''


    '''-------------------------------------------------------------------------------------
       -second step: biuld a model for yourself,we can choose:                             - 
       - --model: ResNet50,InceptionV3,VGG16,VGG19,Xception,InceptionResNetV2,DenseNet201
       - 就这么跟你说吧，要把模型复制到4个GPU上进行训练，加快速度。
       -------------------------------------------------------------------------------------
    '''

    #my_spatial_model = load_model(model_path + '/'+str(train_id)+'/'+model_name+'_'+str(train_id)+'_weights.hdf5')  # 加载权重到模型中
    my_spatial_model = None
    my_spatial_model = load_model(model_path)
    '''
     对模型进行预测
    
    predict = my_spatial_model.predict_generator(generate_validation, steps=98, max_queue_size=50, workers=1,
                                                 use_multiprocessing=False, verbose=1)
    predict_label = np.argmax(predict, axis=1)
    true_label = generate_validation.classes
    print(predict_label, true_label)
    predict_label = predict_label[0:1548] ####这里需要从新设定内容的长度。
    #table = pd.crosstab(true_label, predict_label, rownames=['label'], colnames=['predict'])
    #print("打印预测矩阵")
    #print(predict)
    #print("打印交叉表")
    #print(table)
    '''
    '''
     评估模型。
    '''
    loss, accuracy = my_spatial_model.evaluate_generator(generate_test, steps=test_step,verbose=1) #98需要才能重新确定值的大小
    print("loss: ", loss, "accuracy", accuracy)
    return loss, accuracy

'''
class m_a_test:
    def __init__(self,model_name,model_test_dir):
        self.model_name = model_name
        self.model_test_dir = model_test_dir
'''
'''
#test函数开始测试整个的集合，按照集合的大小确定应该测试那些内容，其集合的大小和内容可以得到改变
def test_model(model_name,train_id_list):
    data_path = ''
    model_path = ''
  
    用于处理单个模型所生成的模型list
    :param model_name: 模型的名字，主要用于后期的标识
    :param model_dir_list: 训练后生成模型的list，按照这个list进行遍历
    :return: 暂无
  
    for train_id in train_id_list :
        test(model_name, data_path, model_path, train_id)

def test_model_all(model_list):
    for model in model_list :
        test_model(model,train_id_list)
'''
def test_main():
    # 创建一个Workbook对象，相当于创建了一个Excel文件
    book = xlwt.Workbook(encoding="utf-8", style_compression=0)
    # 创建一个sheet对象，一个sheet对象对应Excel文件中的一张表格。
    sheet = book.add_sheet('test01', cell_overwrite_ok=True)

    dir_path = 'D:/DeepTool/DeepTool/deal-data-whole5/train_procese'
    data_path = 'D:/DeepTool/DeepTool/deal-data-whole5-part/'
    loss = 0
    acuracy = 0
    # 数据函数已经成功完成
    #generate_train = generate_batch_data(data_path + "/train")
    #generate_validation = generate_batch_data(data_path + "/validation")
    generate_test = generate_batch_data(data_path + "/test")
    i = 0
    for model_path in all_files(dir_path, patterns='*.hdf5'):
        if i < 61 :
            pass
        elif i <= 90  :
            loss,acuracy = test(data_path,generate_test, model_path)
            print(model_path,loss,acuracy)
            sheet.write(i,0,model_path)
            sheet.write(i, 1, loss)
            sheet.write(i, 2, acuracy)
        if i > 90 :break
            #tf.reset_default_graph()
            #tf.contrib.memory_stats.MaxBytesInUse()
        i = i + 1
    book.save('D:/DeepTool/DeepTool/deal-data-whole5/README/test_model_in_train_procese_005.xls')
test_main()