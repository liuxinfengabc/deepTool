#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import  numpy as np
import sys
from pandas import DataFrame  # DataFrame通常来装二维的表格
import pandas as pd  # pandas是流行的做数据分析的包
import os
import shutil

global count
count=1

rootPath='../../test-data/0518'
testDataPath='../../test-data/0518/full-data'

genePath='../../deal-data-whole6/'  #生成路径
dict_label = {}#标签文件

def TraverseDic(filepath,seperate=False): #遍历filepath下所有文件，包括子目录
    global  count
    global dict_label
    files = os.listdir(filepath)
    filepath=filepath.replace("\\",'/')
    for fi in files:
        fi_d = os.path.join(filepath,fi)
        fi_d = fi_d.replace("\\", '/')
        if(fi.find("txt")!=-1):
            continue
        if os.path.isdir(fi_d):#如果是目录，则遍历底层
          TraverseDic(fi_d,seperate)
        else:  #针对文件进行整理
          key=filepath.split("/")
          test_name=key[len(key)-1]
          key=key[len(key)-1]+fi
          label=dict_label.get(key)

          if label == None:
              continue
          keyStat="train/nonkeyhole/"
          if label[0]=='1':
              keyStat = "train/keyhole/"
          currentPath = genePath + keyStat
          if seperate == True:
              currentPath=currentPath+test_name + "/"
          if os.path.exists(currentPath) == False:
              os.makedirs(currentPath)
          fileName = currentPath + test_name + "-L" + str(label[0]) + "-" + fi
          print(fileName)
          shutil.copyfile(fi_d,fileName)
          count = count + 1
    pass

def readLabel():
    global  dict_label
    for i in range(1, 18):
        fileName = rootPath+"/dataLabel/dataLabel" + str(i) + 'deep.txt'
        with open(fileName, 'r') as df:
            for line in df:
                # 如果这行是换行符就跳过，这里用'\n'的长度来找空行
                if line.count('\n') == len(line):
                    continue
                # 对每行清除前后空格（如果有的话），然后用"："分割
                for kv in [line.strip().split(' ')]:
                    # 按照键，把值写进去
                    key = kv[0].split('\\')
                    key = key[len(key) - 2] + key[len(key) - 1]
                    if(len(kv)<2):
                        print("label error:  "+ fileName+str(len(kv)))
                    else:
                        dict_label.setdefault(key, []).append(kv[1])
                        print(kv[1])

readLabel()
#True 每个实验单独生成folder
#False 所有实验生成到一个文件夹下
TraverseDic(testDataPath,False)