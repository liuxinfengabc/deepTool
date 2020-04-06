#coding:utf-8
import sys
import os
import re
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
'''
def find_all_hdf5_fun(root_dir):
    hdf5_file_list = []
    list = os.listdir(root_dir)
    for df in list:
        path  = os.path.join(root_dir,df)
        if os.path.isdir(path):
            hdf5_file_list.extend(find_all_hdf5_fun(path))
        if os.path.isfile(path):
            hdf5_file_list.append(path)
    return  hdf5_file_list
_fs = find_all_hdf5_fun('D:/DeepTool/DeepTool/deal-data-whole2/train_procese')
for f in _fs:
    print(f)
_k = filter(lambda x:re.compile(r'[*].hdf5').search(x),_fs)
'''
import  fnmatch
def all_files(root, patterns='*', single_level=False, yield_folders=False):
    #将模式从字符串中取出放入列表中
    patterns = patterns.split(';') #可以指定多个后缀作为需要获取的文件类型，通过';'作为分隔符
    for path, subdir, files in os.walk(root):

        if yield_folders :
            files.extend(subdir)
        files.sort()
        for name in files:
            for pattern in patterns:
                if fnmatch.fnmatch(name, pattern):
                    yield os.path.join(path, name)
                    break
        if single_level:
            break
        #print(path, subdir, files)
#for path in all_files('D:/DeepTool/DeepTool/deal-data-whole2/train_procese', patterns='*.hdf5'):
#    print(path)