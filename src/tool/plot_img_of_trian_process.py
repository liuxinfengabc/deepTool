import matplotlib.pyplot as plt
import matplotlib.collections as mcol
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
from matplotlib.lines import Line2D
import numpy as np
import fileinput
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 13,
}
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 13,
}
for i in range(110000,110010):  #文件目录，按照顺序排的
    rootpath = 'D:/DeepTool/DeepTool/deal-data-whole5/train_procese/' #数据根目录
    datapath = rootpath + str(i)+'/InceptionResNetV2__lose_epoch.txt'

    for line in fileinput.input(datapath, inplace=1):
        if not fileinput.isfirstline():
            print(line.replace('\n', ''))

    data = np.loadtxt(datapath,delimiter=',')
    plt.figure(1)
    plt.figure(figsize=(12, 8))
    #plt.plot(data[:,0],'r--',data[:,1],'g^',data[:,2],'bs',data[:,3],'bo')
    plt.plot(data[1:,1],'k',marker='<',label='train acc')
    plt.plot(data[1:,3],'b--',markersize=1,label='val acc')
    plt.xlabel('epoch',font2)
    plt.ylabel('acc',font2)
    #plt.title('Vgg16 train accuracy and validation accuracy')
    plt.legend(loc = 0, prop=font1)
    plt.savefig(rootpath+str(i)+"/filename1.png")
    plt.figure(2)
    plt.figure(figsize=(12, 8))
    plt.plot(data[:,0],'r--',label='train loss')
    plt.plot(data[:,2],marker='>',markersize=5,label='val loss')
    plt.xlabel('epoch',font2)
    plt.ylabel('loss',font2)
    #plt.title('Vgg16 train loss and validation loss')
    plt.legend(loc = 0, prop = font1)
    plt.savefig(rootpath+str(i)+"/filename2.png")
    #plt.show()