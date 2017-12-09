# -*-coding:utf-8-*-
'''
Author：    Yu Xianjia，Li Qingqing
Date :      2017.12.9
Dicription :Deal with file which recording TEN actions ONE times in a '.csv' type
Output :    A python list in nine length, the first 8 len is the maxium value of
            an action emg signal, the last len is the label of the action;
'''

import csv
import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pywt
import os.path
import copy
from matplotlib import mlab
from scipy.signal import butter, lfilter

######## 加载数据 #########
def loaddata(filename):
    with open('data/{}'.format(filename),'r') as csvfile:
        reader =  csv.reader(csvfile)
        rows = list(reader)
        for i in range(1,len(rows)):
            for y in range(len(rows[i])):
                rows[i][y] = float(rows[i][y])
        del rows[0]
    #print ">>总共的采样点数为",':',len(rows)
    #print ">>原始数据列数", ':', len(rows[0])
    return rows

######## 计算平均值，并加入数据中 ########
def AddMeanChannel(rows):
    rows_mean = []
    for n in range(len(rows)):
        s = 0
        for m in range(1,len(rows[n])):
            s = s + rows[n][m]
        rows[n].append(s/10)
    rows_mean = rows
    # print ">>载入平均值后的数据列数为", ':', len(rows_mean[0])
    return rows_mean

######## 计算极大值，并加入数据中 ########
def AddMaxChannel(rows,rows_mean):
    rows_mean_max = rows_mean
    for n in range(len(rows)):
        mx = max(rows[n][1:])
        mn = min(rows[n][1:])
        if abs(mx) >= abs(mn) :
            rows_mean_max[n].append(mx)
        else :
            rows_mean_max[n].append(mn)
    # print ">>载入极大值后的数据列数为", ':', len(rows_mean_max[0])
    return rows_mean_max

######## 通过时间轴算出时间，加入数据中，并将行换成列【前8通道，平均值，极大值，时间】 ########
def ToColumns(rows_in):
    temp = []
    for x in range(0,len(rows_in)):
        temp.append(0.000001*(float(rows_in[x][0])-float(rows_in[1][0])))

    cols = []
    for j in range(len(rows_in[1])):
        column = [row[j] for row in rows_in]
        cols.append(column)
    cols.append(temp)
    del cols[0]
    # print ">>最终转换后的数据列数<应为11>", ':', len(cols)
    return cols

#########################
########## 去噪 ##########
#########################

####### highpass_filter ########
def butter_highpass(highcut, Fs, order):
    nyq = 0.5 * Fs
    high = highcut / nyq
    b, a = butter(order, high, btype='highpass')
    return b, a

def butter_highpass_filter(data, highcut, Fs, order):
    b, a = butter_highpass(highcut, Fs, order=order)
    y = lfilter(b, a, data)
    return y

######## 去50HZ工频 ########
def pli_remove(Fs,F0,data):
    Q = 35.0
    w0 = F0/(Fs/2)
    b, a = sg.iirnotch(w0,Q)
    y = lfilter(b,a,data)
    return y

######## 小波去噪 ########
def shift(x,a):
    return np.roll(x,a)

def TI_Denoise(list,wavefunc,lv,m,n):
    # 分解
    left_list = shift(list,-10)
    coeff = pywt.wavedec(left_list,wavefunc,mode='sym',level=lv)
    for i in range(m,n+1):
        cD = coeff[i]
        #sigma = np.median(abs(cD.std()))
        Tr = np.sqrt(2*np.log(len(cD)))  # 计算阈值
        coeff[i] = pywt.threshold(cD,Tr,'soft')
    # 重构
    denoised = pywt.waverec(coeff, wavefunc)
    denoised_signal = shift (denoised, 10)
    return denoised_signal

#########################
########## 分割 ##########
#########################

######## 求移动平均值，使曲线平滑 #######
def MovingAverage(args, boxwidth):
    y = []
    for i in range(len(args)):
        y.append(abs(args[i]))
    return np.convolve(y, np.ones((boxwidth,))/boxwidth)[(boxwidth-1):]

def AllMovingAverage(emg_signal,boxwidth):
    emg_after_movingaverage = []
    for i in range(len(emg_signal)):
        emg_after_movingaverage.append(MovingAverage(emg_signal[i],boxwidth))
    return emg_after_movingaverage

def maxva(args= [], start = 0, end = 0):
    maxval = args[start]
    maxindex = start
    for i in range(start, end):
        if args[i] > maxval:
            maxval = round(args[i], 2)
            maxindex = i
    return maxval, maxindex

def FindMaxIndex(args):
    lrange = [[1040, 3140], [3140, 5140], [5140, 7140], [7140, 9140], [9140, 11140],
              [11140, 13140], [13140, 15140], [15140, 17140], [17140, 19140], [19140, 21140]]
    rangeindex = lrange[:]
    maxvaluenum = 0
    maxvalue = []
    for i in range(0, len(lrange)):
        m = lrange[i][0]
        n = lrange[i][1]
        maxvaluenum, rangeindex[i] = maxva(args, m, n)
        maxvalue.append(maxvaluenum)
        # print('maxvalue', maxvaluenum)
    # print('maxvalue', maxvalue)
    return maxvalue, rangeindex
	
def FindAllMaxVal(args):
    all_maxvalue = []
    all_maxindex = []
    for i in range(0,8):
        maxVal, maxIndex = FindMaxIndex(args[i])
        all_maxindex.append(maxIndex)
        all_maxvalue.append(maxVal)
        if len(maxVal) != 10:
            # print ">>第{}通道最大值".format(i),':',maxVal, '   OK ! '
        # else:
            print ">>第{}通道最大值".format(i),' WRONG !!!'
    return all_maxvalue,all_maxindex

def norm(tmp):
    max_num = max(tmp)
    for i in range(len(tmp)):
        tmp[i] = tmp[i] / max_num
    return tmp

# 将8列10行，转换成10行8列，这样每一行就为一个动作的参数
def Transform_data(all_maxvalue, label):
    max_value = []
    for i in range(0,10):
        tmp = []
        label = i # 每次只执行一个文件，恰好为执行的代号
        for j in range(0,8):
            tmp.append(all_maxvalue[j][i])
        # 归一化
        # tmp = norm(tmp)
        tmp.append(label)
        max_value.append(tmp)
        # print ">> 动作",tmp
    return max_value

# Data Collecting and denoisy
def data_collecting(data_name, label):
    """ d """
    ## 参数 ##
    fs = 200
    f0 = 50
    highcut = 0.3
    noverlap = 148
    lrange = [(800, 3140), (3140, 5140), (5140, 7140), (7140, 9140), (9140, 11440), (11440, 13140), (13140, 15140),
              (15140, 17000), (17000, 19000), (19000, 21140)]
    avpara = 700

    all_maxvalue_t = []
	## 提取数据 ##

    path = os.getcwd()
    # for root,dir,files in os.walk('{}/data'.format(path)):
    # print "######## 提取数据部分 ########"
    # for i in range(len(data_name)):
    for root,dir,files in os.walk('{}/data'.format(path)): # 得到全部的文件
        pass

    wrong_files = [] # 记录处理出错的文件
    for file in range(len(files)):
        rows = loaddata(files[file])
        print ">>处理数据文件 ：{}".format(files[file])
        
        try:
            ## 求出每一行数据的平均值 ##
            rows_mean = AddMeanChannel(rows)
            ## 求出每一行数据的极大值 ##
            rows_in = AddMaxChannel(rows, rows_mean)
            ## 列变行
            cols = ToColumns(rows_in)

            ## 去噪 ##
           #  print "## 数据去噪部分 ##"
            emg_after_denoise = []
            for i in range(len(cols)-1):
                y = butter_highpass_filter(cols[i], highcut, fs, order=6)
                y_pli = pli_remove(fs, f0, y)
                y_ti = TI_Denoise(y_pli,'sym8',10,1,10)
                emg_after_denoise.append(y_ti)
            #  print ">>去噪后的数据列数",':',len(emg_after_denoise)

            ## 分割 ##
            # print "## 数据平滑部分 ##"
            # 平滑 #
            mavg = AllMovingAverage(emg_after_denoise, avpara)
            all_maxvalue, all_maxindex = FindAllMaxVal(mavg)# maxvalue, maxindex
            # print ">>最大值",':',all_maxvalue
            # print ">>相应索引值", ':', all_maxindex

            # 如果数据集是空的，就将此列设为初始列
            # 如果不是空的， 提取生成的每一个动作的八个数据，增加到原来的列
            if not all_maxvalue_t:
                all_maxvalue_t = Transform_data(all_maxvalue, label)
            else:
                tmp = Transform_data(all_maxvalue, label)
                for i in range(len(tmp)):
                    all_maxvalue_t.append(tmp[i])
        except:
            print "Wrong File {}".format(files[file])
            wrong_files.append(files[file])
    return all_maxvalue_t, wrong_files

def KNN_algotithm(dataset1,dataset2):
    # 提取一个数据
    pass

# select the data
def select_batch(data = [],  length = 10):
    x_batch = []
    y_batch = []
    for i in range(len(data)):
	tmp_x = []
	tmp_y_one_hot = np.zeros((10), np.int)
        for j in range(9):
	    if j <= 7:
		tmp_x.append(data[i][j])
	    else:
		tmp_y_one_hot[data[i][j]] = 1
	x_batch.append(tmp_x)
	y_batch.append(np.ndarray.tolist(tmp_y_one_hot))
    return x_batch, y_batch

def run():
    data_name = ['emg_liqingqing', 'emg_qing_', 'emg_yu_0']
    dataset, wrong_files = data_collecting(data_name, label = 0)
    print ">>共得到数据{}组 \n ".format(len(dataset))
    print ">>出现错误的数据 \n" , wrong_files
    x_batch, y_batch = select_batch(dataset)
    return x_batch, y_batch
	
	
