#！-*- coding:utf-8 -*-

import matplotlib
matplotlib.use('TkAgg')

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
    with open(filename,'r') as csvfile:
        reader =  csv.reader(csvfile)
        rows = list(reader)
        for i in range(1,len(rows)):
            for y in range(len(rows[i])):
                rows[i][y] = float(rows[i][y])
        del rows[0]
    print ">>总共的采样点数为",':',len(rows)
    print ">>原始数据列数", ':', len(rows[0])
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
    print ">>载入平均值后的数据列数为", ':', len(rows_mean[0])
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
    print ">>载入极大值后的数据列数为", ':', len(rows_mean_max[0])
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
    print ">>最终转换后的数据列数<应为11>", ':', len(cols)
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
    denoised = pywt.waverec(coeff,wavefunc)
    denoised_signal = shift (denoised,10)
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


def RightBoundry(mvargs, lrangeone, maxvalcoefone, rangeindexone): #args，maxvalcoef, rangeindex是一个通道的数据
    rightbound = []
    for i in range(0, len(rangeindexone)):      #每个通道内动作循环
        tmp = []
        m = rangeindexone[i]
        n = lrangeone[i][1]
        #print('m,n', m, n)
        for j in range(m, n):    #在每个最大值至通道右边界内循环
            num = maxvalcoefone[i] + mvargs[n]
            # print('范围是  m =》,n', m, n)
            # print('mvargs[j]左值', mvargs[j], 'mvargs[j+1]右值', mvargs[j + 1], 'maxvalcoefone[i]成比例缩小后的值', maxvalcoefone[i], 'num 加上噪声的值',num, 'j比较至', j, 'mvargs[n]右边界值', mvargs[n])
            # tmp1 = (float(mvargs[j]) <= float(num))
            # tmp2 = (float(mvargs[j+1]) >= float(num))
            # print('tmp1',tmp1,'tmp2',tmp2)
            if (mvargs[j-1] > num) and (mvargs[j+1] <= num):
                tmp.append(j)
        # if len(tmp) == 0:
        #     print('Can\'t Find right Bound')
            tmp.append(n)
        if len(tmp) >= 1:
            rightbound.append(tmp[0])
        # print('rb', rightbound)
        # print('')
    return rightbound


def LeftBoundry(mvargs, lrangeone, maxvalcoefone, rangeindexone):
    #print('mvargs', mvargs, 'lrangeone', lrangeone, 'maxvalue', maxvalcoefone, 'maxindex', rangeindexone)
    leftbound = []
    for i in range(0, len(rangeindexone)):      #每个通道内动作循环
        tmp = []
        m = lrangeone[i][0]
        n = rangeindexone[i]
        # print('m,n', m, n)
        for j in range(m, n):    #在每个通道左边界至最大值内循环
            num = maxvalcoefone[i] + mvargs[m]
            # print('mvargs[j]', mvargs[j], 'mvargs[j+1]',mvargs[j+1], 'maxvalcoefone[i]',maxvalcoefone[i], 'num', num, 'j', j, 'mvargs[m]', mvargs[m])
            # tmp1 = (float(mvargs[j]) <= float(num))
            # tmp2 = (float(mvargs[j+1]) >= float(num))
            # print('tmp1',tmp1,'tmp2',tmp2)
            if (mvargs[j] < num) and (mvargs[j+1] >= num):
                tmp.append(j)
        # if len(leftbound) == 0:
        #     print('Can\'t Find Left Bound')
        if len(tmp) >= 1:
            leftbound.append(tmp[0] + 500)
        # print('lb', leftbound)
    return leftbound

    #lrangeone 是人为确定的大概动作的范围值，目的是在此基础上缩小区域找到上下界
    #maxvalue 是1⃣️lrange为基础的此范围内的最大值
    #ragneindex 是最大值对应的索引
    #procoefficient 是此范围内的缩小值，以移动平均窗为基础找到的最大值
    #mvargs 时间平均窗的数据，以此为基础查找范围


def FindBoundary(mvargs, lrangeone, maxvalue, maxindex, procoefficient):
    #确定边界的值，按照比例系数乘以峰值来确定便捷的值
    # print('mvargs',mvargs,'lrangeone',lrangeone, 'maxvalue',maxvalue, 'maxindex', maxindex, 'procoefficient',procoefficient)
    maxval = []
    maxindex_fb = maxindex[:]
    for i in range(0, len(maxvalue)):
        maxval.append(round(maxvalue[i] * procoefficient, 2))
    # print('mvargs',mvargs,'lrangeone',lrangeone, 'maxvalue', maxval, 'maxindex', maxindex_fb, 'procoefficient',procoefficient)
    rightbound = RightBoundry(mvargs, lrangeone, maxval, maxindex_fb)
    leftbound = LeftBoundry(mvargs, lrangeone, maxval, maxindex_fb)
    # print('leftbound', leftbound)
    # print('rightbound', rightbound)
    #以最大值为中心向左和向右找到相等的边界值
    return leftbound, rightbound


 # 求出在左右边界之间的重心和面积
def Square( mavg,leftbound,rightbound):
    weight_center = []
    for t in range(len(leftbound)):
        sum_area_list = []
        sum_area = 0
    #求出这一段的面积
        for i in range(leftbound[t], rightbound[t]):
            sum_area = sum_area + mavg[i]
            sum_area_list.append(sum_area)
    #找到重心
        for j in range(0, len(sum_area_list)):
            if(sum_area_list[j]<=(sum_area/2) and sum_area_list[j+1] >=(sum_area/2)):
                weight_center.append(leftbound[t] + j)
                #print('j ', leftbound[t] + j)
    return weight_center

# 求出在左右边界之间的 重心和面积
def AllSquare(mavgs, leftbound, rightbound):
    all_weight_center = []
    for i in range(len(mavgs)):
        all_weight_center.append(Square(mavgs[i], leftbound, rightbound))
    return all_weight_center


def all_final_bound(mavgs, all_weight_center, gestrue_width):
    final_lefbound = []
    final_rightbound = []
    final_lefbound_all_channel = []
    final_rightbound_all_channel = []
    tmp_left = 0
    tmp_right = 0
    for i in range(len(mavgs)):  # 计算每个通道
        for j in range(len(gestrue_width)):  # 计算通道内十个动作的左右边界
            tmp_left = all_weight_center[i][j] - (gestrue_width[j]/2)
            tmp_right = all_weight_center[i][j] + (gestrue_width[j]/2)
            final_lefbound.append(tmp_left)
            final_rightbound.append(tmp_right)
        final_lefbound_all_channel.append(final_lefbound)
        final_rightbound_all_channel.append(final_rightbound)
        final_lefbound = []
        final_rightbound = []
    return final_lefbound_all_channel, final_rightbound_all_channel

# 统一分割宽度，将重心至于每一个分割的中心，然后将得到的左右边界值依次做一个调整，将修正后的左右边界返回
# def UnionWidth(weightcenter,leftbound,rightbound):
#     #找出最大宽度
#     value = 0
#     max_width = 0
#     num = 0
#     for i in range(len(leftbound)):
#         value = rightbound[i] - leftbound[i]
#         if(value >= max_width):
#             max_width = value
#             num = i
#     # 将重心放在中心处，根据最大宽度计算左右边界值
#     print ">>最大宽度", ':', max_width, " i = ", num
#     half_max_width = int(max_width/2)
#     for j in range(len(leftbound)):
#         leftbound[j] = weightcenter[j] - half_max_width
#         rightbound[j] = weightcenter[j] + half_max_width
#     return leftbound, rightbound

# def AllUnionWidth(weightcenter,leftbound,rightbound):
#     FinalLeftBounds = []
#     FinalRightBounds = []
#     for i in range(len(weightcenter)):
#         temp_1, temp_2 = UnionWidth(weightcenter[i], leftbound, rightbound)
#         FinalLeftBounds.append(temp_1)
#         FinalRightBounds.append(temp_2)
#     return FinalLeftBounds, FinalRightBounds

# 根据第九通道计算出每个动作的宽度
def gestrue_width(leftbound, rightbound):
    gestrue_width_ls = []
    for i in range(len(leftbound)):
        width = rightbound[i] - leftbound[i]
        gestrue_width_ls.append(width)
    return gestrue_width_ls


def CutData(lrange, signal, time):
    s_temp = []
    t_temp = []
    for i in range(len(lrange)):
        s_temp.append(signal[lrange[i][0]:lrange[i][1]])
        t_temp.append(time[lrange[i][0]:lrange[i][1]])
    return s_temp, t_temp

def AllCutData(lrange, signal, time):
    temp_1 = []
    temp_2 = []
    for i in range(len(signal)):
        s_temp, t_temp = CutData(lrange, signal[i], time)
        temp_1.append(s_temp)
        temp_2.append(t_temp)
    return temp_1, temp_2


def AfterCutEmg(leftbound,rightbound,col):
    emg_after_cut = []
    for i in range(len(leftbound)):
        emg_after_cut.append(col[leftbound[i]:rightbound[i]])
    return emg_after_cut

def AllCutEmg(leftBounds, rightBounds, emg_signal, emg_time):
    all_emg_cut = []
    time = []
    # time_temp = []
    for i in range(len(emg_signal)):
        all_emg_cut.append(AfterCutEmg(leftBounds, rightBounds, emg_signal[i]))
        time.append(AfterCutEmg(leftBounds, rightBounds, emg_time))
    # time_temp = copy.deepcopy(AfterCutEmg(leftBounds[i], rightBounds[i], emg_time))
    # for j in range(len(time_temp)):
    #     for i in range(len(time_temp[j])):
    #         temp.append(time_temp[j][i] - time_temp[j][0])
    #     time.append(temp)
    #     temp = []
    return time, all_emg_cut

def trans(original_emg):
    cols = []
    for j in range(len(original_emg[0])):
        column = [row[j] for row in original_emg ]
        cols.append(column)
    return cols

#########################
####### 创建文件夹 ########
#########################

######## 创建文件夹 ########
def MakeDir():
    if not os.path.exists("Action_Figure"):
        ##### Action_Figure #####
        ### All_Channels ###
        ## Time_domain ##
        os.makedirs("Action_Figure/All_Channels/Time_Domain/With_Axis")
        os.makedirs("Action_Figure/All_Channels/Time_Domain/Without_Axis")
        ## Spectrum ##
        os.makedirs("Action_Figure/All_Channels/Spectrum/With_Axis")
        os.makedirs("Action_Figure/All_Channels/Spectrum/Without_Axis")
        ## Spectrogram ##
        os.makedirs("Action_Figure/All_Channels/Spectrogram/With_Axis")
        os.makedirs("Action_Figure/All_Channels/Spectrogram/Without_Axis")
        ### Single_Channel_Single_Action ###
        ## Time_Domain ##
        os.makedirs("Action_Figure/Single_Channel_Single_Action/Time_Domain/With_Axis")
        os.makedirs("Action_Figure/Single_Channel_Single_Action/Time_Domain/Without_Axis")
        ## Spectrum ##
        os.makedirs("Action_Figure/Single_Channel_Single_Action/Spectrum/With_Axis")
        os.makedirs("Action_Figure/Single_Channel_Single_Action/Spectrum/Without_Axis")
        ## Spectrogram ##
        os.makedirs("Action_Figure/Single_Channel_Single_Action/Spectrogram/With_Axis")
        os.makedirs("Action_Figure/Single_Channel_Single_Action/Spectrogram/Without_Axis")
        ### Nine_spectrograms_In_One ###
        ## Average ##
        os.makedirs("Action_Figure/Nine_In_One/Average/With_Axis")
        os.makedirs("Action_Figure/Nine_In_One/Average/Without_Axis")
        ## Max ##
        os.makedirs("Action_Figure/Nine_In_One/Max/With_Axis")
        os.makedirs("Action_Figure/Nine_In_One/Max/Without_Axis")
        # ## Spectrum ##
        # os.makedirs("Action_Figure/Nine_In_One/Average/With_Axis")
        # os.makedirs("Action_Figure/Nine_In_One/Average/Without_Axis")


#########################
########## 绘图 ##########
#########################

######### 时域图绘制 ########
def Time_Domain_Plot(time, emg_signal):
    plt.plot(time,emg_signal)

######## 频域图绘制 ########
def Spectrum(Fs,emg_signal):
    Ts = 1.0/Fs
    N = len(emg_signal)
    y= abs(np.fft.fft(emg_signal))
    freq=abs(np.fft.fftfreq(N, d=Ts))
    plt.plot(freq,y)
    #plt.show()

######## 频谱图/语谱图绘制 ########
def Spectrogram(nfft,noverlab,Fs,emg_signal):
    plt.specgram(emg_signal,
                 NFFT = nfft,
                 Fs=Fs,
                 window= mlab.window_hanning,
                 #window= np.hamming,
                 noverlap= noverlab,
                 cmap=cm.jet)
    # #plt.axis('off')  ##### remove axis  #####

######## 绘制每个通道的图<含时域、频域、语谱> ########
def AllChannels(axis, figcate, time,Fs, emg_signal,nfft, noverlap):
    if figcate == 0:    ### time_domain ###
        if axis == 1:    ### 有轴 ###
            for j in range(len(emg_signal)):
                # plt.ion() #plt.show() #plt.pause() plt.close()
                plt.figure("Channel-%d" % (j + 1))
                Time_Domain_Plot(time, emg_signal[j])
                photo_name = "Channel_%d.png" % (j+1)
                plt.savefig("Action_Figure/All_Channels/Time_Domain/With_Axis/%s" % photo_name,bbox_inches='tight')
                plt.close()
        elif axis == 0:   ### 无轴 ###
            for j in range(len(emg_signal)):
                plt.figure("Channel-%d" % (j + 1))
                Time_Domain_Plot(time, emg_signal[j])
                plt.axis('off')
                photo_name = "Channel_%d.png" % (j + 1)
                plt.savefig("Action_Figure/All_Channels/Time_Domain/Without_Axis/%s" % photo_name,bbox_inches='tight')
                plt.close()
    elif figcate == 1: ### 频域图 ###
        if axis == 1:   ### 有轴 ###
            for j in range(len(emg_signal)):
                plt.figure("Channel-%d" % (j + 1))
                Spectrum(Fs, emg_signal[j])
                photo_name = "Channel_%d.png" % (j + 1)
                plt.savefig("Action_Figure/All_Channels/Spectrum/With_Axis/%s" % photo_name, bbox_inches='tight')
                plt.close()
        elif axis == 0:   ### 无轴 ###
            for j in range(len(emg_signal)):
                plt.figure("Channel-%d" % (j + 1))
                Spectrum(Fs, emg_signal[j])
                plt.axis('off')
                photo_name = "Channel_%d.png" % (j + 1)
                plt.savefig("Action_Figure/All_Channels/Spectrum/Without_Axis/%s" % photo_name, bbox_inches='tight')
                plt.close()
    elif figcate == 2:
        if axis == 1:  ### 有轴 ###
            for j in range(len(emg_signal)):
                plt.figure("Channel-%d" % (j + 1))
                Spectrogram(nfft, noverlap, Fs, emg_signal[j])
                photo_name = "Channel_%d.png" % (j + 1)
                plt.savefig("Action_Figure/All_Channels/Spectrogram/With_Axis/%s" % photo_name, bbox_inches='tight')
                plt.close()
        elif axis == 0:  ### 无轴 ###
            for j in range(len(emg_signal)):
                plt.figure("Channel-%d" % (j + 1))
                Spectrogram(nfft, noverlap, Fs, emg_signal[j])
                plt.axis('off')
                photo_name = "Channel_%d.png" % (j + 1)
                plt.savefig("Action_Figure/All_Channels/Spectrogram/Without_Axis/%s" % photo_name,bbox_inches='tight')
                plt.close()

######## 绘制每个通道每个动作的图<含时域、频域、语谱> ########
def Single_Channel_Single_Action(axis, figcate, action, num, t, time,Fs, emg_signal,nfft, noverlap):
    if figcate == 0:    ### 时域图 ###
        if axis == 1:    ### 有轴 ###
            for j in range(len(emg_signal)):
                for i in range(len(emg_signal[j])):
                    # plt.ion() #plt.show() #plt.pause() plt.close()
                    plt.figure("%s-%d-%d-%d" % (action[i],j + 1,num, t))
                    Time_Domain_Plot(time[j][i], emg_signal[j][i])
                    photo_name = "%s-%d-%d-%d.png" % (action[i],j + 1,num, t)
                    plt.savefig("Action_Figure/Single_Channel_Single_Action/Time_Domain/With_Axis/%s" % photo_name,bbox_inches='tight')
                    plt.close()
        elif axis == 0:   ### 无轴 ###
            for j in range(len(emg_signal)):
                for i in range(len(emg_signal[j])):
                    plt.figure("%s-%d-%d-%d" % (action[i], j + 1, num, t))
                    Time_Domain_Plot(time[j][i], emg_signal[j][i])
                    plt.axis('off')
                    photo_name = "%s-%d-%d-%d.png" % (action[i], j + 1, num, t)
                    plt.savefig("Action_Figure/Single_Channel_Single_Action/Time_Domain/Without_Axis/%s" % photo_name,bbox_inches='tight')
                    plt.close()
    elif figcate == 1: ### 频域图 ###
        if axis == 1:   ### 有轴 ###
            for j in range(len(emg_signal)):
                for i in range(len(emg_signal[j])):
                    plt.figure("%s-%d-%d-%d" % (action[i], j + 1, num, t))
                    Spectrum(Fs, emg_signal[j][i])
                    photo_name = "%s-%d-%d-%d.png" % (action[i], j + 1, num, t)
                    plt.savefig("Action_Figure/Single_Channel_Single_Action/Spectrum/With_Axis/%s" % photo_name,bbox_inches='tight')
                    plt.close()
        elif axis == 0:   ### 无轴 ###
            for j in range(len(emg_signal)):
                for i in range(len(emg_signal[j])):
                    plt.figure("%s-%d-%d-%d" % (action[i], j + 1, num, t))
                    Spectrum(Fs, emg_signal[j][i])
                    plt.axis('off')
                    photo_name = "%s-%d-%d-%d.png" % (action[i], j + 1, num, t)
                    plt.savefig("Action_Figure/Single_Channel_Single_Action/Spectrum/Without_Axis/%s" % photo_name,bbox_inches='tight')
                    plt.close()
    elif figcate == 2:
        if axis == 1:  ### 有轴 ###
            for j in range(len(emg_signal)):
                for i in range(len(emg_signal[j])):
                    plt.figure("%s-%d-%d-%d" % (action[i], j + 1, num, t))
                    Spectrogram(nfft, noverlap, Fs, emg_signal[j][i])
                    photo_name = "%s-%d-%d-%d.png" % (action[i], j + 1, num, t)
                    plt.savefig("Action_Figure/Single_Channel_Single_Action/Spectrogram/With_Axis/%s" % photo_name,bbox_inches='tight')
                    plt.close()
        elif axis == 0:  ### 无轴 ###
            for j in range(len(emg_signal)):
                for i in range(len(emg_signal[j])):
                    plt.figure("%s-%d-%d-%d" % (action[i], j + 1, num, t))
                    Spectrogram(nfft, noverlap, Fs, emg_signal[j][i])
                    plt.axis('off')
                    photo_name = "%s-%d-%d-%d.png" % (action[i], j + 1, num, t)
                    plt.savefig("Action_Figure/Single_Channel_Single_Action/Spectrogram/Without_Axis/%s" % photo_name,bbox_inches='tight')
                    plt.close()

######## 一共九张图在一个figure上 <含语谱> ########
def Nine_In_One_Ave(axis, action, num, t, Fs, emg_signal,nfft, noverlap):
    emg_temp = trans(emg_signal)
    if axis == 1:
        for i in range(len(emg_temp)):
            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(13.6, 8))
            for j in range(len(emg_temp[i])-1):
                n = 330+j+1
                plt.subplot(n)
                Spectrogram(nfft,noverlap, Fs, emg_temp[i][j])
            photo_name = "%s - %d - %d.png" % (action[i], num, t)
            plt.savefig("Action_Figure/Nine_In_One/Average/With_Axis/%s" % photo_name,bbox_inches='tight')
            plt.close()
    elif axis == 0:
        for i in range(len(emg_temp)):
            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(13.6, 8))
            for j in range(len(emg_temp[i])-1):
                n = 330 + j + 1
                plt.subplot(n)
                Spectrogram(nfft, noverlap, Fs, emg_temp[i][j])
                plt.axis('off')
            photo_name = "%s - %d - %d.png" % (action[i], num, t)
            plt.savefig("Action_Figure/Nine_In_One/Average/Without_Axis/%s" % photo_name,bbox_inches='tight')
            plt.close()

def Nine_In_One_Max(axis, action, num, t, Fs, emg_signal, nfft, noverlap):
    emg_temp = trans(emg_signal)
    if axis == 1:
        for i in range(len(emg_temp)):
            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(13.6, 8))
            for j in range(len(emg_temp[i])):
                if j == 8:
                    pass
                elif j == 9:
                    n = 330 + j
                    plt.subplot(n)
                    Spectrogram(nfft, noverlap, Fs, emg_temp[i][j])
                else:
                    n = 330 + j + 1
                    plt.subplot(n)
                    Spectrogram(nfft, noverlap, Fs, emg_temp[i][j])
            photo_name = "%s - %d - %d.png" % (action[i], num, t)
            plt.savefig("Action_Figure/Nine_In_One/Max/With_Axis/%s" % photo_name,bbox_inches='tight')
            plt.close()
    elif axis == 0:
        for i in range(len(emg_temp)):
            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(13.6, 8))
            for j in range(len(emg_temp[i])):
                if j == 8:
                    pass
                elif j == 9:
                    n = 330 + j
                    plt.subplot(n)
                    Spectrogram(nfft, noverlap, Fs, emg_temp[i][j])
                    plt.axis('off')
                else:
                    n = 330 + j + 1
                    plt.subplot(n)
                    Spectrogram(nfft, noverlap, Fs, emg_temp[i][j])
                    plt.axis('off')
            photo_name = "%s - %d - %d.png" % (action[i], num, t)
            plt.savefig("Action_Figure/Nine_In_One/Max/Without_Axis/%s" % photo_name,bbox_inches='tight')
            plt.close()


# def CutTime(lrange, time):
#     s_temp = []
#     for i in range(len(lrange)):
#         s_temp.append(time[lrange[i][0]:lrange[i][1]])
#     for i in range(len(s_temp)):

##########################################
########### Main 中主要的实现过程 ###########
##########################################
## 第一步： 提取数据；                    ##
## 第二步： 求平均值与极大值；             ##
## 第三步： 提取每一个通道的数据；          ##
## 第四步： 去噪 <0HZ附近><工频><高频噪声>；##
## 第五步： 信号分割;                     ##
## 第六步： 绘图，保存相关图片;            ###
###########################################

def main():
    ## 参数 ##
    fs = 200
    f0 = 50
    highcut = 0.3
    nfft = 300
    noverlap = 295
    # lrange = [(1360, 2450), (3650, 4750), (5750, 6850), (7900, 8700), (9800, 10900), (11980, 12800), (13900, 14750),
    # #           (15850, 16750), (17700, 18650), (19650, 20700)]

    ## emg_yu_010.csv
    # lrange = [1150, 3450, 5500, 7500, 9450, 11700, 13700,
    #           15600, 17350, 19300]
    # rrange = [2200, 4500, 6500, 8560, 10600, 12700, 14650,
    #           16750, 18350, 20600]

    ##emg_yu_009.csv
    # lrange = [1200, 3500, 5550, 7450, 9550, 11750, 13700,
    #           15600, 17450, 19350]
    # rrange = [2200, 4550, 6500, 8550, 11050, 12850, 14650,
    #           16450, 18350, 20550]

    ##emg_yu_007.csv
    # lrange = [1250, 3450, 5500, 7400, 9550, 11700, 13650,
    #           15550, 17350, 19450]
    # rrange = [2300, 4700, 6450, 8600, 10850, 12600, 14750,
    #           16500, 18350, 20500]

    ##emg_yu_006.csv
    # lrange = [1100, 3250, 5400, 7400, 9400, 11650, 13600,
    #           15350, 17250, 19200]
    # rrange = [2200, 4600, 6450, 8500, 10750, 12600, 14550,
    #           16400, 18250, 20350]

    # ##emg_yu_012.csv
    # lrange = [1300, 3550, 5650, 7600, 9600, 11850, 13800,
    #           15750, 17500, 19500]
    # rrange = [2400, 4650, 6650, 8650, 10900, 12750, 14900,
    #           16550, 18500, 20600]

    ##emg_yu_013.csv
    # lrange = [1100, 3350, 5400, 7400, 9500, 11650, 13600,
    #           15400, 17350, 19400]
    # rrange = [2250, 4500, 6500, 8600, 10800, 12700, 14550,
    #           16350, 18300, 20600]

    ##emg_yu_014.csv
    # lrange = [1250, 3400, 5550, 7600, 9550, 11750, 13800,
    #           15650, 17400, 19400]
    # rrange = [2500, 4550, 6550, 8700, 10900, 12800, 14950,
    #           16550, 18500, 20550]

    ##emg_yu_015.csv
    # lrange = [1200, 3500, 5550, 7400, 9450, 11650, 13700,
    #           15550, 17350, 19300]
    # rrange = [2400, 4550, 6550, 8650, 10900, 12750, 14600,
    #           16400, 18450, 20500]

    ##emg_yu_016.csv
    # lrange = [1150, 3450, 5500, 7350, 9500, 11600, 13550,
    #           15500, 17350, 19300]
    # rrange = [2400, 4450, 6450, 8500, 10850, 12850, 14700,
    #           16550, 18300, 20500]

    ##emg_yu_017.csv
    # lrange = [1150, 3450, 5500, 7550, 9600, 11700, 13700,
    #           15550, 17350, 19300]
    # rrange = [2400, 4750, 6750, 8650, 10750, 12800, 14700,
    #           16650, 18150, 20600]

    # ##emg_yu_020.csv
    # lrange = [1200, 3400, 5350, 7350, 9450, 11600, 13600,
    #           15450, 17300, 19300]
    # rrange = [2200, 4400, 6400, 8550, 10700, 12550, 14550,
    #           16250, 18250, 20350]

    # ##emg_yu_021.csv
    # lrange = [1150, 3350, 5400, 7450, 9450, 11550, 13600,
    #           15400, 17300, 19250]
    # rrange = [2250, 4450, 6400, 8500, 10800, 12900, 14750,
    #           16400, 18250, 20350]

    # lrange = [1290, 3550, 5600, 7600, 9600, 11760, 13800,
    #           15650, 17450, 19500]
    # rrange = [2150, 4430, 6540, 8400, 10650, 12600, 14550,
    #           16500, 18300, 20250]

    # ##emg_yu_022.csv
    # lrange = [1150, 3350, 5400, 7400, 9350, 11550, 13450,
    #           15400, 17250, 19200]
    # rrange = [2350, 4450, 6400, 8500, 10700, 12650, 14600,
    #           16600, 18300, 20400]

    ##emg_yu_024.csv
    # lrange = [1200, 3400, 5500, 7550, 9650, 11750, 13600,
    #           15500, 17300, 19350]
    # rrange = [2400, 4500, 6600, 8600, 10750, 12600, 14550,
    #           16650, 18350, 20400]

    # ##emg_yu_025.csv
    # lrange = [1200, 3400, 5350, 7300, 9600, 11700, 13600,
    #           15350, 17250, 19300]
    # rrange = [2300, 4500, 6400, 8400, 10500, 12500, 14450,
    #           16400, 18200, 20400]

    ##emg_yu_026.csv
    # lrange = [1200, 3350, 5550, 7600, 9550, 11700, 13650,
    #           15400, 17350, 19250]
    # rrange = [2300, 4500, 6550, 8500, 10550, 12650, 14600,
    #           16550, 18250, 20400]

    # ##emg_qing_21.csv
    # lrange = [1150, 3380, 5450, 7500, 9550, 11650, 13650,
    #           15500, 17350, 19250]
    # rrange = [2200, 4500, 6400, 8400, 10550, 12500, 14400,
    #           16300, 18150, 20300]

    # ##emg_qing_22.csv
    # lrange = [1100, 3280, 5350, 7400, 9450, 11550, 13600,
    #           15450, 17250, 19250]
    # rrange = [2100, 4400, 6400, 8350, 10600, 12600, 14500,
    #           16350, 18120, 20250]

    # ##emg_qing_23.csv
    # lrange = [1130, 3280, 5380, 7400, 9450, 11550, 13600,
    #           15450, 17250, 19250]
    # rrange = [2100, 4400, 6400, 8350, 10600, 12600, 14500,
    #           16380, 18220, 20420]

    # ##emg_qing_24.csv
    # lrange = [1130, 3300, 5420, 7420, 9450, 11650, 13600,
    #           15450, 17250, 19300]
    # rrange = [2250, 4500, 6350, 8400, 10650, 12550, 14400,
    #           16430, 18220, 20320]

    # ##emg_qing_25.csv
    # lrange = [1130, 3300, 5420, 7420, 9450, 11650, 13600,
    #           15450, 17250, 19250]
    # rrange = [2100, 4300, 6400, 8400, 10680, 12500, 14450,
    #           16400, 18220, 20280]

    # ##emg_qing_26.csv
    # lrange = [1180, 3400, 5470, 7520, 9450, 11650, 13650,
    #           15500, 17300, 19250]
    # rrange = [2200, 4400, 6400, 8400, 10680, 12600, 14450,
    #           16400, 18250, 20280]

    # ##emg_qing_27.csv
    # lrange = [1180, 3400, 5470, 7520, 9450, 11650, 13650,
    #           15500, 17300, 19250]
    # rrange = [2200, 4400, 6400, 8400, 10680, 12500, 14450,
    #           16440, 18150, 20380]

    # ##emg_qing_28.csv
    # lrange = [1180, 3300, 5440, 7420, 9450, 11650, 13650,
    #           15500, 17300, 19250]
    # rrange = [2200, 4400, 6350, 8450, 10680, 12500, 14450,
    #           16400, 18050, 20280]


    # ##emg_qing_29.csv
    # lrange = [1250, 3400, 5490, 7480, 9520, 11700, 13650,
    #           15550, 17350, 19350]
    # rrange = [2200, 4450, 6400, 8450, 10680, 12550, 14500,
    #           16450, 18100, 20330]

    # ##emg_qing_30.csv
    # lrange = [1150, 3400, 5440, 7480, 9500, 11600, 13650,
    #           15450, 17250, 19200]
    # rrange = [2150, 4400, 6400, 8400, 10630, 12450, 14500,
    #           16400, 18150, 20330]

    # ##emg_qing_31.csv
    # lrange = [1100, 3350, 5380, 7430, 9450, 11600, 13650,
    #           15450, 17250, 19300]
    # rrange = [2100, 4450, 6400, 8400, 10630, 12450, 14500,
    #           16550, 18150, 20330]

    # ##emg_qing_32.csv
    # lrange = [1250, 3450, 5480, 7500, 9650, 11700, 13650,
    #           15700, 17350, 19400]
    # rrange = [2150, 4350, 6500, 8500, 10730, 12550, 14550,
    #           16450, 18250, 20380]

    # ##emg_qing_33.csv  Broken data
    # lrange = [1200, 3350, 5450, 7450, 9550, 11650, 13600,
    #           15450, 17300, 19250]
    # rrange = [2200, 4450, 6450, 8450, 10600, 12600, 14600,
    #           16600, 18250, 20350]

    # ##emg_qing_34.csv  Broken data
    # lrange = [1200, 3350, 5450, 7450, 9500, 11650, 13600,
    #           15500, 17300, 19300]
    # rrange = [2150, 4400, 6450, 8450, 10600, 12600, 14600,
    #           16500, 18250, 20450]

    # ##emg_qing_35.csv  Broken data
    # lrange = [1200, 3350, 5450, 7450, 9500, 11650, 13600,
    #           15500, 17300, 19250]
    # rrange = [2050, 4350, 6400, 8450, 10600, 12600, 14500,
    #           16400, 18250, 20350]

    # ##emg_qing_36.csv  Broken data
    # lrange = [1250, 3350, 5450, 7450, 9500, 11650, 13660,
    #           15550, 17300, 19400]
    # rrange = [2150, 4400, 6450, 8450, 10650, 12600, 14600,
    #           16450, 18250, 20350]

    # ##emg_qing_37.csv  Broken data
    # lrange = [1200, 3350, 5450, 7550, 9550, 11750, 13700,
    #           15550, 17350, 19300]
    # rrange = [2150, 4500, 6450, 8450, 10750, 12600, 14600,
    #           16500, 18250, 20450]

    # ##emg_qing_38.csv  Broken data
    # lrange = [1200, 3350, 5450, 7450, 9500, 11600, 13600,
    #           15500, 17300, 19300]
    # rrange = [2150, 4400, 6450, 8450, 10600, 12600, 14550,
    #           16500, 18250, 20450]

    # ##emg_qing_39.csv  Broken data
    # lrange = [1100, 3350, 5450, 7450, 9500, 11600, 13600,
    #           15500, 17250, 19300]
    # rrange = [2150, 4400, 6450, 8450, 10600, 12500, 14550,
    #           16400, 18150, 20250]

    # ##emg_qing_40.csv  Broken data
    # lrange = [1150, 3350, 5450, 7450, 9450, 11600, 13600,
    #           15500, 17200, 19300]
    # rrange = [2100, 4350, 6450, 8450, 10650, 12550, 14500,
    #           16400, 18200, 20350]

    # ##emg_qing_41.csv  Broken data
    # lrange = [1100, 3350, 5350, 7400, 9500, 11600, 13600,
    #           15500, 17300, 19250]
    # rrange = [2150, 4400, 6350, 8450, 10600, 12550, 14500,
    #           16500, 18200, 20250]
    #
    # ##emg_qing_42.csv  Broken data
    # lrange = [1300, 3550, 5550, 7600, 9600, 11700, 13800,
    #           15700, 17450, 19450]
    # rrange = [2250, 4600, 6550, 8550, 10900, 12650, 14600,
    #           16600, 18450, 20550]

    # ##emg_qing_43.csv  Broken data
    # lrange = [1100, 3350, 5350, 7400, 9500, 11600, 13600,
    #           15500, 17300, 19250]
    # rrange = [2150, 4400, 6350, 8250, 10700, 12550, 14500,
    #           16500, 18150, 20350]

    # ##emg_qing_44.csv  Broken data
    # lrange = [1200, 3350, 5350, 7400, 9500, 11600, 13600,
    #           15500, 17400, 19250]
    # rrange = [2150, 4400, 6450, 8450, 10600, 12550, 14500,
    #           16400, 18200, 20450]

    # ##emg_qing_45.csv  Broken data
    # lrange = [1200, 3350, 5400, 7400, 9600, 11600, 13600,
    #           15500, 17200, 19250]
    # rrange = [2100, 4400, 6450, 8450, 10650, 12500, 14500,
    #           16450, 18300, 20350]

    # ##emg_qing_46.csv  Broken data
    # lrange = [1150, 3350, 5350, 7400, 9500, 11600, 13600,
    #           15500, 17300, 19250]
    # rrange = [2150, 4400, 6350, 8450, 10650, 12450, 14500,
    #           16400, 18200, 20450]

    # ##emg_qing_47.csv  Broken data
    # lrange = [1500, 3850, 5850, 7900, 10000, 12100, 14100,
    #           15900, 17700, 19650]
    # rrange = [2600, 4800, 6750, 8750, 11100, 12950, 14900,
    #           16800, 18700, 20700]

    # ##emg_qing_48.csv  Broken data
    # lrange = [1200, 3400, 5450, 7450, 9500, 11600, 13650,
    #           15500, 17300, 19450]
    # rrange = [2150, 4400, 6450, 8450, 10600, 12550, 14550,
    #           16450, 18400, 20450]

    # ##emg_qing_49.csv  Broken data
    # lrange = [1200, 3350, 5450, 7500, 9500, 11600, 13650,
    #           15500, 17300, 19450]
    # rrange = [2150, 4500, 6500, 8450, 10700, 12550, 14550,
    #           16400, 18250, 20350]

    # ##emg_qing_50.csv  Broken data
    # lrange = [1200, 3350, 5450, 7400, 9500, 11600, 13600,
    #           15500, 17300, 19250]
    # rrange = [2150, 4400, 6400, 8450, 10650, 12550, 14500,
    #           16400, 18150, 20350]

    # ##emg_qing_51.csv  Broken data
    # lrange = [1100, 3350, 5400, 7400, 9450, 11600, 13600,
    #           15500, 17300, 19250]
    # rrange = [2100, 4400, 6400, 8450, 10600, 12450, 14450,
    #           16400, 18200, 20300]

    # ##emg_qing_52.csv  Broken data
    # lrange = [1250, 3350, 5400, 7400, 9500, 11600, 13600,
    #           15500, 17350, 19250]
    # rrange = [2200, 4400, 6450, 8450, 10600, 12550, 14500,
    #           16400, 18200, 20300]

    # ##emg_qing_53.csv  Broken data
    # lrange = [1700, 3850, 5850, 7900, 10000, 12100, 14200,
    #           16050, 17750, 19750]
    # rrange = [2600, 5000, 6950, 8950, 11200, 13050, 15050,
    #           16900, 18750, 20950]

    # ##emg_qing_54.csv  Broken data
    # lrange = [1050, 3350, 5300, 7400, 9600, 11600, 13600,
    #           15500, 17250, 19250]
    # rrange = [2150, 4400, 6450, 8450, 10600, 12550, 14400,
    #           16400, 18250, 20300]

    # ##emg_qing_55.csv  Broken data
    # lrange = [1250, 3350, 5450, 7500, 9500, 11650, 13650,
    #           15500, 17350, 19250]
    # rrange = [2200, 4500, 6450, 8450, 10600, 12550, 14500,
    #           16400, 18200, 20300]

    # ##emg_qing_56.csv  Broken data
    # lrange = [1150, 3350, 5400, 7450, 9500, 11600, 13600,
    #           15500, 17350, 19300]
    # rrange = [2100, 4400, 6450, 8400, 10600, 12350, 14500,
    #           16300, 18200, 20300]

    # ##emg_qing_57.csv  Broken data
    # lrange = [1200, 3350, 5400, 7450, 9500, 11600, 13650,
    #           15500, 17350, 19250]
    # rrange = [2200, 4400, 6450, 8450, 10600, 12550, 14600,
    #           16400, 18250, 20300]

    # ##emg_qing_58.csv  Broken data
    # lrange = [1250, 3450, 5500, 7600, 9600, 11750, 13800,
    #           15600, 17400, 19300]
    # rrange = [2200, 4600, 6550, 8550, 10800, 12550, 14750,
    #           16600, 18350, 20450]

    # ##emg_qing_59.csv  Broken data
    # lrange = [1250, 3350, 5500, 7500, 9600, 11650, 13650,
    #           15550, 17350, 19350]
    # rrange = [2200, 4400, 6450, 8450, 10600, 12550, 14500,
    #           16400, 18300, 20350]

    # ##emg_qing_60.csv  Broken data
    # lrange = [1150, 3350, 5450, 7400, 9450, 11600, 13600,
    #           15500, 17250, 19250]
    # rrange = [2100, 4400, 6350, 8450, 10600, 12550, 14500,
    #           16400, 18200, 20300]

    # ##emg_qing_61.csv  Broken data
    # lrange = [1250, 3350, 5400, 7450, 9500, 11600, 13600,
    #           15500, 17350, 19450]
    # rrange = [2200, 4400, 6450, 8450, 10650, 12550, 14500,
    #           16450, 18250, 20400]

    # ##emg_qing_62.csv  Broken data
    # lrange = [1250, 3450, 5400, 7500, 9500, 11700, 13700,
    #           15600, 17350, 19350]
    # rrange = [2200, 4400, 6450, 8400, 10650, 12550, 14600,
    #           16450, 18300, 20400]

    # ##emg_qing_63.csv  Broken data
    # lrange = [1250, 3450, 5400, 7500, 9550, 11600, 13650,
    #           15500, 17250, 19350]
    # rrange = [2200, 4400, 6450, 8400, 10650, 12550, 14600,
    #           16450, 18200, 20350]

    # ##emg_qing_64.csv  Broken data
    # lrange = [1200, 3400, 5450, 7500, 9500, 11700, 13800,
    #           15500, 17350, 19400]
    # rrange = [2150, 4400, 6400, 8400, 10650, 12550, 14600,
    #           16350, 18200, 20400]

    # ##emg_qing_65.csv  Broken data
    # lrange = [1450, 3450, 5500, 7500, 9600, 11700, 13600,
    #           15600, 17550, 19350]
    # rrange = [2200, 4500, 6450, 8450, 10750, 12550, 14600,
    #           16500, 18450, 20450]

    # ##emg_qing_66.csv  Broken data
    # lrange = [1150, 3450, 5400, 7450, 9500, 11650, 13650,
    #           15500, 17250, 19300]
    # rrange = [2150, 4400, 6400, 8450, 10650, 12550, 14550,
    #           16350, 18200, 20300]

    # ##emg_qing_67.csv  Broken data
    # lrange = [1150, 3350, 5400, 7450, 9500, 11650, 13650,
    #           15500, 17300, 19250]
    # rrange = [2150, 4400, 6400, 8450, 10650, 12550, 14550,
    #           16350, 18200, 20350]

    # ##emg_qing_68.csv  Broken data
    # lrange = [1150, 3450, 5450, 7550, 9600, 11700, 13650,
    #           15550, 17300, 19350]
    # rrange = [2150, 4400, 6450, 8500, 10700, 12650, 14550,
    #           16450, 18300, 20450]

    # ##emg_qing_69.csv  Broken data
    # lrange = [1150, 3450, 5500, 7550, 9550, 11650, 13650,
    #           15550, 17450, 19400]
    # rrange = [2150, 4400, 6400, 8450, 10750, 12550, 14550,
    #           16450, 18400, 20400]

    # ##emg_qing_70.csv  Broken data
    # lrange = [1200, 3450, 5450, 7500, 9500, 11650, 13700,
    #           15500, 17350, 19300]
    # rrange = [2150, 4400, 6450, 8450, 10700, 12600, 14600,
    #           16400, 18350, 20400]

    # ##emg_qing_71.csv  Broken data
    # lrange71 = [1100, 3350, 5400, 7500, 9500, 11650, 13600,
    #           15500, 17250, 19300]
    # rrange71 = [2100, 4350, 6350, 8350, 10600, 12550, 14500,
    #           16350, 18150, 20300]
    #
    # ##emg_qing_72.csv  Broken data
    # lrange72 = [1150, 3400, 5400, 7500, 9500, 11650, 13650,
    #           15500, 17350, 19350]
    # rrange72 = [2100, 4450, 6400, 8350, 10700, 12550, 14500,
    #           16450, 18250, 20300]
    #
    # ##emg_qing_73.csv  Broken data
    # lrange73 = [1100, 3350, 5400, 7400, 9500, 11600, 13600,
    #           15450, 17250, 19300]
    # rrange73 = [2100, 4250, 6350, 8350, 10600, 12550, 14500,
    #           16300, 18150, 20300]
    #
    # ##emg_qing_74.csv  Broken data
    # lrange74 = [1100, 3350, 5450, 7450, 9550, 11600, 13600,
    #           15450, 17250, 19300]
    # rrange74 = [2100, 4250, 6450, 8400, 10650, 12550, 14500,
    #           16350, 18250, 20500]
    #
    # ##emg_qing_75.csv  Broken data
    # lrange75 = [1100, 3350, 5450, 7450, 9550, 11600, 13600,
    #           15500, 17250, 19300]
    # rrange75 = [2100, 4350, 6450, 8400, 10650, 12450, 14600,
    #           16450, 18250, 20300]
    #
    # ##emg_qing_76.csv  Broken data
    # lrange76 = [1300, 3450, 5500, 7550, 9550, 11700, 13700,
    #           15570, 17350, 19400]
    # rrange76 = [2250, 4400, 6550, 8500, 10850, 12650, 14600,
    #           16450, 18250, 20350]
    #
    # ##emg_qing_77.csv  Broken data
    # lrange77 = [1200, 3400, 5500, 7500, 9500, 11700, 13650,
    #           15570, 17300, 19300]
    # rrange77 = [2150, 4400, 6400, 8500, 10750, 12550, 14500,
    #           16450, 18200, 20250]
    #
    # ##emg_qing_78.csv  Broken data
    # lrange78 = [1150, 3400, 5450, 7500, 9500, 11700, 13650,
    #           15570, 17300, 19300]
    # rrange78 = [2150, 4400, 6400, 8400, 10650, 12500, 14500,
    #           16450, 18200, 20300]
    #
    # ##emg_qing_79.csv  Broken data
    # lrange79 = [1150, 3400, 5450, 7500, 9600, 11650, 13650,
    #           15570, 17300, 19400]
    # rrange79 = [2150, 4400, 6400, 8400, 10650, 12500, 14500,
    #           16400, 18200, 20300]
    #
    # ##emg_qing_80.csv  Broken data
    # lrange80 = [1150, 3400, 5450, 7500, 9600, 11650, 13650,
    #           15500, 17300, 19300]
    # rrange80 = [2150, 4400, 6400, 8400, 10650, 12500, 14500,
    #           16450, 18300, 20300]
    #
    # ##emg_qing_81.csv  Broken data
    # lrange81 = [1100, 3300, 5350, 7500, 9600, 11550, 13650,
    #           15500, 17300, 19300]
    # rrange81 = [2100, 4350, 6400, 8400, 10650, 12500, 14500,
    #           16400, 18200, 20350]
    #
    # ##emg_qing_82.csv  Broken data
    # lrange82 = [1100, 3300, 5350, 7500, 9600, 11550, 13650,
    #           15500, 17350, 19300]
    # rrange82 = [2100, 4350, 6400, 8400, 10650, 12550, 14500,
    #           16400, 18200, 20350]
    #
    # ##emg_qing_83.csv  Broken data
    # lrange83 = [1150, 3400, 5450, 7500, 9600, 11650, 13650,
    #           15550, 17300, 19300]
    # rrange83 = [2150, 4350, 6400, 8400, 10650, 12500, 14500,
    #           16400, 18250, 20350]
    #
    # ##emg_qing_84.csv  Broken data
    # lrange84 = [1100, 3300, 5350, 7500, 9500, 11650, 13650,
    #           15500, 17300, 19400]
    # rrange84 = [2100, 4350, 6400, 8400, 10650, 12500, 14500,
    #           16400, 18200, 20350]
    #
    #
    # ##emg_qing_85.csv  Broken data
    # lrange85 = [1100, 3350, 5400, 7500, 9600, 11750, 13650,
    #           15700, 17300, 19300]
    # rrange85 = [2150, 4450, 6450, 8450, 10650, 12600, 14600,
    #           16500, 18300, 20350]
    #
    # ##emg_qing_86.csv  Broken data
    # lrange86 = [1150, 3350, 5350, 7550, 9700, 11600, 13650,
    #           15500, 17300, 19300]
    # rrange86 = [2100, 4350, 6400, 8400, 10650, 12500, 14500,
    #           16400, 18200, 20400]
    #
    # ##emg_qing_87.csv  Broken data
    # lrange87 = [1250, 3500, 5600, 7550, 9650, 11750, 13700,
    #           15550, 17400, 19400]
    # rrange87 = [2250, 4450, 6500, 8500, 10650, 12500, 14650,
    #           16500, 18300, 20450]
    #
    # ##emg_qing_88.csv  Broken data
    # lrange88 = [1100, 3400, 5450, 7500, 9600, 11650, 13650,
    #           15550, 17300, 19300]
    # rrange88 = [2100, 4350, 6400, 8400, 10650, 12600, 14550,
    #           16500, 18300, 20350]
    #
    # ##emg_qing_89.csv  Broken data
    # lrange89 = [1150, 3400, 5450, 7500, 9600, 11700, 13650,
    #           15550, 17300, 19400]
    # rrange89 = [2100, 4450, 6500, 8400, 10650, 12750, 14550,
    #           16500, 18350, 20350]
    #
    # ##emg_qing_90.csv  Broken data
    # lrange90 = [1100, 3400, 5450, 7500, 9600, 11650, 13650,
    #           15550, 17300, 19400]
    # rrange90 = [2100, 4350, 6400, 8400, 10600, 12500, 14550,
    #           16400, 18200, 20350]
    #
    # ##emg_qing_91.csv  Broken data
    # lrange91 = [1100, 3400, 5450, 7500, 9500, 11650, 13650,
    #           15550, 17300, 19300]
    # rrange91 = [2100, 4350, 6400, 8400, 10550, 12400, 14500,
    #           16400, 18300, 20250]
    #
    # ##emg_qing_92.csv  Broken data
    # lrange92 = [1150, 3400, 5450, 7500, 9550, 11650, 13650,
    #           15500, 17300, 19300]
    # rrange92 = [2100, 4400, 6400, 8400, 10650, 12500, 14500,
    #           16500, 18300, 20300]
    #
    # ##emg_qing_93.csv  Broken data
    # lrange93 = [1100, 3400, 5450, 7500, 9500, 11650, 13650,
    #           15550, 17500, 19250]
    # rrange93 = [2150, 4450, 6400, 8400, 10600, 12600, 14650,
    #           16500, 18300, 20350]
    #
    # ##emg_qing_94.csv  Broken data
    # lrange94 = [1200, 3400, 5500, 7500, 9500, 11650, 13650,
    #           15600, 17300, 19500]
    # rrange94 = [2150, 4400, 6400, 8450, 10650, 12600, 14600,
    #           16450, 18300, 20450]
    #
    # ##emg_qing_95.csv  Broken data
    # lrange95 = [1500, 3700, 5700, 7800, 9800, 11950, 13950,
    #           15800, 17550, 19700]
    # rrange95 = [2450, 4700, 6750, 8750, 10900, 12600, 14600,
    #           16750, 18550, 20650]
    #
    # ##emg_qing_96.csv  Broken data
    # lrange96 = [1200, 3400, 5500, 7500, 9500, 11650, 13650,
    #           15500, 17300, 19300]
    # rrange96 = [2150, 4400, 6400, 8450, 10650, 12600, 14600,
    #           16450, 18300, 20350]
    #
    # ##emg_qing_97.csv  Broken data
    # lrange97 = [1200, 3400, 5500, 7500, 9500, 11650, 13650,
    #           15600, 17400, 19400]
    # rrange97 = [2150, 4400, 6450, 8450, 10650, 12600, 14600,
    #           16550, 18300, 20400]
    #
    # ##emg_qing_98.csv  Broken data
    # lrange98 = [1100, 3400, 5450, 7500, 9500, 11650, 13650,
    #           15600, 17350, 19350]
    # rrange98 = [2150, 4450, 6450, 8450, 10650, 12500, 14550,
    #           16400, 18300, 20350]
    #
    # ##emg_qing_99.csv  Broken data
    # lrange99 = [1200, 3400, 5550, 7550, 9500, 11650, 13650,
    #           15600, 17400, 19350]
    # rrange99 = [2150, 4450, 6500, 8550, 10650, 12600, 14600,
    #           16450, 18400, 20450]
    #
    # ##emg_qing_100.csv  Broken data
    # lrange100 = [1150, 3400, 5400, 7400, 9500, 11650, 13650,
    #           15600, 17200, 19300]
    # rrange100 = [2100, 4400, 6400, 8450, 10650, 12600, 14400,
    #           16450, 18200, 20350]
    #
    # ##emg_qing_101.csv  Broken data
    # lrange101 = [1150, 3400, 5400, 7400, 9500, 11700, 13650,
    #           15600, 17300, 19300]
    # rrange101 = [2100, 4500, 6400, 8450, 10650, 12550, 14500,
    #           16450, 18200, 20350]
    #
    # ##emg_qing_102.csv  Broken data
    # lrange102 = [1150, 3400, 5450, 7450, 9500, 11650, 13650,
    #           15550, 17300, 19300]
    # rrange102 = [2150, 4400, 6400, 8450, 10650, 12600, 14500,
    #           16550, 18250, 20350]
    #
    # ##emg_qing_103.csv  Broken data
    # lrange103 = [1200, 3450, 5400, 7400, 9500, 11650, 13650,
    #           15500, 17250, 19250]
    # rrange103 = [2150, 4600, 6450, 8450, 10650, 12600, 14600,
    #           16450, 18300, 20450]
    #
    # ##emg_qing_104.csv  Broken data
    # lrange104 = [1150, 3300, 5400, 7400, 9500, 11650, 13650,
    #           15550, 17200, 19300]
    # rrange104 = [2000, 4400, 6400, 8450, 10700, 12650, 14500,
    #           16450, 18200, 20300]
    #
    # ##emg_qing_105.csv  Broken data
    # lrange105 = [1400, 3700, 5700, 7700, 9700, 11850, 13850,
    #           15700, 17550, 19500]
    # rrange105 = [2400, 4700, 6750, 8700, 10900, 12700, 14700,
    #           16550, 18550, 20600]
    #
    # ##emg_qing_106.csv  Broken data
    # lrange106 = [1150, 3400, 5400, 7450, 9600, 11650, 13650,
    #           15550, 17400, 19300]
    # rrange106 = [2100, 4400, 6400, 8450, 10700, 12650, 14500,
    #           16450, 18300, 20300]
    #
    # ##emg_qing_107.csv  Broken data
    # lrange107 = [1150, 3350, 5400, 7500, 9500, 11650, 13650,
    #           15550, 17250, 19300]
    # rrange107 = [2200, 4500, 6500, 8450, 10700, 12650, 14500,
    #           16450, 18250, 20350]
    #
    # ##emg_qing_108.csv  Broken data
    # lrange108 = [1400, 3600, 5700, 7700, 9700, 11850, 13850,
    #           15700, 17550, 19550]
    # rrange108 = [2350, 4700, 6650, 8700, 10900, 12800, 14700,
    #           16650, 18550, 20550]
    #
    # ##emg_qing_109.csv  Broken data
    # lrange109 = [1150, 3350, 5400, 7500, 9500, 11700, 13650,
    #           15550, 17250, 19200]
    # rrange109 = [2200, 4500, 6500, 8450, 10700, 12650, 14550,
    #           16500, 18300, 20350]
    #
    # ##emg_qing_110.csv  Broken data
    # lrange110 = [1150, 3350, 5450, 7500, 9500, 11650, 13650,
    #           15550, 17350, 19300]
    # rrange110 = [2200, 4500, 6500, 8450, 10700, 12450, 14550,
    #           16500, 18300, 20450]
    #
    # ##emg_qing_111.csv  Broken data
    # lrange111 = [1250, 3350, 5450, 7500, 9500, 11650, 13650,
    #           15550, 17350, 19300]
    # rrange111 = [2300, 4500, 6500, 8450, 10700, 12550, 14550,
    #           16500, 18300, 20350]
    #
    # ##emg_qing_112.csv  Broken data
    # lrange112 = [1400, 3600, 5650, 7650, 9700, 11850, 13850,
    #           15700, 17550, 19450]
    # rrange112 = [2350, 4600, 6650, 8600, 10900, 12700, 14700,
    #           16600, 18450, 20500]
    #
    # ##emg_qing_113.csv  Broken data
    # lrange113 = [1250, 3550, 5650, 7700, 9700, 11850, 13750,
    #           15700, 17500, 19400]
    # rrange113 = [2300, 4500, 6550, 8500, 10700, 12650, 14650,
    #           16500, 18350, 20400]
    #
    # ##emg_qing_114.csv  Broken data
    # lrange114 = [1250, 3350, 5450, 7500, 9500, 11650, 13650,
    #           15550, 17350, 19300]
    # rrange114 = [2200, 4500, 6500, 8450, 10700, 12550, 14550,
    #           16500, 18300, 20350]
    #
    # ##emg_qing_115.csv  Broken data
    # lrange115 = [1250, 3450, 5450, 7500, 9500, 11650, 13650,
    #           15550, 17350, 19300]
    # rrange115 = [2200, 4500, 6500, 8450, 10700, 12550, 14550,
    #           16500, 18300, 20350]
    #
    # ##emg_qing_116.csv  Broken data
    # lrange116 = [1250, 3400, 5450, 7500, 9500, 11650, 13650,
    #           15550, 17350, 19300]
    # rrange116 = [2200, 4300, 6500, 8450, 10700, 12550, 14550,
    #           16500, 18300, 20350]
    #
    # ##emg_qing_117.csv  Broken data
    # lrange117 = [1250, 3400, 5550, 7600, 9700, 11800, 13750,
    #           15650, 17450, 19400]
    # rrange117 = [2300, 4500, 6700, 8650, 10800, 12650, 14750,
    #           16650, 18400, 20350]
    #
    # #emg_qing_118.csv  Broken data
    # lrange118 = [1250, 3500, 5450, 7500, 9500, 11650, 13700,
    #           15750, 17350, 19300]
    # rrange118 = [2200, 4500, 6600, 8450, 10750, 12600, 14600,
    #           16600, 18300, 20400]
    #
    # ##emg_qing_119.csv  Broken data
    # lrange119 = [1150, 3350, 5450, 7500, 9500, 11650, 13650,
    #           15450, 17350, 19300]
    # rrange119 = [2100, 4400, 6400, 8450, 10700, 12550, 14450,
    #           16300, 18300, 20250]
    #
    # ##emg_qing_120.csv  Broken data
    # lrange120 = [1250, 3550, 5500, 7600, 9600, 11750, 13750,
    #           15650, 17400, 19400]
    # rrange120 = [2200, 4450, 6600, 8550, 10750, 12650, 14650,
    #           16550, 18400, 20400]
    #
    # ##emg_qing_121.csv  Broken data
    # lrange121 = [1150, 3400, 5450, 7500, 9500, 11650, 13650,
    #           15550, 17350, 19350]
    # rrange121 = [2150, 4500, 6500, 8450, 10700, 12550, 14550,
    #           16500, 18300, 20350]

    # ##emg_qing_130.csv  Broken data
    # lrange = [1150, 3400, 5400, 7400, 9500, 11650, 13650,
    #           15400, 17350, 19250]
    # rrange = [2200, 4400, 6450, 8500, 10700, 12600, 14550,
    #           16450, 18300, 20300]
    #
    # ##emg_qing_131.csv  Broken data
    # lrange = [1150, 3400, 5400, 7450, 9450, 11650, 13650,
    #           15450, 17350, 19250]
    # rrange = [2100, 4400, 6650, 8500, 10700, 12550, 14550,
    #           16450, 18300, 20300]

    # ##emg_qing_132.csv  Broken data
    # lrange = [1200, 3400, 5400, 7450, 9450, 11650, 13650,
    #           15500, 17350, 19200]
    # rrange = [2050, 4400, 6450, 8500, 10700, 12550, 14550,
    #           16400, 18250, 20250]

    # ##emg_qing_133.csv  Broken data
    # lrange = [1200, 3400, 5400, 7450, 9450, 11650, 13650,
    #           15500, 17250, 19300]
    # rrange = [2050, 4400, 6450, 8500, 10600, 12550, 14750,
    #           16400, 18250, 20250]

    # ##emg_qing_134.csv  Broken data
    # lrange = [1400, 3600, 5800, 7700, 9850, 11850, 13850,
    #           15600, 17550, 19550]
    # rrange = [2250, 4500, 6650, 8700, 10900, 12800, 14700,
    #           16580, 18490, 20500]

    # ##emg_qing_135.csv  Broken data
    # lrange = [1200, 3400, 5400, 7450, 9450, 11650, 13650,
    #           15500, 17350, 19200]
    # rrange = [2050, 4400, 6450, 8500, 10700, 12650, 14550,
    #           16400, 18250, 20250]
    #
    # ##emg_qing_136.csv  Broken data
    # lrange = [1200, 3400, 5400, 7450, 9450, 11650, 13650,
    #           15500, 17350, 19300]
    # rrange = [2150, 4400, 6400, 8500, 10700, 12550, 14550,
    #           16400, 18250, 20250]

    # ##emg_qing_137.csv  Broken data
    # lrange = [1300, 3400, 5400, 7450, 9450, 11650, 13650,
    #           15500, 17350, 19250]
    # rrange = [2250, 4500, 6450, 8350, 10750, 12600, 14550,
    #           16400, 18250, 20250]

    # ##emg_qing_138.csv  Broken data
    # lrange = [1200, 3400, 5400, 7450, 9450, 11650, 13650,
    #           15500, 17350, 19100]
    # rrange = [2100, 4400, 6450, 8500, 10700, 12550, 14550,
    #           16400, 18250, 20250]
    #
    # ##emg_qing_139.csv  Broken data
    # lrange = [1150, 3400, 5400, 7450, 9450, 11650, 13650,
    #           15500, 17350, 19100]
    # rrange = [2100, 4650, 6450, 8500, 10700, 12550, 14550,
    #           16500, 18250, 20250]

    # ##emg_qing_140.csv  Broken data
    # lrange = [1200, 3400, 5400, 7450, 9450, 11650, 13650,
    #           15500, 17350, 19300]
    # rrange = [2100, 4400, 6450, 8500, 10700, 12650, 14550,
    #           16400, 18250, 20250]


    avpara = 700
    procoefficient = 0.3
    # skewing = 250   #校正移动平均的偏移量
    # action = ['AG', 'CH', 'OH', 'EH', 'FH', 'FG', 'TM', 'TRF', 'TLF', 'GF']  # ## 绘制单个通道单个动作的图 ##
    # # Time_domain #
    # Single_Channel_Single_Action(0, 0, action, num, t, time_seg, fs, emg_seg, nfft, noverlap)
    # Single_Channel_Single_Action(1, 0, action, num, t, time_seg, fs, emg_seg, nfft, noverlap)
    num = 2 #测试者代号
    t = 20 #测试者测试次数

    ## 提取数据 ##
    print "######## 提取数据部分 ########"
    rows = loaddata('emg_qing_140.csv')

    ## 求出每一行数据的平均值 ##
    rows_mean = AddMeanChannel(rows)
    ## 求出每一行数据的极大值 ##
    rows_in = AddMaxChannel(rows,rows_mean)
    ## 提取每一个通道
    cols = ToColumns(rows_in)

    ## 去噪 ##
    print "######## 数据去噪部分 ########"
    emg_after_denoise = []
    for i in range(len(cols)-1):
        y = butter_highpass_filter(cols[i], highcut, fs, order=6)
        y_pli = pli_remove(fs, f0, y)
        y_ti = TI_Denoise(y_pli,'sym8',10,1,10)
        emg_after_denoise.append(y_ti)
    print ">>去噪后的数据列数",':',len(emg_after_denoise)

    ## 分割 ##
    print "############ 数据分割部分 ###########"
    # 平滑 #
    mavg = AllMovingAverage(emg_after_denoise, avpara)
    # Time_Domain_Plot(cols[10],mavg[0])
    # plt.show()
    # print mavg
    print ">>平滑后的数据列数",':', len(mavg)

    time_seg, emg_seg = AllCutEmg(lrange, rrange, emg_after_denoise, cols[10])


    plt.figure()
    plt.plot(cols[10],emg_after_denoise[8])
    for j in range(len(lrange)):
        plt.axvline(cols[10][lrange[j]])
        plt.axvline(cols[10][rrange[j]])
    mng = plt.get_current_fig_manager()
    # mng.window.showMaximized()
    plt.show()

    # for i in range(len(emg_after_denoise)):
    #     plt.figure(i)
    #     plt.plot(cols[10],emg_after_denoise[i])
    #     for j in range(len(lrange)):
    #         plt.axvline(cols[10][lrange[j]])
    #         plt.axvline(cols[10][rrange[j]])
    #     mng = plt.get_current_fig_manager()
    #     mng.window.showMaximized()
    #     plt.show()


    # ss, tt = AllCutData(lrange, emg_after_denoise, cols[10])
    # print len(ss[0])


    # plt.figure()
    # for i in range(len(s)):
    #     Spectrogram(nfft, noverlap, fs, s[i])
    #     plt.show()
    # for i in range(len(emg_after_denoise)):
    #     plt.figure(i+1)
    #     plt.plot(cols[10], emg_after_denoise[i])
    #     for j in range(len(final_rightbound_all_channel[i])):
    #         plt.axvline(cols[10][final_rightbound_all_channel[i][j]])
    #         plt.axvline(cols[10][final_leftbound_all_channel[i][j]])
    # plt.show()


    # ## 创建图片存放文件夹 ##
    # MakeDir()
    #
    # # 绘制所有通道的图 ##
    # # # Time_Domain #
    # AllChannels(0, 0, cols[10], fs, emg_after_denoise, nfft, noverlap)
    # AllChannels(1, 0, cols[10], fs, emg_after_denoise, nfft, noverlap)
    # # Spectrum #
    # AllChannels(0, 1, cols[10], fs, emg_after_denoise, nfft, noverlap)
    # AllChannels(1, 1, cols[10], fs, emg_after_denoise, nfft, noverlap)
    # # Spectrogram #
    # AllChannels(0, 2, cols[10], fs, emg_after_denoise, nfft, noverlap)
    # AllChannels(1, 2, cols[10], fs, emg_after_denoise, nfft, noverlap)
    #
    # # 绘制单个通道单个动作的图 ##
    # # Time_domain #
    # Single_Channel_Single_Action(0, 0, action, num, t, time_seg, fs, emg_seg, nfft, noverlap)
    # Single_Channel_Single_Action(1, 0, action, num, t, time_seg, fs, emg_seg, nfft, noverlap)
    #
    #
    # # Spectrum #
    # Single_Channel_Single_Action(0, 1, action, num, t, time_seg, fs, emg_seg, nfft, noverlap)
    # Single_Channel_Single_Action(1, 1, action, num, t, time_seg, fs, emg_seg, nfft, noverlap)
    # # Spectrogram #
    # Single_Channel_Single_Action(0, 2, action, num, t, time_seg, fs, emg_seg, nfft, noverlap)
    # Single_Channel_Single_Action(1, 2, action, num, t, time_seg, fs, emg_seg, nfft, noverlap)
    #
    # ## 绘制九个图在一个figures上 ##
    # Nine_In_One_Ave(0, action, num, t, fs, emg_seg, nfft, noverlap)
    # Nine_In_One_Ave(1, action, num, t, fs, emg_seg, nfft, noverlap)
    #
    # Nine_In_One_Max(0, action, num, t, fs, emg_seg, nfft, noverlap)
    # Nine_In_One_Max(1, action, num, t, fs, emg_seg, nfft, noverlap)


main()
