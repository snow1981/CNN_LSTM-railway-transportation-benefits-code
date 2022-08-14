# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 10:27:01 2021

@author: xxj
"""
import math

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import matplotlib.pyplot as plt
import os
import operator
from functools import reduce
#%matplotlib inline
from sklearn.preprocessing import MinMaxScaler
import time


df=pd.read_csv('lx.csv')
data_train =df.iloc[:int(df.shape[0] * 0.7), :]
data_test = df.iloc[int(df.shape[0] * 0.7):, :]
#print(data_train.shape, data_test.shape)
#print(data_test)




#print(data_train)
from tensorflow import keras
from tensorflow.keras.layers import Input,Dense,LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.optimizers import Nadam
output_dim = 1
batch_size = 48
epochs = 150
seq_len = 5
#hidden_size = 128
TIME_STEPS = 5
INPUT_DIM = 13
lstm_units = 130
def rmse_value(y_true, y_pred):
    n = len(y_true)
    mse = (sum(np.square(y_true - y_pred))/n)**0.5
    return mse

def mape_value(y_true, y_pred):
    n = len(y_pred)
    mape = sum(np.abs((y_true - y_pred)/y_true))/n*100
    return mape

def mae_value(y_true, y_pred):
    n = len(y_pred)
    mae = sum(np.abs(y_true - y_pred))/n
    return mae
#标准化

data_train=np.abs(data_train-np.min(data_train,axis=0))/(np.max(data_train,axis=0)-np.min(data_train,axis=0))  #标准化
min=np.min(data_test,axis=0)
max=np.max(data_test,axis=0)
data_test=np.abs(data_test-min)/(max-min)  #标准化

X_train = np.array([data_train.values[i : i + seq_len, :] for i in range(data_train.shape[0] - seq_len)])
y_train = np.array([data_train.values[i + seq_len, 12] for i in range(data_train.shape[0] - seq_len)])
X_test = np.array([data_test.values[i : i + seq_len, :] for i in range(data_test.shape[0] - seq_len)])
y_test = np.array([data_test.values[i + seq_len, 12] for i in range(data_test.shape[0] - seq_len)])
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print(y_test)



inputs = Input(shape=(TIME_STEPS, INPUT_DIM))

x = Conv1D(filters = 64, kernel_size = 2, activation = 'relu')(inputs) #卷积核的数目64，卷积核2
x = MaxPooling1D(pool_size = 2)(x)
x = Dropout(0.2)(x)
print(x.shape)

lstm_out = LSTM(lstm_units, activation='sigmoid')(x)
print(lstm_out.shape)
output = Dense(1, activation='sigmoid')(lstm_out)
model = Model(inputs=inputs, outputs=output)
print(model.summary())

checkpoint_save_path = "./checkpoint/lxg.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.009), metrics=['accuracy'])
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=False, validation_data=(X_test, y_test), validation_freq=1, callbacks=[cp_callback])
y_pred = model.predict(X_test)

# 反标准化数据 --- 目的是保证MSE,MAPE以及图片数值的准确性
y_test=np.array(y_test)*(max[12]-min[12])+min[12]

y_pred=np.array(y_pred)*(max[12]-min[12])+min[12]

#设置字体
plt.rcParams['font.sans-serif']=['FangSong']
plt.rcParams['axes.unicode_minus']=False

plt.rcParams['xtick.direction'] = 'in'#将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内


fig = plt.figure(figsize=(25, 8))  # dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80

#绘图不要框
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.plot(range(len(y_pred)), y_pred, 'k--',label='Prediction value',linewidth=0.8)

plt.plot(range(len(y_test)), y_test, 'k',label='Test value',linewidth=0.8)

x = range(0,16,1)
plt.xticks(x,('2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020'),rotation=60,fontsize=12,fontproperties="Times New Roman")
plt.yticks(fontsize=12,fontproperties="Times New Roman")


plt.legend(fontsize=12) # 显示图例
 
plt.xlabel('Year',fontsize=12,fontproperties="Times New Roman")

plt.ylabel('GDP(Unit: RMB 100 million )',fontsize=12,fontproperties="Times New Roman")
plt.ylabel('GDP(Unit: RMB 100 million )',fontsize=12,fontproperties="Times New Roman")
plt.rcParams['savefig.dpi'] = 300 # 图片像素
plt.rcParams['figure.dpi'] = 300 # 分辨率
plt.savefig('Figure 21 Prediction results of the CNN-LSTM model.pdf')



np.savetxt("data.txt",y_test, fmt="%.4f");
np.savetxt("data1.txt",y_pred, fmt="%.4f");

import numpy

fp=open('data.txt','r')
ls=[]
for line in fp:
    line=line.strip('\n')   #将\n去掉
    ls.append(line.split(' '))   #将空格作为分隔符将一个字符切割成一个字符数组

fp.close()
ls=numpy.array(ls,dtype=float)   #将其转换成numpy的数组，并定义数据类型为float
print(ls)


import numpy

fp=open('data1.txt','r')
ls1=[]
for line1 in fp:
    line1=line1.strip('\n')   #将\n去掉
    ls1.append(line1.split(' '))   #将空格作为分隔符将一个字符切割成一个字符数组

fp.close()
ls1=numpy.array(ls1,dtype=float)   #将其转换成numpy的数组，并定义数据类型为float
print(ls1)


mse_value = rmse_value(ls, ls1)
print ("RMSE on test data",format("%d"%mse_value))
mape = mape_value(ls, ls1)
print ("MAPE on test data",format(mape))
mae = mae_value(ls, ls1)
print ("MAE on test data",format(mae))




