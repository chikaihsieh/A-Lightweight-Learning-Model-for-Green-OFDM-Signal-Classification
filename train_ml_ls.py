import numpy as np
# import scipy.interpolate 
import math
import os
from utils import *
from einops import rearrange, reduce
from PixelHop_unit.ofdmhop_noprint import *
from PixelHop_unit.LAG import *
from PixelHop_unit.rft import *
import xgboost as xg
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random
def get_pixelhop_feature(batch_x, weight_name, channel_response, patch_num, AC_kernel_list, getK, useDC, add_bias):
    if patch_num != len(AC_kernel_list):
        raise Exception(f'mismatch between patch_num {patch_num} and len of AC_kernel_list {len(AC_kernel_list)}')
    weight_name_list = [f"{weight_name.split('.')[0]}{i}.{weight_name.split('.')[1]}" for i in range(patch_num)]
    patch_size = batch_x.shape[1] // patch_num
    ret_feature = None
    for patch in range(patch_num):
        batch = batch_x[:,patch*patch_size:(patch+1)*patch_size]
        feature = PixelHop_Unit(batch, num_AC_kernels=AC_kernel_list[patch], pad='reflect', weight_name=weight_name_list[patch], getK=getK, useDC=useDC, add_bias=add_bias)
        # if patch == 0:
        #     ret_feature = np.tile(np.fft.fft(channel_response, n=64).real.T, (batch_x.shape[0], 1))[:,None]
        # elif patch == 1:
        #     feature = np.tile(np.fft.fft(channel_response, n=64).imag.T, (batch_x.shape[0], 1))[:,None]
        #     ret_feature = np.concatenate((ret_feature, feature), axis=2)
        if patch == 0:
            ret_feature = feature
        else:
            ret_feature = np.concatenate((ret_feature, feature), axis=2)
    # print(f'---------  final shape after PixelHop: {ret_feature.shape}  ---------' )
    return ret_feature

def cal_mse_ber(pred, label, task):
    if task == 'classification':
        rmse = np.sqrt(np.linalg.norm(label-pred, ord=2))
        BER = 1 - np.mean(np.equal(pred, label).astype(np.float32))
    if task == 'regression':
        rmse = np.sqrt(np.linalg.norm(label-pred, ord=2))
        BER = 1 - np.mean(np.equal(np.sign(pred-0.5), np.sign(label-0.5).astype(np.float32)).astype(np.float32))

    return rmse, BER

# def regression_model(train_feature, test_feature, batch_y, test_batch_y, train_batch_size, test_batch_size):
#     train_dmatrix = xg.DMatrix(data = train_feature.reshape(train_batch_size, -1), label = batch_y)
#     test_dmatrix = xg.DMatrix(data = test_feature.reshape(test_batch_size, -1), label = test_batch_y)
#     param = {"booster":"gblinear", "objective":"reg:squarederror", "tree_method": "gpu_hist"}
#     xgb_r = xg.train(params = param, dtrain = train_dmatrix, num_boost_round = 100)
#     train_pred = xgb_r.predict(train_dmatrix)
#     test_pred = xgb_r.predict(test_dmatrix)
#     # print(test_pred)
#     # print(train_pred.shape, test_pred.shape)

#     return train_pred, test_pred

def regression_model(train_feature, train_labels, test_feature, n_estimators=400, colsample=1, max_depth=8, lr=0.35):
    bst = xg.XGBRegressor(n_estimators=n_estimators, 
                          max_depth=max_depth, 
                          gamma=0.1, 
                          tree_method='hist', 
                          gpu_id=0, 
                          colsample_bytree=colsample, 
                          learning_rate=lr, 
                          objective='binary:logistic',  # 使用回归的目标函数
                          seed=42, reg_lambda = 1000)
    
    start_time = time.time()
    bst.fit(train_feature, train_labels)
    # 结束计时
    end_time = time.time()
    
    # 计算训练时间
    training_time = end_time - start_time
    print(f"XGBoost 模型训练时间：{training_time} 秒")
    
    # 预测并返回结果
    train_pred = bst.predict(train_feature)
    test_pred = bst.predict(test_feature)
    return train_pred, test_pred, training_time

def classification_model(train_feature, train_labels, test_feature, n_estimators=250, colsample=1, max_depth=6, lr=0.01):
    bst = xg.XGBClassifier(n_estimators=n_estimators, 
                           max_depth=max_depth, 
                           gamma=0.5, 
                           tree_method='hist', 
                           gpu_id=0, 
                           colsample_bytree=colsample, 
                           learning_rate=lr, 
                           objective='binary:logistic',
                           seed=42)
    start_time = time.time()
    bst.fit(train_feature, train_labels)
        # 结束计时
    end_time = time.time()
    
    # 计算训练时间
    training_time = end_time - start_time
    print(f"XGBoost 模型训练时间：{training_time} 秒")
    train_pred = bst.predict(train_feature)
    test_pred = bst.predict(test_feature)
    return train_pred, test_pred, training_time

def classification_DNN(train_feature, train_labels, test_feature, hidden_units=[ 500, 250, 120], learning_rate=0.001, epochs=50):
    model = Sequential()
    for units in hidden_units:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(16, activation='sigmoid'))
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, decay=0.1)
    # 编译模型
    model.compile(optimizer=optimizer,
                  loss='MSE',
                  metrics=['mae'])
    start_time = time.time()
    # 训练模型
    
    model.fit(train_feature, train_labels, epochs=epochs, batch_size=256, shuffle=True)
    end_time = time.time()
    # 计算训练时间
    training_time = end_time - start_time
    print(f"DNN 模型训练时间：{training_time} 秒")
    # 使用模型进行预测
    train_pred = model.predict(train_feature)
    test_pred = model.predict(test_feature)

    return train_pred, test_pred, training_time

def ml_model(train_feature, test_feature, code):
    # 合并特征以形成复数数组
    Y_pilot_train = train_feature[:, 0] + 1j * train_feature[:, 64]
    Y_pilot_test = test_feature[:, 0] + 1j * test_feature[:, 64]
    Y_data_train = train_feature[:, 128] + 1j * train_feature[:, 192]
    Y_data_test = test_feature[:, 128] + 1j * test_feature[:, 192]
    
    # 计算 H
    H_train = Y_pilot_train / code
    H_test = Y_pilot_test / code
    
    # 定义 QPSK 的四个符号
    qpsk_symbols = np.array([(1+1j)/np.sqrt(2), (1-1j)/np.sqrt(2), (-1+1j)/np.sqrt(2), (-1-1j)/np.sqrt(2)])
    # 符号到位的映射
    symbol_to_bits = {
        (1+1j)/np.sqrt(2): [1, 1],
        (1-1j)/np.sqrt(2): [1, 0],
        (-1+1j)/np.sqrt(2): [0, 1],
        (-1-1j)/np.sqrt(2): [0, 0]
    }
    
    # 计算最接近的 QPSK 符号
    def closest_qpsk_symbol(y_data, H):
        # 计算每个样本对所有 QPSK 符号的距离
        distances = np.abs(y_data[:, np.newaxis] - H[:, np.newaxis] * qpsk_symbols)
        # 找到最近的符号的索引
        closest_indices = np.argmin(distances, axis=1)
        # 返回最接近的符号
        closest_symbols = qpsk_symbols[closest_indices]
        print(closest_symbols.shape)
        # 转换为位表示
        return np.array([symbol_to_bits[symbol] for symbol in closest_symbols])
    
    # 应用函数计算训练和测试数据的最接近的符号
    train_pred = closest_qpsk_symbol(Y_data_train, H_train)
    print(train_pred.shape)
    test_pred = closest_qpsk_symbol(Y_data_test, H_test)
    
    return train_pred, test_pred
    
def train(config):    
    K = config.Pilots
    CP = K//4
    P = config.Pilots # number of pilot carriers per OFDM block
    allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
    mu = 2
    CP_flag = config.with_CP_flag
    if P<K:
        pilotCarriers = allCarriers[::K//P] # Pilots is every (K/P)th carrier.
        dataCarriers = np.delete(allCarriers, pilotCarriers)
        
    else:   # K = P
        pilotCarriers = allCarriers
        dataCarriers = []

    if config.qpsk:
        # payloadBits_per_OFDM = K*mu  
        # bits = np.ones((128,1))
        # bits[1::2] = 0
        
        payloadBits_per_OFDM = 256-K*mu
        bits = np.ones((K*mu,1))
        pilotValue = Modulation(bits,mu)
        # pilotValue = np.ones(K,)
    else:
        payloadBits_per_OFDM = K
        pilotValue = np.ones((64,1))
    SNRdb = config.SNR  # signal to noise-ratio in dB at the receiver 
    Clipping_Flag = config.Clipping 


    
    CP_flag = config.with_CP_flag

    

    # ---------- training ----------
    
    input_samples = []
    input_labels = []

    for i in range(0, config.train_batch_size):
        channel_response = np.random.normal(0, (1/32)**0.5, 16) + 1j*np.random.normal(0, (1/32)**0.5, 16)
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
        if config.qpsk:
            signal_output, para = qpsk_simulate_vary_pilot_with_insert(bits,channel_response,SNRdb, mu, CP_flag, K, P, CP, pilotValue,pilotCarriers, dataCarriers,Clipping_Flag)
        else:
            signal_output, para = bpsk_simulate(bits,channel_response,SNRdb, mu, CP_flag, K, P, CP, pilotValue,pilotCarriers, dataCarriers,Clipping_Flag)

        input_labels.append(bits[config.pred_range])
        input_samples.append(signal_output)
    train_feature = np.asarray(input_samples)

    batch_y = np.asarray(input_labels)
    batch_y_signal = np.where(batch_y == 1, 2**(-0.5), -2**(-0.5))
    
    # ---------- testing ----------

    input_labels_test = []
    input_samples_test = []

    for i in range(0, config.test_batch_size):
        channel_response = np.random.normal(0, (1/32)**0.5, 16) + 1j*np.random.normal(0, (1/32)**0.5, 16)
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
        if config.qpsk:                        
            signal_output, para = qpsk_simulate_vary_pilot_with_insert(bits,channel_response,SNRdb,mu, CP_flag, K, P, CP, pilotValue,pilotCarriers, dataCarriers,Clipping_Flag)
        else:
            signal_output, para = bpsk_simulate(bits,channel_response,SNRdb,mu, CP_flag, K, P, CP, pilotValue,pilotCarriers, dataCarriers,Clipping_Flag)
        
        input_labels_test.append(bits[config.pred_range])
        input_samples_test.append(signal_output)
    test_feature = np.asarray(input_samples_test)

    test_batch_y = np.asarray(input_labels_test)

    avg_training_result = [0, 0]
    avg_testing_result = [0, 0]
    total_training_time = 0


    cla_model_input = [train_feature, batch_y, test_feature]
    train_pred, test_pred = ml_model(train_feature, test_feature, pilotValue[0])
    # 使用 NumPy 将小于0.5的值改为0，大于等于0.5的值改为1
    threshold = 0.5
    train_pred_binary = (train_pred >= threshold).astype(int)
    test_pred_binary = (test_pred >= threshold).astype(int)
    # 逐元素比较每个样本的预测结果是否与标签相匹配
    train_errors = np.sum(train_pred_binary != batch_y, axis=1)
    
    # 计算错误的数量
    training_BER = np.sum(train_errors)/train_pred_binary.shape[0]/train_pred_binary.shape[1]
    # 逐元素比较每个样本的预测结果是否与标签相匹配
    test_errors = np.sum(test_pred_binary != test_batch_y, axis=1)
    
    # 计算错误的数量
    testing_BER = np.sum(test_errors)/test_pred_binary.shape[0]/test_pred_binary.shape[1]
    print(training_BER)


    config.BER_realization = testing_BER
