import numpy as np
# import scipy.interpolate 
import math
import os
from utils import *
from einops import rearrange, reduce

import xgboost as xg
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random

    
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
    else:
        payloadBits_per_OFDM = K
        pilotValue = np.ones((64,1))
    SNRdb = config.SNR  # signal to noise-ratio in dB at the receiver 
    Clipping_Flag = config.Clipping 


    
    CP_flag = config.with_CP_flag

    

    # ---------- training ----------
    
    input_samples = []
    input_labels = []

    block_size = config.train_batch_size/config.num_block

    for i in range(0, config.train_batch_size):
        channel_response = np.random.normal(0, (1/32)**0.5, 16) + 1j*np.random.normal(0, (1/32)**0.5, 16)

        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
        if config.qpsk:
            signal_output = qpsk_simulate_vary_pilot_with_insert_ML(bits,channel_response,SNRdb, mu, CP_flag, K, P, CP, pilotValue,pilotCarriers, dataCarriers,Clipping_Flag)
        else:
            signal_output, para = bpsk_simulate(bits,channel_response,SNRdb, mu, CP_flag, K, P, CP, pilotValue,pilotCarriers, dataCarriers,Clipping_Flag)

        input_labels.append(bits[config.pred_range])
        input_samples.append(signal_output[config.pred_range])
    train_feature = np.asarray(input_samples)

    batch_y = np.asarray(input_labels)

    
    # ---------- testing ----------

    input_labels_test = []
    input_samples_test = []

    for i in range(0, config.test_batch_size):

        channel_response = np.random.normal(0, (1/32)**0.5, 16) + 1j*np.random.normal(0, (1/32)**0.5, 16)

        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
        if config.qpsk:                        
            signal_output = qpsk_simulate_vary_pilot_with_insert_ML(bits,channel_response,SNRdb,mu, CP_flag, K, P, CP, pilotValue,pilotCarriers, dataCarriers,Clipping_Flag )
        else:
            signal_output, para = bpsk_simulate(bits,channel_response,SNRdb,mu, CP_flag, K, P, CP, pilotValue,pilotCarriers, dataCarriers,Clipping_Flag)
        
        input_labels_test.append(bits[config.pred_range])
        input_samples_test.append(signal_output[config.pred_range])
    test_feature = np.asarray(input_samples_test)

    test_batch_y = np.asarray(input_labels_test)

    avg_training_result = [0, 0]
    avg_testing_result = [0, 0]
    total_training_time = 0


    train_pred = train_feature
    # print(train_pred.shape)
    test_pred = test_feature
    threshold = 0.5
    train_pred_binary = (train_pred >= threshold).astype(int)
    test_pred_binary = (test_pred >= threshold).astype(int)

    train_errors = np.sum(train_pred_binary != batch_y, axis=1)
    

    test_errors = np.sum(test_pred_binary != test_batch_y, axis=1)
    testing_BER = np.sum(test_errors)/test_pred_binary.shape[0]/test_pred_binary.shape[1]

    config.BER_realization = testing_BER
