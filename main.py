import os
import numpy as np
from train_xgb_vary_pilot import train #change train_xxx can modify the train function
import pickle
# from test import test


class sysconfig(object):
    weight_name = 'pixelhop.pkl'
    Pilots = 64       # number of pilots
    train_batch_size = 3000000 # number of training data
    test_batch_size = 3000000 # number of testing data
    pred_range = np.arange(0,1) # set the index of OFDM subcarrier
    with_CP_flag = True 
    SNR = 25
    Clipping = False

    BER_realization = 0 # record the BER of maximum likelihood
    selection_loss = []
    num_block = 10000 # set the number of the block in block fading

    use_dc = True
    add_bias = True
    qpsk = True

if __name__ == '__main__':
    config = sysconfig()
    BER_list = []
    # maximum likelihood
    for i in range(1): # set the number of Monte Carlo
        train(config)
        BER_list.append(config.BER_realization)
    avg_BER = sum(BER_list) / len(BER_list)

    # DNN or GL
    # train(config)   
    