from __future__ import division
import numpy as np
# import scipy.interpolate 
# import tensorflow as tf
import math
import os
import time
from sklearn.decomposition import PCA
def print_something():
    print ('utils.py has been loaded perfectly')


def get_channel(Channel_idx, H_folder):
    channel_response_set_train = []
    for train_idx in range(Channel_idx['train'][0],Channel_idx['train'][1]):
        H_file = H_folder['train'] + str(train_idx) + '.txt'
        with open(H_file) as f:
            for line in f:
                numbers_str = line.split()
                numbers_float = [float(x) for x in numbers_str]
                h_response = np.asarray(numbers_float[0:int(len(numbers_float)/2)])+1j*np.asarray(numbers_float[int(len(numbers_float)/2):len(numbers_float)])
                channel_response_set_train.append(h_response)
                break
                
                
    channel_response_set_test = []
    for test_idx in range(Channel_idx['test'][0],Channel_idx['test'][1]):
        H_file = H_folder['test'] + str(test_idx) + '.txt'
        with open(H_file) as f:
            for line in f:
                numbers_str = line.split()
                numbers_float = [float(x) for x in numbers_str]
                h_response = np.asarray(numbers_float[0:int(len(numbers_float)/2)])+1j*np.asarray(numbers_float[int(len(numbers_float)/2):len(numbers_float)])
                channel_response_set_test.append(h_response)
                break

    return channel_response_set_train, channel_response_set_test



def Clipping (x, CL):
    sigma = np.sqrt(np.mean(np.square(np.abs(x))))
    CL = CL*sigma
    x_clipped = x  
    clipped_idx = abs(x_clipped) > CL
    x_clipped[clipped_idx] = np.divide((x_clipped[clipped_idx]*CL),abs(x_clipped[clipped_idx]))
    return x_clipped

def PAPR(x):
    Power = np.abs(x)**2
    PeakP = np.max(Power)
    AvgP = np.mean(Power)
    PAPR_dB = 10*np.log10(PeakP/AvgP)
    return PAPR_dB

def Modulation(bits,mu):
                                          
    bit_r = bits.reshape((int(len(bits)/mu), mu))                  
    return ((2*bit_r[:,0]-1)+1j*(2*bit_r[:,1]-1))/np.sqrt(2)                                    # This is just for QAM modulation


def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)

def addCP(OFDM_time, CP, CP_flag, mu, K):
    
    if CP_flag == False:
        # add noise CP
        bits_noise = np.random.binomial(n=1, p=0.5, size=(K*mu, ))
        codeword_noise = Modulation(bits_noise, mu)
        OFDM_data_nosie = codeword_noise
        OFDM_time_noise = np.fft.ifft(OFDM_data_nosie)
        cp = OFDM_time_noise[-CP:]
    else:
        cp = OFDM_time[-CP:]               # take the last CP samples ...
    #cp = OFDM_time[-CP:] 
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning

def channel(signal,channelResponse,SNRdb):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb/10)  
    noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
    return convolved + noise

def removeCP(signal, CP, K):
    return signal[CP:(CP+K)]

def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)


def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest

def get_payload(equalized):
    return equalized[dataCarriers]


def PS(bits):
    return bits.reshape((-1,))

def add_noise(codeword, SNR, qpsk=True):
    signal_power = np.mean(abs(codeword**2))
    sigma2 = signal_power * 10**(-SNR/10)  
    if qpsk:
        noise = np.sqrt(sigma2/2) * (np.random.randn(*codeword.shape)+1j*np.random.randn(*codeword.shape))
    else:
        noise = np.sqrt(sigma2) * np.random.randn(*codeword.shape)
    # print(codeword.dtype, noise.dtype)
    codeword += noise
    return codeword


def ofdm_simulate(codeword, channelResponse,SNRdb, mu, CP_flag, K, P, CP, pilotValue,pilotCarriers, dataCarriers,Clipping_Flag):  
    payloadBits_per_OFDM = mu*len(dataCarriers)
    # --- training inputs ----
    if P < K:
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))    
        QAM = Modulation(bits,mu)
        OFDM_data = np.zeros(K, dtype=complex)
        OFDM_data[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
        OFDM_data[dataCarriers] = QAM
    else:
        OFDM_data = pilotValue
    
    OFDM_time = IDFT(OFDM_data)  # (64,)
    OFDM_withCP = addCP(OFDM_time, CP, CP_flag, mu, K) # (80,)
    #OFDM_withCP = addCP(OFDM_time)
    OFDM_TX = OFDM_withCP
    if Clipping_Flag:
        OFDM_TX = Clipping(OFDM_TX,CR)                            # add clipping 
    OFDM_RX = channel(OFDM_TX, channelResponse, SNRdb)
    OFDM_RX_noCP = removeCP(OFDM_RX, CP,K)
    # OFDM_RX_noCP = DFT(OFDM_RX_noCP)
    #OFDM_RX_noCP = removeCP(OFDM_RX)
    # ----- target inputs ---
    symbol = np.zeros(K, dtype=complex)
    codeword_qam = Modulation(codeword,mu)
    if len(codeword_qam) != K:
	    print('length of code word is not equal to K, error !!')
    symbol = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
    OFDM_withCP_cordword = addCP(OFDM_time_codeword, CP, CP_flag, mu, K)
    #OFDM_withCP_cordword = addCP(OFDM_time_codeword)
    if Clipping_Flag:
        OFDM_withCP_cordword = Clipping(OFDM_withCP_cordword,CR) # add clipping 
    OFDM_RX_codeword = channel(OFDM_withCP_cordword, channelResponse,SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword,CP,K)
    # OFDM_RX_noCP_codeword = DFT(OFDM_RX_noCP_codeword)
    #OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword)
    return np.concatenate((np.concatenate((np.real(OFDM_RX_noCP),np.imag(OFDM_RX_noCP))), np.concatenate((np.real(OFDM_RX_noCP_codeword),np.imag(OFDM_RX_noCP_codeword))))), abs(channelResponse) #sparse_mask


def qpsk_simulate(codeword, channelResponse,SNRdb, mu, CP_flag, K, P, CP, pilotValue,pilotCarriers, dataCarriers,Clipping_Flag):  
    payloadBits_per_OFDM = mu*len(dataCarriers)
    # --- training inputs ----
    if P < K:
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))    
        QAM = Modulation(bits,mu)
        OFDM_data = np.zeros(K, dtype=complex)
        OFDM_data[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
        OFDM_data[dataCarriers] = QAM
    else:
        OFDM_data = pilotValue
    
    H = np.fft.fft(channelResponse, n=64)
    OFDM_RX_noCP = np.matmul(np.diag(H), OFDM_data)
    signal_power = np.mean(abs(OFDM_RX_noCP**2))
    sigma2 = signal_power * 10**(-SNRdb/10)  
    noise1 = np.sqrt(sigma2/2) * (np.random.randn(*OFDM_RX_noCP.shape)+1j*np.random.randn(*OFDM_RX_noCP.shape))
    OFDM_RX_noCP += noise1
    #OFDM_RX_noCP = removeCP(OFDM_RX)
    # ----- target inputs ---
    symbol = np.zeros(K, dtype=complex)
    codeword_qam = Modulation(codeword,mu)
    if len(codeword_qam) != K:
	    print('length of code word is not equal to K, error !!')
    symbol = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_RX_noCP_codeword = np.matmul(np.diag(H), OFDM_data_codeword)
    OFDM_RX_noCP_codeword = add_noise(OFDM_RX_noCP_codeword, SNRdb)

    return np.concatenate((np.concatenate((np.real(OFDM_RX_noCP),np.imag(OFDM_RX_noCP))), np.concatenate((np.real(OFDM_RX_noCP_codeword),np.imag(OFDM_RX_noCP_codeword))))), abs(channelResponse) #sparse_mask

def qpsk_simulate_AP(codeword, channelResponse,SNRdb, mu, CP_flag, K, P, CP, pilotValue,pilotCarriers, dataCarriers,Clipping_Flag):  
    payloadBits_per_OFDM = mu*len(dataCarriers)
    # --- training inputs ----
    if P < K:
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))    
        QAM = Modulation(bits,mu)
        OFDM_data = np.zeros(K, dtype=complex)
        OFDM_data[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
        OFDM_data[dataCarriers] = QAM
    else:
        OFDM_data = pilotValue
    
    H = np.fft.fft(channelResponse, n=64)
    OFDM_RX_noCP = np.matmul(np.diag(H), OFDM_data)
    signal_power = np.mean(abs(OFDM_RX_noCP**2))
    sigma2 = signal_power * 10**(-SNRdb/10)  
    noise1 = np.sqrt(sigma2/2) * (np.random.randn(*OFDM_RX_noCP.shape)+1j*np.random.randn(*OFDM_RX_noCP.shape))
    OFDM_RX_noCP += noise1
    #OFDM_RX_noCP = removeCP(OFDM_RX)
    # ----- target inputs ---
    symbol = np.zeros(K, dtype=complex)
    codeword_qam = Modulation(codeword,mu)
    if len(codeword_qam) != K:
	    print('length of code word is not equal to K, error !!')
    symbol = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_RX_noCP_codeword = np.matmul(np.diag(H), OFDM_data_codeword)
    OFDM_RX_noCP_codeword = add_noise(OFDM_RX_noCP_codeword, SNRdb)

    return np.concatenate((np.concatenate((np.abs(OFDM_RX_noCP),np.angle(OFDM_RX_noCP))), np.concatenate((np.abs(OFDM_RX_noCP_codeword),np.angle(OFDM_RX_noCP_codeword))))), abs(channelResponse)

def qpsk_simulate_vary_pilot(codeword, channelResponse,SNRdb, mu, CP_flag, K, P, CP, pilotValue,pilotCarriers, dataCarriers,Clipping_Flag):  
    payloadBits_per_OFDM = mu*len(dataCarriers)
    codeword_qam = Modulation(codeword,mu)
    codeword_qam_part1 = codeword_qam[0:64-K]
    codeword_qam_part2 = codeword_qam[64-K:]
    # --- training inputs ----
    if P < K:
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))    
        QAM = Modulation(bits,mu)
        OFDM_data = np.zeros(K, dtype=complex)
        OFDM_data[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
        OFDM_data[dataCarriers] = QAM
    else:
        OFDM_data = np.concatenate((pilotValue,  codeword_qam_part1)) 
    
    H = np.fft.fft(channelResponse, n=64)/np.sqrt(64)
    OFDM_RX_noCP = np.matmul(np.diag(H), OFDM_data)
    signal_power = np.mean(abs(OFDM_RX_noCP**2))
    sigma2 = signal_power * 10**(-SNRdb/10)  
    noise1 = np.sqrt(sigma2/2) * (np.random.randn(*OFDM_RX_noCP.shape)+1j*np.random.randn(*OFDM_RX_noCP.shape))
    OFDM_RX_noCP += noise1
    #OFDM_RX_noCP = removeCP(OFDM_RX)
    # ----- target inputs ---
    symbol = np.zeros(128-K, dtype=complex)
    
    if len(codeword_qam_part2) != len(OFDM_data):
	    print('length of code word is not equal to K, error !!')
    symbol = codeword_qam_part2
    OFDM_data_codeword = symbol
    OFDM_RX_noCP_codeword = np.matmul(np.diag(H), OFDM_data_codeword)
    OFDM_RX_noCP_codeword = add_noise(OFDM_RX_noCP_codeword, SNRdb)

    return  np.concatenate((np.concatenate((np.real(OFDM_RX_noCP),np.imag(OFDM_RX_noCP))), np.concatenate((np.real(OFDM_RX_noCP_codeword),np.imag(OFDM_RX_noCP_codeword))))), abs(channelResponse) #sparse_mask

def qpsk_simulate_vary_pilot_with_pilot(codeword, channelResponse,SNRdb, mu, CP_flag, K, P, CP, pilotValue,pilotCarriers, dataCarriers,Clipping_Flag):  
    payloadBits_per_OFDM = mu*len(dataCarriers)
    codeword_qam = Modulation(codeword,mu)
    codeword_qam_part1 = codeword_qam[0:64-K]
    codeword_qam_part2 = codeword_qam[64-K:]
    # --- training inputs ----
    if P < K:
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))    
        QAM = Modulation(bits,mu)
        OFDM_data = np.zeros(K, dtype=complex)
        OFDM_data[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
        OFDM_data[dataCarriers] = QAM
    else:
        # 計算每個位置應插入的數值
        long_array = codeword_qam_part1
        short_array = pilotValue
        insert_len = (len(long_array)+len(short_array))//len(short_array)
        insert_len2 = len(long_array)//len(short_array)
        # 將兩個 array 合併成一個新的 array
        result_array = np.empty(len(long_array) + len(short_array), dtype=long_array.dtype)
        result_array[0::insert_len] = short_array     
        # 对于 long_array 剩下的位置，按顺序安插
        for i in range(len(short_array)):
            result_array[i*insert_len+1 :i*insert_len+insert_len2+1] = long_array[i*insert_len2:(i+1)*insert_len2]
        OFDM_data = result_array
    
    H = np.fft.fft(channelResponse, n=64)/np.sqrt(64)
    OFDM_RX_noCP = np.matmul(np.diag(H), OFDM_data)
    signal_power = np.mean(abs(OFDM_RX_noCP**2))
    sigma2 = signal_power * 10**(-SNRdb/10)  
    noise1 = np.sqrt(sigma2/2) * (np.random.randn(*OFDM_RX_noCP.shape)+1j*np.random.randn(*OFDM_RX_noCP.shape))
    OFDM_RX_noCP += noise1
    #OFDM_RX_noCP = removeCP(OFDM_RX)
    # ----- target inputs ---
    
    if len(codeword_qam_part2) != len(OFDM_data):
	    print('length of code word is not equal to K, error !!')
    symbol = codeword_qam_part2  
    OFDM_data_codeword = symbol
    OFDM_RX_noCP_codeword = np.matmul(np.diag(H), OFDM_data_codeword)
    OFDM_RX_noCP_codeword = add_noise(OFDM_RX_noCP_codeword, SNRdb)
    return  np.concatenate((np.concatenate((np.concatenate((np.real(OFDM_RX_noCP),np.imag(OFDM_RX_noCP))), np.concatenate((np.real(OFDM_RX_noCP_codeword),np.imag(OFDM_RX_noCP_codeword))))), np.concatenate((np.real(pilotValue[0:15]),np.imag(pilotValue[0:15]))) )), abs(channelResponse) #sparse_mask

def qpsk_simulate_vary_pilot_AP(codeword, channelResponse,SNRdb, mu, CP_flag, K, P, CP, pilotValue,pilotCarriers, dataCarriers,Clipping_Flag):  
    payloadBits_per_OFDM = mu*len(dataCarriers)
    codeword_qam = Modulation(codeword,mu)
    codeword_qam_part1 = codeword_qam[0:64-K]
    codeword_qam_part2 = codeword_qam[64-K:]
    # --- training inputs ----
    if P < K:
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))    
        QAM = Modulation(bits,mu)
        OFDM_data = np.zeros(K, dtype=complex)
        OFDM_data[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
        OFDM_data[dataCarriers] = QAM
    else:
        OFDM_data = np.concatenate((pilotValue,  codeword_qam_part1)) 
    
    H = np.fft.fft(channelResponse, n=64)/np.sqrt(64)
    OFDM_RX_noCP = np.matmul(np.diag(H), OFDM_data)
    signal_power = np.mean(abs(OFDM_RX_noCP**2))
    sigma2 = signal_power * 10**(-SNRdb/10)  
    noise1 = np.sqrt(sigma2/2) * (np.random.randn(*OFDM_RX_noCP.shape)+1j*np.random.randn(*OFDM_RX_noCP.shape))
    OFDM_RX_noCP += noise1
    #OFDM_RX_noCP = removeCP(OFDM_RX)
    # ----- target inputs ---
    symbol = np.zeros(128-K, dtype=complex)
    
    if len(codeword_qam_part2) != len(OFDM_data):
	    print('length of code word is not equal to K, error !!')
    symbol = codeword_qam_part2
    OFDM_data_codeword = symbol
    OFDM_RX_noCP_codeword = np.matmul(np.diag(H), OFDM_data_codeword)
    OFDM_RX_noCP_codeword = add_noise(OFDM_RX_noCP_codeword, SNRdb)

    return np.concatenate((np.concatenate((np.abs(OFDM_RX_noCP),np.angle(OFDM_RX_noCP))), np.concatenate((np.abs(OFDM_RX_noCP_codeword),np.angle(OFDM_RX_noCP_codeword))))), abs(channelResponse)

def qpsk_simulate_vary_pilot_with_insert(codeword, channelResponse,SNRdb, mu, CP_flag, K, P, CP, pilotValue,pilotCarriers, dataCarriers,Clipping_Flag):  
    payloadBits_per_OFDM = mu*len(dataCarriers)
    codeword_qam = Modulation(codeword,mu)
    codeword_qam_part1 = codeword_qam[0:64-K]
    codeword_qam_part2 = codeword_qam[64-K:]
    # --- training inputs ----
    if P < K:
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))    
        QAM = Modulation(bits,mu)
        OFDM_data = np.zeros(K, dtype=complex)
        OFDM_data[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
        OFDM_data[dataCarriers] = QAM
    else:
        # 計算每個位置應插入的數值
        long_array = codeword_qam_part1
        short_array = pilotValue
        insert_len = (len(long_array)+len(short_array))//len(short_array)
        insert_len2 = len(long_array)//len(short_array)
        # 將兩個 array 合併成一個新的 array
        result_array = np.empty(len(long_array) + len(short_array), dtype=long_array.dtype)
        result_array[0::insert_len] = short_array     
        # 对于 long_array 剩下的位置，按顺序安插
        for i in range(len(short_array)):
            result_array[i*insert_len+1 :i*insert_len+insert_len2+1] = long_array[i*insert_len2:(i+1)*insert_len2]
        OFDM_data = result_array
    
    H = np.fft.fft(channelResponse, n=64)/np.sqrt(64)
    OFDM_RX_noCP = np.matmul(np.diag(H), OFDM_data)
    signal_power = np.mean(abs(OFDM_RX_noCP**2))
    sigma2 = signal_power * 10**(-SNRdb/10)  
    noise1 = np.sqrt(sigma2/2) * (np.random.randn(*OFDM_RX_noCP.shape)+1j*np.random.randn(*OFDM_RX_noCP.shape))
    OFDM_RX_noCP += noise1
    #OFDM_RX_noCP = removeCP(OFDM_RX)
    # ----- target inputs ---
    
    if len(codeword_qam_part2) != len(OFDM_data):
	    print('length of code word is not equal to K, error !!')
    symbol = codeword_qam_part2  
    OFDM_data_codeword = symbol
    OFDM_RX_noCP_codeword = np.matmul(np.diag(H), OFDM_data_codeword)
    OFDM_RX_noCP_codeword = add_noise(OFDM_RX_noCP_codeword, SNRdb)

    return  np.concatenate((np.concatenate((np.real(OFDM_RX_noCP),np.imag(OFDM_RX_noCP))), np.concatenate((np.real(OFDM_RX_noCP_codeword),np.imag(OFDM_RX_noCP_codeword))))), abs(channelResponse) #sparse_mask

def qpsk_simulate_vary_pilot_with_insert_AP(codeword, channelResponse,SNRdb, mu, CP_flag, K, P, CP, pilotValue,pilotCarriers, dataCarriers,Clipping_Flag):  
    payloadBits_per_OFDM = mu*len(dataCarriers)
    codeword_qam = Modulation(codeword,mu)
    codeword_qam_part1 = codeword_qam[0:64-K]
    codeword_qam_part2 = codeword_qam[64-K:]
    # --- training inputs ----
    if P < K:
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))    
        QAM = Modulation(bits,mu)
        OFDM_data = np.zeros(K, dtype=complex)
        OFDM_data[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
        OFDM_data[dataCarriers] = QAM
    else:
        # 計算每個位置應插入的數值
        long_array = codeword_qam_part1
        short_array = pilotValue
        insert_len = (len(long_array)+len(short_array))//len(short_array)
        insert_len2 = len(long_array)//len(short_array)
        # 將兩個 array 合併成一個新的 array
        result_array = np.empty(len(long_array) + len(short_array), dtype=long_array.dtype)
        result_array[0::insert_len] = short_array     
        # 对于 long_array 剩下的位置，按顺序安插
        for i in range(len(short_array)):
            result_array[i*insert_len+1 :i*insert_len+insert_len2+1] = long_array[i*insert_len2:(i+1)*insert_len2]
        OFDM_data = result_array
    
    H = np.fft.fft(channelResponse, n=64)/np.sqrt(64)
    OFDM_RX_noCP = np.matmul(np.diag(H), OFDM_data)
    signal_power = np.mean(abs(OFDM_RX_noCP**2))
    sigma2 = signal_power * 10**(-SNRdb/10)  
    noise1 = np.sqrt(sigma2/2) * (np.random.randn(*OFDM_RX_noCP.shape)+1j*np.random.randn(*OFDM_RX_noCP.shape))
    OFDM_RX_noCP += noise1
    #OFDM_RX_noCP = removeCP(OFDM_RX)
    # ----- target inputs ---
    
    if len(codeword_qam_part2) != len(OFDM_data):
	    print('length of code word is not equal to K, error !!')
    symbol = codeword_qam_part2  
    OFDM_data_codeword = symbol
    OFDM_RX_noCP_codeword = np.matmul(np.diag(H), OFDM_data_codeword)
    OFDM_RX_noCP_codeword = add_noise(OFDM_RX_noCP_codeword, SNRdb)

    return np.concatenate((np.concatenate((np.abs(OFDM_RX_noCP),np.angle(OFDM_RX_noCP))), np.concatenate((np.abs(OFDM_RX_noCP_codeword),np.angle(OFDM_RX_noCP_codeword))))), abs(channelResponse)

def qpsk_simulate_without_CP(codeword, channelResponse,SNRdb, mu, CP_flag, K, P, CP, pilotValue,pilotCarriers, dataCarriers,Clipping_Flag):  
    payloadBits_per_OFDM = mu*len(dataCarriers)
    codeword_qam = Modulation(codeword,mu)
    codeword_qam_part1 = codeword_qam[0:64-K]
    codeword_qam_part2 = codeword_qam[64-K:]
    # --- training inputs ----
    if P < K:
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))    
        QAM = Modulation(bits,mu)
        OFDM_data = np.zeros(K, dtype=complex)
        OFDM_data[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
        OFDM_data[dataCarriers] = QAM
    else:
        # 計算每個位置應插入的數值
        long_array = codeword_qam_part1
        short_array = pilotValue
        insert_len = (len(long_array)+len(short_array))//len(short_array)
        insert_len2 = len(long_array)//len(short_array)
        # 將兩個 array 合併成一個新的 array
        result_array = np.empty(len(long_array) + len(short_array), dtype=long_array.dtype)
        result_array[0::insert_len] = short_array     
        # 对于 long_array 剩下的位置，按顺序安插
        for i in range(len(short_array)):
            result_array[i*insert_len+1 :i*insert_len+insert_len2+1] = long_array[i*insert_len2:(i+1)*insert_len2]
        OFDM_data = result_array
        OFDM_data_time_domain = np.fft.ifft(OFDM_data, n=64)*np.sqrt(64)
    OFDM_RX_noCP_time_domain = channel(OFDM_data_time_domain, channelResponse,SNRdb)
    OFDM_RX_noCP = np.fft.fft(OFDM_RX_noCP_time_domain[0:64], n=64)/np.sqrt(64)
    # OFDM_RX_noCP = OFDM_RX_noCP_time_domain[0:64]

    # ----- target inputs ---
    
    if len(codeword_qam_part2) != len(OFDM_data):
	    print('length of code word is not equal to K, error !!')
    symbol = codeword_qam_part2  
    OFDM_data_codeword = symbol
    OFDM_data_codeword_time_domain = np.fft.ifft(OFDM_data_codeword, n=64)*np.sqrt(64)
    OFDM_RX_noCP_codeword_time_domain = channel(OFDM_data_codeword_time_domain, channelResponse,SNRdb)
    OFDM_RX_noCP_codeword = np.fft.fft(OFDM_RX_noCP_codeword_time_domain[0:64], n=64)/np.sqrt(64)
    # OFDM_RX_noCP_codeword = OFDM_RX_noCP_codeword_time_domain[0:64]
    return  np.concatenate((np.concatenate((np.real(OFDM_RX_noCP),np.imag(OFDM_RX_noCP))), np.concatenate((np.real(OFDM_RX_noCP_codeword),np.imag(OFDM_RX_noCP_codeword))))), abs(channelResponse) #sparse_mask

def bpsk_simulate(codeword, channelResponse,SNRdb, mu, CP_flag, K, P, CP, pilotValue,pilotCarriers, dataCarriers,Clipping_Flag):  
    payloadBits_per_OFDM = mu*len(dataCarriers)
    # --- training inputs ----
    if P < K:
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))    
        QAM = Modulation(bits,mu)
        OFDM_data = np.zeros(K, dtype=complex)
        OFDM_data[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
        OFDM_data[dataCarriers] = QAM
    else:
        OFDM_data = pilotValue
    H = np.fft.fft(channelResponse, n=64)
    OFDM_RX_noCP_real = np.matmul(np.diag(H.real), np.squeeze(OFDM_data))
    OFDM_RX_noCP_real = add_noise(OFDM_RX_noCP_real, SNRdb, qpsk=False)

    OFDM_RX_noCP_img = np.matmul(np.diag(H.imag), np.squeeze(OFDM_data))
    OFDM_RX_noCP_img = add_noise(OFDM_RX_noCP_img, SNRdb, qpsk=False)
    #OFDM_RX_noCP = removeCP(OFDM_RX)
    # ----- target inputs ---
    symbol = np.zeros(K, dtype=complex)
    # if len(codeword_qam) != K:
	#     print('length of code word is not equal to K, error !!')
    OFDM_data_codeword = codeword
    OFDM_RX_noCP_codeword_real = np.matmul(np.diag(H.real), OFDM_data_codeword)
    OFDM_RX_noCP_codeword_real = add_noise(OFDM_RX_noCP_codeword_real, SNRdb, qpsk=False)

    OFDM_RX_noCP_codeword_img = np.matmul(np.diag(H.imag), OFDM_data_codeword)
    OFDM_RX_noCP_codeword_img = add_noise(OFDM_RX_noCP_codeword_img, SNRdb, qpsk=False)
    return np.concatenate((np.concatenate((OFDM_RX_noCP_real,OFDM_RX_noCP_img)), np.concatenate((OFDM_RX_noCP_codeword_real,OFDM_RX_noCP_codeword_img)))), abs(channelResponse) #sparse_mask

def find_best_threshold(X, y, num_thresholds=16, num_classes=2):
    num_features = X.shape[1]
    best_thresholds = []

    for feature_index in range(num_features):
        feature = X[:, feature_index]
        feature_best_entropy = float('inf')
        best_threshold = None
        best_probs_left = None
        best_probs_right = None

        thresholds = np.linspace(np.min(feature), np.max(feature), num_thresholds+2)

        for threshold in thresholds:
            mask = feature < threshold
            total_samples = len(y)
            total_samples_left = len(y[mask])
            total_samples_right = len(y[~mask])
            probs_left = [np.sum(y[mask] == c) / total_samples_left for c in range(num_classes)]
            probs_right = [np.sum(y[~mask] == c) / total_samples_right for c in range(num_classes)]

            entropy_left = -np.sum([p * np.log2(p + 1e-10) for p in probs_left])
            entropy_right = -np.sum([p * np.log2(p + 1e-10) for p in probs_right])

            total_entropy = (np.sum(mask) / total_samples) * entropy_left + (np.sum(~mask) / total_samples) * entropy_right

            if total_entropy < feature_best_entropy:
                feature_best_entropy = total_entropy
                best_threshold = threshold
                best_probs_left = probs_left
                best_probs_right = probs_right

        best_thresholds.append((feature_index, feature_best_entropy, best_probs_left, best_probs_right))

    return np.array(best_thresholds)

def feature_selection(X, y, X_test, num_selected_features=128, num_classes=2):
    thresholds_info = find_best_threshold(X, y, num_classes=num_classes)
    selected_features = np.argsort([info[1] for info in thresholds_info])[:num_selected_features]
    selected_features = np.sort(selected_features)
    X_selected = X[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    return X_selected, X_test_selected


def feature_selection_general_DFT(X, y, X_test, pilot_y, num_selected_features=128, num_classes=2):
    num_of_DFT = int(num_selected_features/2)
    num_of_DFT_frame = int(num_of_DFT/2)
    selected_list = []
    # for idx in range(num_of_DFT):
    #     thresholds_info = find_best_threshold(X, pilot_y[:,idx], num_classes=2)
    #     selected_features = np.argsort([info[1] for info in thresholds_info])[:2]
    #     selected_features = np.sort(selected_features)
    #     selected_list = selected_list + selected_features.tolist()  
    selected_features = np.concatenate([np.arange(start, start + 8) for start in [0, 64,128,192]])
    selected_list = selected_list + selected_features.tolist() 
    print(selected_features)
    # for idx in range(num_of_DFT):
    #     thresholds_info = find_best_threshold(X, y[:,idx], num_classes=num_classes)
    #     selected_features = np.argsort([info[1] for info in thresholds_info])[:2]
    #     selected_features = np.sort(selected_features)
    #     selected_list = selected_list + selected_features.tolist()  
    #     print(selected_features)
    selected_features = list(set(selected_list))    
    selected_features = np.sort(selected_features)
    print(selected_features)
    selected_features = np.array(selected_features)
    X_selected = X[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    return X_selected, X_test_selected

def combine_real_imaginary(arr):
    # 將array分割為實數和虛數部分
    K = arr.shape[1]
    real_part =  arr[:, :K//2]
    imag_part = arr[:, K//2:] 
    # 將實數和虛數部分相加
    result = real_part + 1j * imag_part
    return result

def complex_to_real(data):
    return np.concatenate((np.real(data),np.imag(data)), axis=1)

def feature_selection_general(X, y, X_test, y_signal, num_selected_features=128, num_classes=2):
    num_of_DFT = int(num_selected_features/2)
    num_of_DFT_frame = int(num_of_DFT/2)
    CSI_features = np.concatenate((X[:, :num_of_DFT_frame], X[:, 64:(64+num_of_DFT_frame)]), axis=1)
    CSI = combine_real_imaginary(CSI_features)
    print(CSI.shape)
    data = combine_real_imaginary(y_signal)
    print(data.shape)
    selected_list = []
    pilot_features = np.multiply(CSI,data)
    pilot_y = complex_to_real(pilot_features)
    print(pilot_y.shape)
    # thresholds_info, total_entropies = find_best_threshold_RFT(X, CSI_features)   
    # selected_features = np.argsort([info[1] for info in thresholds_info])[:32]
    # selected_features = np.sort(selected_features)
    # selected_list = selected_list + selected_features.tolist()
    selected_features = np.concatenate([np.arange(start, start + 8) for start in [0, 64]])
    selected_list = selected_list + selected_features.tolist() 
    print(selected_features)
    thresholds_info, total_entropies = find_best_threshold_RFT(X, pilot_y)   
    selected_features = np.argsort([info[1] for info in thresholds_info])[:32]
    selected_features = np.sort(selected_features)
    selected_list = selected_list + selected_features.tolist()
    print(selected_features)
    # for idx in range(num_of_DFT):
    #     thresholds_info = find_best_threshold(X, y[:,idx], num_classes=num_classes)
    #     selected_features = np.argsort([info[1] for info in thresholds_info])[:2]
    #     selected_features = np.sort(selected_features)
    #     selected_list = selected_list + selected_features.tolist()   
    selected_features = list(set(selected_list))    
    selected_features = np.sort(selected_features)
    print(selected_features)
    selected_features = np.array(selected_features)
    X_selected = X[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    return X_selected, X_test_selected

def feature_selection_with_1pilot(X, y, X_test, num_selected_features=128, num_classes=2):
    thresholds_info = find_best_threshold(X, y, num_classes=num_classes)
    
    selected_features = np.argsort([info[1] for info in thresholds_info])[:num_selected_features]
    selected_features = np.sort(selected_features)
    pilot_features = selected_features - 128
    new_selected_features = np.concatenate((selected_features, pilot_features))
    X_selected = X[:, new_selected_features]
    X_test_selected = X_test[:, new_selected_features]
    return X_selected, X_test_selected

def feature_selection_with_128pilot(X, y, X_test, num_selected_features=128, num_classes=2):
    thresholds_info = find_best_threshold(X, y, num_classes=num_classes)
    
    selected_features = np.argsort([info[1] for info in thresholds_info])[:num_selected_features]
    selected_features = np.sort(selected_features)
    pilot_features = np.arange(128)
    new_selected_features = np.concatenate((selected_features, pilot_features))
    X_selected = X[:, new_selected_features]
    X_test_selected = X_test[:, new_selected_features]
    return X_selected, X_test_selected

def feature_selection_RFT(X, y, X_test, num_selected_features=128):
    thresholds_info, total_entropies = find_best_threshold_RFT(X, y)
    
    selected_features = np.argsort([info[1] for info in thresholds_info])[:num_selected_features]
    selected_features = np.sort(selected_features)
    print(selected_features)
    X_selected = X[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    return X_selected, X_test_selected, total_entropies

def find_best_threshold_RFT(X, y, num_thresholds=16):
    num_features = X.shape[1]
    best_thresholds = []

    for feature_index in range(num_features):
        feature = X[:, feature_index]
        feature_best_mse = float('inf')
        best_threshold = None
        best_left_mean = None
        best_right_mean = None

        thresholds = np.linspace(np.min(feature), np.max(feature), num_thresholds+2)

        for threshold in thresholds:
            mask = feature < threshold
            total_samples = len(y)
            total_samples_left = len(y[mask])
            total_samples_right = len(y[~mask])
            left_mean = np.mean(y[mask])
            right_mean = np.mean(y[~mask])

            mse_left = np.mean((y[mask] - left_mean) ** 2)
            mse_right = np.mean((y[~mask] - right_mean) ** 2)

            total_mse = (np.sum(mask) / total_samples) * mse_left + (np.sum(~mask) / total_samples) * mse_right

            if total_mse < feature_best_mse:
                feature_best_mse = total_mse
                best_threshold = threshold
                best_left_mean = left_mean
                best_right_mean = right_mean

        best_thresholds.append((feature_index, feature_best_mse, best_left_mean, best_right_mean))
    # import matplotlib.pyplot as plt
    feature_indices, total_entropies, _, _ = zip(*best_thresholds)
    # # 获取排序后的索引
    # sorted_indices = np.argsort(total_entropies)
    
    # # 根据排序后的索引获取排序后的列表
    # sorted_list = np.array(total_entropies)[sorted_indices]

    # # 绘制图形
    # plt.plot(sorted_list, marker='o', linestyle='-')
    
    # # 添加标签和标题
    # plt.xlabel('Sorted Feature Index')
    # plt.ylabel('RFT Loss')
    # plt.title('Input=Re-Im')
    
    # # 显示图形
    # plt.show()       
    return np.array(best_thresholds), total_entropies

def feature_selection_RFT_decide(X, y, X_test, num_selected_features=128):
    # selected_features = np.sort([128,129,130,131,132,133,134,135,192,193,194,195,196,197,198,199])
    selected_features = np.concatenate([np.arange(start, start + 8) for start in [0, 64, 128, 192]])
    # print(selected_features)
    X_selected = X[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    return X_selected, X_test_selected

def feature_selection_decide(X, y, X_test, num_selected_features=128, num_classes=2):
    selected_features = np.sort([128,192])
    # print(selected_features)
    X_selected = X[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    return X_selected, X_test_selected

def RFT_plot(entropy_list):
    import matplotlib.pyplot as plt
    # 使用 zip 函數將相應位置的子列表 B1 和 B2 組合
    combined_lists = zip(*entropy_list)
    
    # 對每對相應位置的元素計算平均值
    total_entropies = [sum(pair) / len(pair) for pair in combined_lists]
    # 获取排序后的索引
    sorted_indices = np.argsort(total_entropies)
    selected_features = np.argsort(total_entropies)[:16]
    selected_features = np.sort(selected_features)
    print(selected_features)    
    # 根据排序后的索引获取排序后的列表
    sorted_list = np.array(total_entropies)[sorted_indices]

    # 绘制图形
    plt.plot(sorted_list, marker='o', linestyle='-')
    
    # 添加标签和标题
    plt.xlabel('Sorted Feature Index')
    plt.ylabel('RFT Loss')
    plt.title('Input=Re-Im')
    
    # 显示图形
    plt.show()       
    
def least_square(train_input, train_output, test_input):
    start_time = time.time()
    # 添加截距項
    X_train = np.c_[np.ones(train_input.shape[0]), train_input]
    X_test = np.c_[np.ones(test_input.shape[0]), test_input]

    # 使用最小二乘法計算權重
    w = np.linalg.lstsq(X_train, train_output, rcond=None)[0]
    end_time = time.time()
    # 預測
    train_pred = X_train @ w
    test_pred = X_test @ w
    training_time = end_time - start_time
    print(f"LS 模型训练时间：{training_time} 秒")
    return train_pred, test_pred, training_time

def mmse(train_input, train_output, test_input):
    start_time = time.time()
    mx = np.mean(train_output, axis=0)
    my = np.mean(train_input, axis=0)

    # Subtract the mean from the input and output
    x_minus_mx = train_output - mx
    y_minus_my = train_input - my

    # Calculate the cross-correlation matrix
    Kxy = (x_minus_mx.T @ y_minus_my)/(train_input.shape[0]-1)
    
    # Calculate the autocorrelation matrix of the input
    Kyy = np.cov(train_input, rowvar=False)
    invcov_Y = np.linalg.inv(Kyy)
    # Calculate the MMSE weights
    W_mmse = Kxy @ invcov_Y

    # Calculate the bias term
    b = np.mean(train_output, axis=0) - np.mean(train_input, axis=0) @ W_mmse.T
    b_mmse = np.tile(b.T, (train_input.shape[0], 1))
    end_time = time.time()
    # 預測
    train_pred = train_input @ W_mmse.T + b_mmse
    test_pred = test_input @ W_mmse.T + b_mmse
    training_time = end_time - start_time
    print(f"MMSE 模型训练时间：{training_time} 秒")
    return train_pred, test_pred, training_time

def perform_pca(training_data, testing_data):
    # 分區
    num_features = training_data.shape[1]
    partition_size = num_features 

    # 初始化 PCA
    pca = PCA()

    # 初始化結果
    pca_training_data = np.empty_like(training_data)
    pca_testing_data = np.empty_like(testing_data)

    # 分區進行 PCA
    for i in range(1):
        start_idx = i * partition_size
        end_idx = (i + 1) * partition_size

        # # 移除平均值
        # mean_training_partition = training_data[:, start_idx:end_idx].mean(axis=0)

        # training_partition = training_data[:, start_idx:end_idx] - mean_training_partition
        # testing_partition = testing_data[:, start_idx:end_idx] - mean_training_partition

        # PCA 轉換
        pca_training_partition = pca.fit_transform(training_data)
        pca_testing_partition = pca.transform(testing_data)

        # 放回原始數據
        pca_training_data[:, start_idx:end_idx] = pca_training_partition
        pca_testing_data[:, start_idx:end_idx] = pca_testing_partition

    return pca_training_data, pca_testing_data

def my_pca(training_partition, testing_partition):
    mean_training_partition = training_partition.mean(axis=0)

    training_partition = training_partition - mean_training_partition
    testing_partition = testing_partition - mean_training_partition
    # 計算共變異矩陣
    covariance_matrix = np.cov(training_partition, rowvar=False)

    # 找到特徵向量
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # 將特徵向量按照特徵值降序排列
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]

    # 選取前 k 個特徵向量，這裡 k 可以是你希望保留的主成分數量
    k = 256
    selected_eigenvectors = eigenvectors[:, :k]

    # 投影數據
    pca_training_partition = np.dot(training_partition, selected_eigenvectors)
    pca_testing_partition = np.dot(testing_partition, selected_eigenvectors)

    return pca_training_partition, pca_testing_partition

def qpsk_simulate_vary_pilot_with_insert_ML(codeword, channelResponse,SNRdb, mu, CP_flag, K, P, CP, pilotValue,pilotCarriers, dataCarriers,Clipping_Flag):  
    payloadBits_per_OFDM = mu*len(dataCarriers)
    codeword_qam = Modulation(codeword,mu)
    codeword_qam_part1 = codeword_qam[0:64-K]
    codeword_qam_part2 = codeword_qam[64-K:]
    # --- training inputs ----
    if P < K:
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))    
        QAM = Modulation(bits,mu)
        OFDM_data = np.zeros(K, dtype=complex)
        OFDM_data[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
        OFDM_data[dataCarriers] = QAM
    else:
        # 計算每個位置應插入的數值
        long_array = codeword_qam_part1
        short_array = pilotValue
        insert_len = (len(long_array)+len(short_array))//len(short_array)
        insert_len2 = len(long_array)//len(short_array)
        # 將兩個 array 合併成一個新的 array
        result_array = np.empty(len(long_array) + len(short_array), dtype=long_array.dtype)
        result_array[0::insert_len] = short_array     
        # 对于 long_array 剩下的位置，按顺序安插
        for i in range(len(short_array)):
            result_array[i*insert_len+1 :i*insert_len+insert_len2+1] = long_array[i*insert_len2:(i+1)*insert_len2]
        OFDM_data = result_array
    
    H = np.fft.fft(channelResponse, n=64)/np.sqrt(64)
    OFDM_RX_noCP = np.matmul(np.diag(H), OFDM_data)
    signal_power = np.mean(abs(OFDM_RX_noCP**2))
    sigma2 = signal_power * 10**(-SNRdb/10)  
    noise1 = np.sqrt(sigma2/2) * (np.random.randn(*OFDM_RX_noCP.shape)+1j*np.random.randn(*OFDM_RX_noCP.shape))
    OFDM_RX_noCP += noise1
    #OFDM_RX_noCP = removeCP(OFDM_RX)
    # ----- target inputs ---
    
    if len(codeword_qam_part2) != len(OFDM_data):
	    print('length of code word is not equal to K, error !!')
    symbol = codeword_qam_part2  
    OFDM_data_codeword = symbol
    OFDM_RX_noCP_codeword = np.matmul(np.diag(H), OFDM_data_codeword)
    OFDM_RX_noCP_codeword = add_noise(OFDM_RX_noCP_codeword, SNRdb)
    def ML_cost_function(y, X, H_hat,sigma):
        error_list = []
        for i in range(4):
            error = np.linalg.norm(y - np.dot(H_hat, X[i]))**2
            error_list.append(error)
        return error_list
    def find_ML(cost_function, X_initial):
        minimum_dist = cost_function[0]
        X_mle = X_initial[0]
        for i in range(4):
            if(minimum_dist > cost_function[i]):
                minimum_dist = cost_function[i]
                X_mle = X_initial[i]
        return X_mle
    def qpsk_mapping(receive_symbol):
        real_part = np.real(receive_symbol)
        imag_part = np.imag(receive_symbol)
        real_bit = np.where(real_part >0, 1, 0)
        imag_bit = np.where(imag_part >0, 1, 0)

        return real_bit,imag_bit
    codeword_hat_list = []
    for i in range(len(OFDM_RX_noCP_codeword)):  
        #X ?絲憪?
        X_initial = [np.exp(1j * math.pi/4), np.exp(1j * math.pi*3/4), np.exp(1j * math.pi*5/4), np.exp(1j * math.pi*7/4)]
        cost_function = ML_cost_function(OFDM_RX_noCP_codeword[i], X_initial, H[i], sigma2)
        # ?撠??格??賣
        X_mle = find_ML(cost_function, X_initial)

    
        # ?拍 QPSK ??撠?X_mle 閫???拙???
    
        real_bit, imag_bit = qpsk_mapping(X_mle) 
        codeword_hat_list.append(real_bit)
        codeword_hat_list.append(imag_bit)
    codeword_hat_list = np.asarray(codeword_hat_list)
        
    return codeword_hat_list