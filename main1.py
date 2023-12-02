# -*- coding: utf-8 -*-
"""
Created on    2023/11/16 10:36

@author: roger
Reference: https://www.kaggle.com/code/tarushijat/music-genre-classification-using-cnn
"""
import os
import sys
import librosa
import numpy as np
from package import utils
from package import ml_tools


#files folder pathway
MUSIC_PATH = {
    "GENRE_ORI": r"./genres_original/", #all data
    "DEBUG_ORI": r"./genres_original_debug/" #small amount data for debug
}

# Feature extraction parameters
fextrac_para = {
    "SR": 22050, # sampling rate( total samples / total secs)
    "TRACK_SIZE": 30000, # how many ms a track is
    "FRAME_SIZE": 32, #ms
    "FRAME_SIFT": 10, #ms
    "n_mfcc": 258, #number of mfcc per frame(default = 258)
    "n_fft": 2048, #n_fft=2048(default = 2048) #the biger the run time longer
    "width": 272 #3D array feature structure width  #better to be a factor of "frame number per track"
    #frame number per track = int((TRACK_SIZE - FRAME_SIZE)/(FRAME_SIZE - FRAME_SIFT)) + 1 - 3
}

SPLIT_RATE = 0.2 #train test split -- testing ratio
NOR_METHOD = 1  #Normalization method--  0: StandardScaler, 1: MinMaxScaler, Else: no 

#%% Feature Extraction 

# read datasetpath and corressponding target class
music_dataset_blue_train,  genre_target_blue_train = utils.load_files(MUSIC_PATH["DEBUG_ORI"] + "training/blues/", natsort = True)
music_dataset_blue_test,  genre_target_blue_test = utils.load_files(MUSIC_PATH["DEBUG_ORI"] + "testing/blues/", natsort = True)
music_dataset_classical_train,  genre_target_classical_train = utils.load_files(MUSIC_PATH["DEBUG_ORI"] + "training/classical/", natsort = True)
music_dataset_classical_test,  genre_target_classical_test = utils.load_files(MUSIC_PATH["DEBUG_ORI"] + "training/classical/", natsort = True)
music_dataset_country_train,  genre_target_country_train = utils.load_files(MUSIC_PATH["DEBUG_ORI"] + "training/country/", natsort = True)
music_dataset_country_test,  genre_target_country_test = utils.load_files(MUSIC_PATH["DEBUG_ORI"] + "training/country/", natsort = True)

wave_list_blue_train = utils.load_wavs(music_dataset_blue_train, fextrac_para["SR"])
wave_list_blue_test = utils.load_wavs(music_dataset_blue_test, fextrac_para["SR"])
wave_list_classical_train = utils.load_wavs(music_dataset_classical_train, fextrac_para["SR"])
wave_list_classical_test = utils.load_wavs(music_dataset_classical_test, fextrac_para["SR"])
wave_list_country_train = utils.load_wavs(music_dataset_country_train, fextrac_para["SR"])
wave_list_country_test = utils.load_wavs(music_dataset_country_test, fextrac_para["SR"])

#X ndarray(n * width, 256) and y ndarray(n, )
X_blue_train, y_blue_train = utils.mfcc_extraction(digit_list = wave_list_blue_train, target = genre_target_blue_train, para = fextrac_para, label = 0)
X_blue_test, y_blue_test = utils.mfcc_extraction(digit_list = wave_list_blue_test, target = genre_target_blue_test, para = fextrac_para, label = 0)
X_classical_train, y_classical_train = utils.mfcc_extraction(digit_list = wave_list_classical_train, target = genre_target_classical_train, para = fextrac_para, label = 1)
X_classical_test, y_classical_test = utils.mfcc_extraction(digit_list = wave_list_classical_test, target = genre_target_classical_test, para = fextrac_para, label = 1)
X_country_train, y_country_train = utils.mfcc_extraction(digit_list = wave_list_country_train, target = genre_target_country_train, para = fextrac_para, label = 2)
X_country_test, y_country_test = utils.mfcc_extraction(digit_list = wave_list_country_test, target = genre_target_country_test, para = fextrac_para, label = 2)

#%% Preprocessing

# normalize and reshape
X_train, X_test = ml_tools.data_process(X_blue_train, X_blue_test, X_classical_train, X_classical_test, X_country_train, X_country_test , NOR_METHOD, fextrac_para["width"])
y_train, y_test = np.concatenate((y_blue_train, y_classical_train, y_country_train), axis = 0), np.concatenate((y_blue_test, y_classical_test, y_country_test), axis = 0)

"""
print("\n\nDEBUG MODE\n\n")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_train)
print(y_test.shape)
print(y_test) """

#save digit data to file X_train X_test 4Darray,  y_train y_test 1Darray
ml_tools.feature_save(X_train, X_test, y_train, y_test, ml_tools.feature_path)



