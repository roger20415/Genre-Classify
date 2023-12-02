# -*- coding: utf-8 -*-
"""
Created on    2023/11/16 10:36 

@author: roger
"""

import os
import sys
import numpy as np
from natsort import natsorted
import librosa
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


#from folder path to data path list & target list 
def load_files(PATH, natsort = False):
    dataset_path = list() #empty list for music original data path
    target  = list()  #empty list for music genre target class
    # Read filepaths and target class
    for root, dirs, filenames in os.walk(PATH):
        for name in filenames:
            filepath = os.path.join(root, name)
            filepath = filepath.replace('\\', '/')
            #print(f"Read:{filepath}")
            
            dataset_path.append(filepath)
            target.append(filepath.split('/')[-2])

    target = target_to_digit(target)

    if natsort == True:
        dataset_path, target = natsorted(dataset_path), natsorted(target)
        
    #print 10 genre target classes
    print(f"\nInput folder:\t{PATH}")
    print(f"Total data:\t{len(dataset_path)}\n")
    return dataset_path, target

#from dataset_path(list) to dataset_digital(list) by librosa
def load_wavs(dataset, sampling_rate):
    
    wavs_digit_list = list()
    for filepath in tqdm(dataset, desc = "Waves Loading"):
        wav_array, _ = librosa.load(filepath, sr = sampling_rate, mono = True)
        wavs_digit_list.append(wav_array)

    print("--Load Wavs Success--\n")

    return wavs_digit_list

# from wavs_digit_list to X ndarray(n * width, 256) and y ndarray(n, ) 
def mfcc_extraction(digit_list, target, para, label):
    sr_ms = para["SR"] / 1000 # samling num/sec to sampling num/ ms
    tot_track_sample = sr_ms * para["TRACK_SIZE"] # float
    tot_frame_sample = sr_ms * para["FRAME_SIZE"] # float
    tot_framesift_sample = sr_ms * para["FRAME_SIFT"] # float

    frames_per_track = (int((tot_track_sample - tot_frame_sample) / (tot_frame_sample - tot_framesift_sample)) + 1) - 3 #3 is to discard each track tail 

    # make frames_per_track be a factor of width, so each set of width are from the same track
    discard = frames_per_track % para["width"] 
    frames_per_track -= discard

    mfccs = list()
    # all data range
    for digit in tqdm(digit_list, desc = "Feature Extracting"):  # i: track number;  digit: track digit data
        # print(f"\n{target[i]} mfcc extracting\n----") #debug
        

    
        #all track range
        for j in range(frames_per_track): #j: frame number in the track
            
            start = round(j * (tot_frame_sample - tot_framesift_sample))
            end = round(start + tot_frame_sample)


            mfcc_ndarr = librosa.feature.mfcc(y = digit[start : end],
                                        sr = para["SR"], 
                                        n_mfcc = para["n_mfcc"], 
                                        dct_type = 2, 
                                        norm = 'ortho',
                                        lifter = 0,
                                        n_fft = para["n_fft"])
            mfcc_ndarr = mfcc_ndarr
            mfcc_ndarr = mfcc_ndarr.reshape((mfcc_ndarr.shape[0]*mfcc_ndarr.shape[1]))
            mfccs.append(mfcc_ndarr)

            #if (j == tot_frame_num - 1):
            #    print(f"{target[i]}--total frame:{j + 1}")
            #    print(mfcc_ndarr.shape) #(2, 128) #debug
            #   print(mfcc_ndarr) #debug
        
        #print(f"{target[i]}:{i + 1} mfcc extracting finish") #debug
        #print(f"mfcc len now: {len(mfccs)}") #debug
        #print(f"label len now: {len(label)}\n\n") #debug

    X = np.array(mfccs)
    y = np.ones(int(X.shape[0] / para["width"]), ) * label

    print("--MFCC extraction finish--\n\n")
    #print(f"y: {y}\n\n")
    return X, y

# convert target from string to digital
def target_to_digit(target):

    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(target)
    return target


"""
# combine label(list, len n*width) to y(ndarray, shape(n, ))
def generate_y(label, width):

    y = list()
    for i in range(int(len(label) / width)):   #width must be a factor of len(label) 
        y.append(label[i * width])

    y = np.array(y)
    return y
"""



