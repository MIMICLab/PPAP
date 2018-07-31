import tensorflow as tf
import numpy as np
from utils.utils import *
import sys
import os
import math
import time
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data
from glob import glob
from random import shuffle
from utils.download import download_celeb_a, download_lsun
from utils.Lsun import Lsun

def data_loader(dataset):
    if dataset == 'mnist':
        mb_size = 256
        X_dim = 784
        width = 28
        height = 28
        channels = 1
        len_x_train = 60000
        len_x_test = 10000
        x_train = input_data.read_data_sets('data/MNIST_data', one_hot=True)
        x_test,_ = x_train.test.next_batch(len_x_test, shuffle=False)
        x_test = np.reshape(x_test,[-1,28,28,1])
        
    if dataset == 'svhn':
        mb_size = 256
        X_dim = 1024
        width = 32
        height = 32
        channels = 3    
        len_x_train = 604388
        len_x_test = 26032

        train_location = 'data/SVHN/train_32x32.mat'
        extra_location = 'data/SVHN/extra_32x32.mat'
        test_location = 'data/SVHN/test_32x32.mat'

        train_dict = sio.loadmat(train_location)
        x_ = np.asarray(train_dict['X'])
        x_train = []
        for i in range(x_.shape[3]):
            x_train.append(x_[:,:,:,i])
        x_train = np.asarray(x_train)

        extra_dict = sio.loadmat(extra_location)
        x_ex = np.asarray(extra_dict['X'])
        x_extra = []
        for i in range(x_ex.shape[3]):
            x_extra.append(x_ex[:,:,:,i])
        x_extra = np.asarray(x_extra)
        x_train = np.concatenate((x_train, x_extra), axis=0)
        x_train = normalize(x_train)
        
        test_dict = sio.loadmat(test_location)
        x_ = np.asarray(test_dict['X'])
        x_test = []
        for i in range(x_.shape[3]):
            x_test.append(x_[:,:,:,i])
        x_test = np.asarray(x_test)
        
    if dataset == 'cifar10':
        mb_size = 256
        X_dim = 1024
        len_x_train = 50000
        len_x_test = 10000
        width = 32
        height = 32
        channels = 3    
        (x_train, y_train), (x_test, y_test) = load_data()
        x_train = normalize(x_train)
        x_test = normalize(x_test)

    if dataset == 'celebA':
        mb_size = 128
        X_dim = 4096
        width = 64
        height = 64
        channels = 3     
        download_celeb_a("data")
        data_files = glob(os.path.join("data/celebA/*.jpg"))
        len_x_train = 200000
        len_x_test = 2599
        sample = [get_image(sample_file, 128, True, 64, is_grayscale = 0) for sample_file in data_files]
        sample_images = np.array(sample).astype(np.float32)
        x_train = sample_images[:200000]
        x_test = sample_images[200000:]
        
    if dataset == 'lsun':
        mb_size = 128
        X_dim = 4096
        width = 64
        height = 64
        channels = 3    
        download_lsun("data")
        lsun = Lsun("data/lsun/bedroom_train_lmdb")
        len_x_train = 3000000  
        len_x_test = 33042
        sample_images = lsun.load_data(len_x_train)
        x_train = sample_images[:3000000]
        x_test = sample_images[3000000:]      
        
    return mb_size, X_dim, width, height, channels,len_x_train, x_train, len_x_test, x_test 