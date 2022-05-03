#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multiprocessing as mp
import multiprocessing.pool
import numpy as np
import cnn_train as cnn
import csv


# wrapper function for multiprocessing
from e2epp import E2epp


def arg_wrapper_mp(args):
    return args[0](*args[1:])

def encodage_for_e2epp(net,size=30):
    """
    give the encodage of the individuals
    Returns
    -------
    encodage
    """
    encoding = []
    #for each layer encode it in number version
    for name, in1, in2 in net:
        if name == 'input' in name:
            encoding.append([1, in1, in2, 0, 0])
        elif name == 'full':
            encoding.append([2, in1, in2, 0, 0])
        elif name == 'Max_Pool':
            encoding.append([3, in1, in2, 0, 0])
        elif name == 'Avg_Pool':
            encoding.append([4, in1, in2, 0, 0])
        elif name == 'Concat':
            encoding.append([5, in1, in2, 0, 0])
        elif name == 'Sum':
            encoding.append([6, in1, in2, 0, 0])
        else:
            key = name.split('_')
            down = key[0]
            func = key[1]
            out_size = int(key[2])
            kernel = int(key[3])
            if down == 'S':
                if func == 'ConvBlock':
                    encoding.append([7, in1, in2, out_size, kernel])
                elif func == 'ResBlock':
                    encoding.append([8, in1, in2, out_size, kernel])

    # add layer 0 empty for have fixed size
    for _ in range(len(encoding), size):
        encoding.append([0, 0, 0, 0, 0])
    return encoding
# Evaluation of CNNs
def cnn_eval(net, gpu_id, epoch_num, batchsize, dataset, verbose, imgSize, predictor,acc_size, alpha,len_net):
    """
    Evaluate a Neural network
    Parameters
    ----------
    net: network
    gpu_id: ID of the gpu (multiprocessing)
    epoch_num: number of epoch for training the network
    batchsize: batch size for the training
    dataset: dataset for train
    verbose: boolean value for print info
    imgSize: size of the image
    predictor: predictor for eval

    Returns
    -------
    Accuracy of the network
    """
    evaluation = 0
    if predictor is None and not acc_size:
        print("classic training")
        train = cnn.CNN_train(dataset, validation=True, verbose=verbose, imgSize=imgSize, batchsize=batchsize)
        evaluation = train(net, gpu_id, epoch_num=epoch_num, out_model=None)
        print("evaluation: ", evaluation)
    elif isinstance(predictor, E2epp) and not acc_size:
        print("e2epp")
        encoding = encodage_for_e2epp(net)
        evaluation = predictor.predict_performance(encoding)
        print("evaluation: ", evaluation)
    elif isinstance(predictor, E2epp) and acc_size:
        print("e2epp AAS")
        encoding = encodage_for_e2epp(net)
        acc = predictor.predict_performance(encoding)
        size = alpha / len_net
        print("acc: ", acc)
        print("size", size)
        evaluation = acc + size
        print("evaluation: ", evaluation)
    elif predictor is None and acc_size:
        print("training AAS")
        train = cnn.CNN_train(dataset, validation=True, verbose=verbose, imgSize=imgSize, batchsize=batchsize)
        acc = train(net, gpu_id, epoch_num=epoch_num, out_model=None)
        size = alpha / len_net
        print("acc: ", evaluation)
        print("size", size)
        evaluation = acc + size
        print("evaluation: ", evaluation)
        fw = open("CIFAR100_AAS_0_05_EVAL", 'a')
        writer = csv.writer(fw, lineterminator='\n')
        writer.writerow([acc,len_net,size,evaluation])
        fw.close()
    return evaluation


class CNNEvaluation(object):
    def __init__(self, gpu_num, dataset='cifar10', verbose=True, epoch_num=50, batchsize=16, imgSize=32, predictor = None, acc_size = False,alpha=10):
        self.gpu_num = gpu_num
        self.epoch_num = epoch_num
        self.batchsize = batchsize
        self.dataset = dataset
        self.verbose = verbose
        self.imgSize = imgSize
        self.predictor = predictor
        self.acc_size = acc_size
        self.alpha = alpha

    def __call__(self, net_lists):
        print("Net list of a CNNEvaluation:  ",net_lists) # check net_list

        evaluations = np.zeros(len(net_lists))
        for i in np.arange(0, len(net_lists)):
            evaluations[i] = cnn_eval(net_lists[i], 0, self.epoch_num, self.batchsize, self.dataset,
                                      self.verbose, self.imgSize, self.predictor, self.acc_size, self.alpha, len(net_lists[i]))
            print("size", len(net_lists[i]))

        return evaluations


# network configurations
class CgpInfoConvSet(object):
    def __init__(self, rows=30, cols=40, level_back=40, min_active_num=8, max_active_num=50):
        self.input_num = 1
        # "S_" means that the layer has a convolution layer without downsampling.
        # "D_" means that the layer has a convolution layer with downsampling.
        # "Sum" means that the layer has a skip connection.
        self.func_type = ['S_ConvBlock_32_1',    'S_ConvBlock_32_3',   'S_ConvBlock_32_5',
                          'S_ConvBlock_128_1',    'S_ConvBlock_128_3',   'S_ConvBlock_128_5',
                          'S_ConvBlock_64_1',     'S_ConvBlock_64_3',    'S_ConvBlock_64_5',
                          'S_ResBlock_32_1',     'S_ResBlock_32_3',    'S_ResBlock_32_5',
                          'S_ResBlock_128_1',     'S_ResBlock_128_3',    'S_ResBlock_128_5',
                          'S_ResBlock_64_1',      'S_ResBlock_64_3',     'S_ResBlock_64_5',
                          'Concat', 'Sum',
                          'Max_Pool', 'Avg_Pool']
                          
        self.func_in_num = [1, 1, 1,
                            1, 1, 1,
                            1, 1, 1,
                            1, 1, 1,
                            1, 1, 1,
                            1, 1, 1,
                            2, 2,
                            1, 1]

        self.out_num = 1
        self.out_type = ['full']
        self.out_in_num = [1]

        # CGP network configuration
        self.rows = rows
        self.cols = cols
        self.node_num = rows * cols
        self.level_back = level_back
        self.min_active_num = min_active_num
        self.max_active_num = max_active_num

        self.func_type_num = len(self.func_type)
        self.out_type_num = len(self.out_type)
        self.max_in_num = np.max([np.max(self.func_in_num), np.max(self.out_in_num)])
