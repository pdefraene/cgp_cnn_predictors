#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pickle
import pandas as pd

from cgp import *
from cgp_config import *
from cnn_model import CGP2CNN
from cnn_train import CNN_train
from e2epp import E2epp

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evolving CAE structures')
    #parser.add_argument('--gpu_num', '-g', type=int, default=1, help='Num. of GPUs')
    parser.add_argument('--lam', '-l', type=int, default=2, help='Num. of offsprings')
    parser.add_argument('--net_info_file', default='network_info.pickle', help='Network information file name')
    parser.add_argument('--mode', '-m', default='evolution', help='Mode (evolution / retrain / reevolution)')
    parser.add_argument('--init', '-i', action='store_true')
    parser.add_argument('--dataset', '-d', default='cifar10')
    parser.add_argument('--predictor', '-p', default='training')
    parser.add_argument('--acc_size', '-aas', default="false")
    parser.add_argument('--alpha', '-a', default=0.05)
    args = parser.parse_args()

    if args.acc_size == "false":
        acc_size = False
    else:
        acc_size = True

    # --- Optimization of the CNN architecture ---
    if args.mode == 'evolution':
        if args.predictor == 'training':
            # Create CGP configuration and save network information
            network_info = CgpInfoConvSet(rows=5, cols=30, level_back=10, min_active_num=1, max_active_num=30)
            with open(args.net_info_file, mode='wb') as f:
                pickle.dump(network_info, f)
            # Evaluation function for CGP (training CNN and return validation accuracy)
            imgSize = 32

            eval_f = CNNEvaluation(gpu_num=1, dataset=args.dataset, verbose=True, epoch_num=50, batchsize=128,
                                   imgSize=imgSize, predictor=None, acc_size=acc_size, alpha=args.alpha)

            # Execute evolution
            cgp = CGP(network_info, eval_f, lam=args.lam, imgSize=imgSize, init=args.init)
            # cgp.modified_evolution(max_eval=250, mutation_rate=0.1, log_file=args.log_file)
            cgp.modified_evolution(max_eval=50, mutation_rate=0.1)
        elif args.predictor == 'e2epp':
            network_info = CgpInfoConvSet(rows=5, cols=30, level_back=10, min_active_num=1, max_active_num=30)
            data_file = "e2epp_data_" + args.dataset + ".txt"
            predictor = E2epp(nb_trees=1000,training_data=data_file)
            imgSize = 32
            eval_f = CNNEvaluation(gpu_num=1, dataset=args.dataset, verbose=True, epoch_num=50, batchsize=128,
                                   imgSize=imgSize, predictor=predictor, acc_size=acc_size, alpha=args.alpha)
            cgp = CGP(network_info, eval_f, lam=args.lam, imgSize=32, init=args.init)

            cgp.modified_evolution(max_eval=1000, mutation_rate=0.1)



    # --- Retraining evolved architecture ---
    elif args.mode == 'retrain':
        print('Retrain')
        time_start = time.time()


        # In the case of existing log_cgp.txt
        # Load CGP configuration
        with open(args.net_info_file, mode='rb') as f:
            network_info = pickle.load(f)
        # Load network architecture
        cgp = CGP(network_info, None, w=False)
        data = pd.read_csv("log_e2epp_aas_0.05_svhn.txt", header=None)  # Load log file
        cgp.load_log(list(data.tail(1).values.flatten().astype(int)))  # Read the log at final generation
        print(cgp._log_data(net_info_type='active_only', start_time=0))
        # Retraining the network
        #temp = CNN_train(args.dataset, validation=False, verbose=True, batchsize=128)
        temp = CNN_train(args.dataset, validation=True, verbose=True, batchsize=128)
        #acc = temp(cgp.pop[0].active_net_list(), 0, epoch_num=500, out_model='retrained_net.model')
        acc = temp(cgp.pop[0].active_net_list(), 0, epoch_num=50, out_model='retrained_net.model')
        print("Accuracy:", acc)
        print("Time:  ", time.time() - time_start)
        net = CGP2CNN(cgp.pop[0].active_net_list(), 3, 10, 32)
        print("Number of trainable parameters: ", net.count_params())

        # # otherwise (in the case where we do not have a log file.)
        # temp = CNN_train('haze1', validation=False, verbose=True, imgSize=128, batchsize=16)
        # cgp = [['input', 0], ['S_SumConvBlock_64_3', 0], ['S_ConvBlock_64_5', 1], ['S_SumConvBlock_128_1', 2], ['S_SumConvBlock_64_1', 3], ['S_SumConvBlock_64_5', 4], ['S_DeConvBlock_3_3', 5]]
        # acc = temp(cgp, 0, epoch_num=500, out_model='retrained_net.model')

    elif args.mode == 'reevolution':
        # restart evolution
        print('Restart Evolution')
        imgSize = 32
        with open('network_info.pickle', mode='rb') as f:
            network_info = pickle.load(f)
        eval_f = CNNEvaluation(gpu_num=1, dataset=args.dataset, verbose=True, epoch_num=50, batchsize=128, imgSize=imgSize,
                               predictor=None, acc_size=acc_size, alpha=args.alpha)
        cgp = CGP(network_info, eval_f, lam=args.lam, imgSize=imgSize)

        data = pd.read_csv('./log_training_svhn.txt', header=None)
        cgp.load_log(list(data.tail(1).values.flatten().astype(int)))
        cgp.modified_evolution(max_eval=50, mutation_rate=0.1)

    else:
        print('Undefined mode. Please check the "-m evolution or retrain or reevolution" ')
