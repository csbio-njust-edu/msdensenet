# train_on_cell_datasets.py
# !/usr/bin/env	python3
"""
train network using pytorch
author Long-Chen Shen & Yu-Hang Yin
"""

import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from train_test_api.train_api import train
from train_test_api.test_api import eval_training

from conf import settings
from utils import get_network, get_training_dataloader, get_valid_dataloader, get_test_dataloader, \
    get_parameter_number, save_best_result
import sys
import csv
rootPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(rootPath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    args.with_cuda = True
    cuda_condition = torch.cuda.is_available() and args.with_cuda

    args.net = "msdensenet"
    args.device = torch.device("cuda:0" if cuda_condition else "cpu")
    args.b = 128
    args.warm = 0
    args.lr = 0.002

    dataset_path = os.path.join("Dataset", "cell_dataset")
    dataset_list = sorted(os.listdir(dataset_path))

    result_csv_dir = os.path.join(rootPath, "cell_datasets_result", "result_csv")
    if not os.path.exists(result_csv_dir):
        os.makedirs(result_csv_dir)

    data_index = 1
    for dataset_ in dataset_list:
        print("This is the ", data_index, " dataset",dataset_)
        data_index += 1
        net = get_network(args)
        # print("NET:")
        # print(net)
        patience = settings.PATIENCE
        # network parameters
        print(get_parameter_number(net))
        # data preprocessing:
        dataset = os.path.join(dataset_path, dataset_)
        dna_training_loader = get_training_dataloader(path=dataset, num_workers=0, batch_size=args.b, shuffle=True)
        dna_valid_loader = get_valid_dataloader(path=dataset, num_workers=0, batch_size=args.b, shuffle=False)
        dna_test_loader = get_test_dataloader(path=dataset, num_workers=0, batch_size=args.b, shuffle=False)

        loss_function = nn.CrossEntropyLoss()
        softmax_output = nn.Softmax(dim=1)

        optimizer = optim.SGD(params=net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.8)
        recent_folder = ""
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW, dataset_)

        # use tensorboard
        if not os.path.exists(settings.LOG_DIR):
            os.mkdir(settings.LOG_DIR)

        #  record the epoch
        df_path = os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW, dataset_)
        if not os.path.exists(df_path):
            os.makedirs(df_path)
        df_file = os.path.join(df_path, "df_log.pickle")
        if not os.path.isfile(df_file):
            df_ = pd.DataFrame(columns=["epoch", "lr", "train_loss", "train_acc",
                                        "valid_loss", "valid_acc", "valid_auc",
                                        "test_loss", "test_acc", "test_auc"])
            df_.to_pickle(df_file)
            print("log DataFrame created!")

        # create model_weights folder to save model
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

        best_auc = 0.0
        best_testAUC = 0.0
        best_testAcc = 0.0
        best_testPrec = 0.0
        best_testRecall = 0.0
        best_testF1 = 0.0
        best_epoch = 0


        for epoch in range(1, settings.EPOCH + 1):

            output_interval = settings.OUTPUT_INTERVAL
            log_dic = train(net, dna_training_loader, optimizer, loss_function, epoch, args,
                            output_interval)

            if epoch > args.warm:
                train_scheduler.step()

            epoch_, auc_valid, cur_result, pred_result_test, acc, prec, rec, f1 = eval_training(net, dna_valid_loader, dna_test_loader,
                                                                    loss_function, softmax_output, args,
                                                                    epoch=epoch, df_file=df_file,
                                                                    log_dic=log_dic, train_after=True)
            # start to save best performance model after learning rate decay to 0.01
            if best_auc < auc_valid:
                weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
                print('saving weights file to {}'.format(weights_path))
                torch.save(net.state_dict(), weights_path)
                best_auc = auc_valid
                patience = settings.PATIENCE
                # save best result
                save_best_result(df_path, pred_result_test)
                best_testAUC = cur_result
                best_testAcc = acc
                best_testPrec = prec
                best_testRecall = rec
                best_testF1 = f1
                best_epoch = epoch_
                continue

            if not epoch % settings.SAVE_EPOCH:
                weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
                print('saving weights file to {}'.format(weights_path))
                torch.save(net.state_dict(), weights_path)
            patience -= 1
            if patience == 0:
                print("DatasetName:", dataset_,",The best epoch:",best_epoch , ", The best AUC:", best_testAUC)
                print("The end!")
                break
        """记录bestAUC"""
        bestAUC_csv = os.path.join(result_csv_dir, "CellDataset_BestResult.csv")
        if data_index == 2:
            with open(bestAUC_csv, 'w+', newline="") as f:
                csv_write = csv.writer(f)
                csv_head = ["dataset","epoch", "AUC", "ACC", "Precision", "Recall", "F1score"]
                csv_write.writerow(csv_head)
        with open(bestAUC_csv, 'a+', newline="") as f:
            csv_write = csv.writer(f)
            data_row = [dataset_, best_epoch, best_testAUC, best_testAcc, best_testPrec, best_testRecall, best_testF1]
            csv_write.writerow(data_row)