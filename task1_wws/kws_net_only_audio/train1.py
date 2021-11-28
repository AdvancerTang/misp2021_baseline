# -*- coding: utf-8 -*-
import argparse
import os
import time
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import sys
import random

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append("..")
from tools import utils

# test
# from model.Resnet import KWS_Net
# from model.Densenet import KWS_Net
# good
from model.DS_CNN import KWS_Net
# from model.res15 import KWS_Net
# from model.audio_kwsnet import KWS_Net

# bad
# from model.DScnn_attention import KWS_Net
# from model.DCN import KWS_Net

from reader.data_reader_audio import myDataLoader, myDataset


def main(args):
    # Compile and configure all the model parameters.
    model_path = args.project
    log_dir = args.logdir
    logger = utils.get_logger(log_dir + '/' + args.project)
    seed_torch(args.seed)

    # load mean and var
    fb40_train_mean_var = np.load('scp_dir/train_mean_var_fb40_.npz')
    fb40_train_mean = fb40_train_mean_var['_mean']
    fb40_train_var = fb40_train_mean_var['_var']

    # file path
    file_train_positive_path = 'scp_dir/positive_train.scp'
    file_train_negative_path = 'scp_dir/negative_train.scp'
    file_dev_positive_path_middle = 'scp_dir/positive_dev_middle.scp'
    file_dev_positive_path_far = 'scp_dir/positive_dev_far.scp'
    file_dev_negative_path_middle = 'scp_dir/negative_dev_middle.scp'
    file_dev_negative_path_far = 'scp_dir/negative_dev_far.scp'

    # define the dataloader
    print("loading the dataset ...")
    dataset_train = myDataset(file_train_positive_path, file_train_negative_path, fb40_train_mean, fb40_train_var)
    dataset_dev_middle = myDataset(file_dev_positive_path_middle, file_dev_negative_path_middle, fb40_train_mean,
                                   fb40_train_var)
    dataset_dev_far = myDataset(file_dev_positive_path_far, file_dev_negative_path_far, fb40_train_mean, fb40_train_var)
    dataloader_train = myDataLoader(dataset=dataset_train,
                                    batch_size=args.minibatchsize_train,
                                    shuffle=True,
                                    num_workers=args.train_num_workers)
    dataloader_dev_middle = myDataLoader(dataset=dataset_dev_middle,
                                         batch_size=args.minibatchsize_dev,
                                         shuffle=False,
                                         num_workers=args.dev_num_workers)
    dataloader_dev_far = myDataLoader(dataset=dataset_dev_far,
                                      batch_size=args.minibatchsize_dev,
                                      shuffle=False,
                                      num_workers=args.dev_num_workers)
    dataloader_dev = [dataloader_dev_middle, dataloader_dev_far]
    # name_dev = ['middle', 'far']
    name_dev = ['middle']
    # name_dev = ['far']
    print("- done.")
    all_file = len(dataloader_train)
    all_file_dev = len(dataloader_dev_middle)
    print("- {} training samples, {} dev middle samples, {} dev far samples".format(len(dataset_train),
                                                                                    len(dataset_dev_middle),
                                                                                    len(dataset_dev_far)))
    print("- {} training batch, {} dev middle batch, {} dev far batch".format(len(dataloader_train),
                                                                              len(dataloader_dev_middle),
                                                                              len(dataloader_dev_far)))

    nnet = KWS_Net(args=args)
    nnet = nnet.cuda()

    # training setups
    optimizer = optim.Adam(nnet.parameters(), lr=args.lr)
    BCE_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0))

    # tensorboard
    total_train_step = 0
    total_test_step = 0
    writer = SummaryWriter("logs_train")

    for iter_ in range(args.end_iter):
        print("-----------第 {} 轮训练开始----------".format(iter_ + 1))

        start_time = time.time()
        running_loss = 0.0
        total_train_step = 0
        nnet.train()
        for audio_feature, data_label_torch, current_frame in dataloader_train:  # 数据，标签，句子的长度current_frame
            audio_feature = audio_feature.cuda()
            data_label_torch = data_label_torch.cuda()
            outputs = nnet(audio_feature, current_frame)
            loss = BCE_loss(outputs, data_label_torch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # writer.add_scalar("train_loss", running_loss, total_train_step)

            audio_feature_mean = torch.mean(audio_feature, dim=2)
            audio_feature_mean = audio_feature_mean.unsqueeze(1)
            audio_feature_mean = audio_feature_mean.transpose(1, 0)

            # print(audio_feature.shape)
            # print(audio_feature_mean.shape)

            writer.add_image("audio_feature_mean", audio_feature_mean, total_train_step)
            total_train_step = total_train_step + 1

        logger.info("Iteration:{0}, loss = {1:.6f} ".format(iter_, running_loss / all_file))

        # eval
        # all_file_dev = [len(dataloader_dev_middle), len(dataloader_dev_far)]
        all_file_dev = [len(dataloader_dev_middle)]
        # all_file_dev = [len(dataloader_dev_far)]
        nnet.eval()
        for i in range(1):
            pre_list, pre_list_d, label_list = [], [], []
            total_test_step = 0
            with torch.no_grad():
                running_loss_dev = 0.0
                pre_sum = 0.0
                for audio_feature_dev, data_label_torch_dev, current_frame_dev in dataloader_dev[i]:
                    audio_feature_dev = audio_feature_dev.cuda()
                    data_label_torch_dev = data_label_torch_dev.cuda()
                    outputs_dev = nnet(audio_feature_dev, current_frame_dev)
                    loss_dev = BCE_loss(outputs_dev, data_label_torch_dev)

                    running_loss_dev += loss_dev.item()
                    outputs_dev_np = (torch.ceil(torch.sigmoid(outputs_dev) - 0.5)).data.cpu().numpy()
                    pre_list.extend(outputs_dev_np[:, 0])
                    label_list.extend(list(data_label_torch_dev.data.cpu().numpy()))

                writer.add_scalar("train_loss", running_loss / all_file, iter_)
                writer.add_scalar("{}_test_loss".format(name_dev[i]), running_loss_dev / all_file_dev[i], iter_)
                # total_test_step = total_test_step + 1

                TP, FP, TN, FN = utils.cal_indicator(pre_list, label_list)

                FAR = FP / (FP + TN)
                FRR = 1 - TP / (TP + FN)
                Score = FAR + FRR

                writer.add_scalar("FAR", FAR, iter_)
                writer.add_scalar("FRR", FRR, iter_)
                writer.add_scalar("Score", Score, iter_)

                logger.info("%s Epoch:%d, Train loss=%.4f, Valid loss=%.4f, FAR=%.4f, FRR:%.4f, Score:%.4f" % (name_dev[i], iter_,
                                                                                                   running_loss / all_file,
                                                                                                   running_loss_dev /
                                                                                                   all_file_dev[i], FAR,
                                                                                                   FRR, Score))

        torch.save(nnet.state_dict(), os.path.join(model_path, "{}_{}.model".format(args.model_name, iter_)))
        nnet.train()

        end_time = time.time()
        logger.info("Time used for each epoch training: {} seconds.".format(end_time - start_time))
        logger.info("*" * 50)
    writer.close()
    for param_tensor in nnet.state_dict():
        print(param_tensor, "\t", nnet.state_dict()[param_tensor].size())


def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


if __name__ == "__main__":
    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=2e-4, type=float, help="learning rate")
    parser.add_argument("--minibatchsize_train", default=4, type=int)
    parser.add_argument("--minibatchsize_dev", default=1, type=int)
    parser.add_argument("--input_dim", default=256, type=int)
    parser.add_argument("--hidden_sizes", default=256, type=int)
    parser.add_argument("--output_dim", default=1, type=int)
    parser.add_argument("--lstm_num_layers", default=1, type=int)
    parser.add_argument("--seed", default=617, type=int)
    parser.add_argument("--project", default='audio_model', type=str)
    parser.add_argument("--logdir", default='./log/', type=str)
    parser.add_argument("--model_name", default='kws_audio', type=str)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=13, type=int)
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--train_num_workers", default=2, type=int, help="number of training workers")
    parser.add_argument("--dev_num_workers", default=1, type=int, help="number of validation workers")

    parser.add_argument('--every-frame', default=False, action='store_true', help='predicition based on every frame')
    parser.add_argument('--path', default='', help='path to model')
    parser.add_argument('--mode', default='backendLSTM', help='resnet, backendLSTM, finetuneLSTM')
    args = parser.parse_args()

    main(args)
