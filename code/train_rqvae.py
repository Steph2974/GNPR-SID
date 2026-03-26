import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from RQVAE.rqvae import RQVAE
from POIdataset import EmbDataset
import os
import random
import argparse
from trainer import Trainer
import numpy as np
import logging
import sys
import atexit
from utils import setup_logging


def str2bool(v):
    """把命令行字符串正确解析为 bool，支持 0/1、true/false、yes/no 等。"""
    if isinstance(v, bool):
        return v
    s = str(v).lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"需要布尔值，得到: {v}")


def parse_args():

    parser = argparse.ArgumentParser(description="Index")

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=3000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_workers', type=int, default=8, )
    parser.add_argument('--eval_step', type=int, default=50, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument('--lr_scheduler_type', type=str, default="constant", help='scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='warmup epochs')
    parser.add_argument("--data_mode", type=str, default="NYC", help="data mode")
    parser.add_argument("--data_path", type=str, default=None, help="Input data path; default datasets/{data_mode}/poi_info.csv")

    parser.add_argument("--weight_decay", type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.1, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type") # mse, l1
    parser.add_argument("--kmeans_init", type=str2bool, default=False, help="use kmeans_init or not (0/1, true/false)")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--use_sk', type=bool, default=False, help="use sinkhorn or not")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.003], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")
    parser.add_argument("--use-liner", type=int, default=0, help="use-liner")
    parser.add_argument("--use_bridge", type=str2bool, default=False, help="use bridge or not (0/1, true/false)")

    parser.add_argument("--device", type=str, default="cuda:0", help="gpu or cpu") # cuda:0 is the first GPU

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[32,32,32], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=64, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument("--beta", type=float, default=0.25, help="Beta for commitment loss")
    parser.add_argument("--lamda", type=float, default=0, help="Lamda for diversity loss")
    parser.add_argument('--layers', type=int, nargs='+', default=[512, 256, 128],
                        help='hidden sizes of every layer')

    parser.add_argument('--save_limit', type=int, default=5)
    parser.add_argument("--ckpt_dir", type=str, default="save", help="output directory for model")
    parser.add_argument("--version", type=str, default="v1", help="version")

    args = parser.parse_args()
    if args.data_path is None:
        args.data_path = os.path.join("datasets", args.data_mode, "poi_info.csv")
    return args



if __name__ == '__main__':
    """fix the random seed"""
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    args = parse_args()
    data_mode = args.data_mode
    if data_mode == "NYC" or data_mode == "TKY":
        args.num_emb_list = [32,32,32]
    elif data_mode == "CA":
        args.num_emb_list = [64,64,64]
    else:
        raise ValueError("Invalid data mode. Choose from 'NYC', 'TKY', or 'CA'.")   
    log_file = setup_logging(args, data_mode)
    print("=================================================")
    print(args)
    print("=================================================")
    logging.info("Log file: %s", log_file)
    """build dataset"""
    data = EmbDataset(args.data_path)
    input_dim = data[0][1].shape[0]
    model = RQVAE(
            in_dim=input_dim, # 输入维度，即 POI 特征向量的维度
            num_emb_list=args.num_emb_list, # 每层 codebook 的大小, 默认 [32,32,32], 表示 3 层 codebook
            e_dim=args.e_dim, # 输出维度, 默认 64
            layers=args.layers, # 每层 MLP 的维度, 默认 [512, 256, 128], 表示 3 层 MLP
            dropout_prob=args.dropout_prob, # dropout 概率
            bn=args.bn, # 是否使用 batch normalization
            loss_type=args.loss_type, # 损失类型, mse 或 l1
            quant_loss_weight=args.quant_loss_weight, # 量化损失的权重
            kmeans_init=args.kmeans_init, # 是否使用 kmeans 初始化
            kmeans_iters=args.kmeans_iters, # kmeans 迭代次数
            sk_epsilons=args.sk_epsilons, # sinkhorn 迭代次数
            sk_iters=args.sk_iters, # sinkhorn 最大迭代次数
            use_linear=args.use_liner, # 是否使用线性量化, 0 或 1
            use_sk=args.use_sk, # 是否使用 sinkhorn 量化, 0 或 1
            beta=args.beta, # beta 损失的权重
            diversity_loss=args.lamda, # 多样性损失的权重
            use_bridge=args.use_bridge,
    )
    data_loader = DataLoader(data, num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True,
                             pin_memory=True)

    trainer = Trainer(args, model, len(data_loader))
    best_loss, best_collision_rate = trainer.fit(data_loader)

    logging.info("Best Loss: %f", best_loss)
    logging.info("Best Collision Rate: %f", best_collision_rate)
