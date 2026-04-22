import torch
from torch.utils.data import DataLoader
import pandas as pd
from RQVAE.rqvae import RQVAE
from POIdataset import EmbDataset
import csv
from collections import Counter
import os
import argparse
import random
import numpy as np
import logging
from tqdm import tqdm
from utils import set_color


def str2bool(v):
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
    parser.add_argument('--epochs', type=int, default=3000, help='number of epochs') # 这个参数好像没用
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
    parser.add_argument("--use_bridge", type=int, default=False, help="use bridge or not (0/1, true/false)")

    parser.add_argument("--device", type=str, default="cuda:7", help="gpu or cpu")
    parser.add_argument("--seed", type=int, default=2024, help="random seed")

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[32,32,32], help='emb num of every vq') # NYC 是 32，其他 2 个是 64
    parser.add_argument('--e_dim', type=int, default=64, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument("--beta", type=float, default=0.25, help="Beta for commitment loss")
    parser.add_argument("--lamda", type=float, default=0, help="Lamda for diversity loss")
    parser.add_argument('--layers', type=int, nargs='+', default=[512, 256, 128],
                        help='hidden sizes of every layer')

    parser.add_argument('--save_limit', type=int, default=5)
    parser.add_argument("--ckpt_dir", type=str, default="save", help="output directory for model")
    parser.add_argument("--version", type=str, default="v0", help="version")
    parser.add_argument(
        "--use_geo_emb",
        type=str2bool,
        default=False,
        help="Must match training: concatenate geo_emb from CSV (0/1)",
    )
    parser.add_argument(
        "--geo_emb_col",
        type=str,
        default="geo_emb",
        help="Column name for geo embedding (must match training)",
    )
    parser.add_argument(
        "--use_catname",
        type=str2bool,
        default=True,
        help="Override when checkpoint has no field; else taken from checkpoint",
    )
    parser.add_argument(
        "--use_region",
        type=str2bool,
        default=True,
        help="Override when checkpoint has no field; else taken from checkpoint",
    )

    args = parser.parse_args()
    if args.data_path is None:
        args.data_path = os.path.join("datasets", args.data_mode, "poi_info.csv")
    return args


if __name__ == '__main__':
    cli = parse_args()
    seed = cli.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_mode = cli.data_mode
    if data_mode == "NYC" or data_mode == "TKY":
        cli.num_emb_list = [32, 32, 32]
    elif data_mode == "CA":
        cli.num_emb_list = [64, 64, 64]
    else:
        raise ValueError("Invalid data mode. Choose from 'NYC', 'TKY', or 'CA'.")

    best_collision_ckpt = "best_collision_model.pth"
    current_dir = os.getcwd()
    best_collision_ckpt_file = (
        cli.ckpt_dir + f"/{data_mode}/{cli.version}/{cli.lamda}/{best_collision_ckpt}"
    )

    checkpoint = torch.load(
        best_collision_ckpt_file, map_location=cli.device, weights_only=False
    )

    args = checkpoint["args"]
    use_geo_emb = getattr(args, "use_geo_emb", cli.use_geo_emb)
    geo_emb_col = getattr(args, "geo_emb_col", cli.geo_emb_col)
    use_catname = getattr(args, "use_catname", cli.use_catname)
    use_region = getattr(args, "use_region", cli.use_region)
    data_path = cli.data_path

    print("=================================================")
    print("CLI:", cli)
    print(
        "Checkpoint use_geo_emb=%s use_catname=%s use_region=%s geo_emb_col=%s data_path=%s"
        % (use_geo_emb, use_catname, use_region, geo_emb_col, data_path)
    )
    print("=================================================")

    logging.basicConfig(level=logging.DEBUG)

    data = EmbDataset(
        data_path,
        use_geo_emb=use_geo_emb,
        geo_emb_col=geo_emb_col,
        use_catname=use_catname,
        use_region=use_region,
    )
    input_dim = data[0][1].shape[0]
    data_loader = DataLoader(
        data,
        num_workers=cli.num_workers,
        batch_size=cli.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    args.device = cli.device
    model = RQVAE(
            in_dim=input_dim, 
            num_emb_list=args.num_emb_list, 
            e_dim=args.e_dim,
            layers=args.layers,
            dropout_prob=args.dropout_prob,
            bn=args.bn,
            loss_type=args.loss_type,
            quant_loss_weight=args.quant_loss_weight,
            kmeans_init=args.kmeans_init,
            kmeans_iters=args.kmeans_iters,
            sk_epsilons=args.sk_epsilons,
            sk_iters=args.sk_iters,
            use_linear=args.use_liner,
            use_sk=args.use_sk,
            beta=args.beta,
            diversity_loss=args.lamda,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(args.device)
    model.eval()

    codebooks = {}
    vectors = {}

    iter_data = tqdm(
                data_loader,
                total=len(data_loader),
                ncols=100,
                desc=set_color(f"Generate codebooks ", "pink"),
                )
    
    for batch_idx, data in enumerate(iter_data):
            pids, data = data[0], data[1]
            pids = pids.tolist()
            data = data.to(args.device)
            vector, indices, _ = model.get_indices(data)
            for indx, poi in enumerate(pids):
                codebooks[poi] = indices[indx].tolist()
                vectors[poi] = vector[indx].tolist()

    # print(codebooks)

    value_counts = Counter(tuple(value) for value in codebooks.values())
    seen_values = {}

    updated_dict = {}
    for key, value in codebooks.items():
        value_tuple = tuple(value)  
        if value_counts[value_tuple] > 1: 
            if value_tuple not in seen_values:
                seen_values[value_tuple] = 0
            else:
                seen_values[value_tuple] += 1
            updated_dict[key] = value + [seen_values[value_tuple]]
        else:
            updated_dict[key] = value 

    csv_file = current_dir+f"/datasets/{data_mode}/codebooks_{args.version}_{args.lamda}.csv"


    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        writer.writerow(["Pid", "Codebook", "Vector"])

        for key, value in updated_dict.items():
            writer.writerow([key, value, vectors[key]])
