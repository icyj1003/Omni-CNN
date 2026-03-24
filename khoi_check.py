from __future__ import print_function
import json
import os
import argparse
import time
import copy
import pickle

import torch
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils import (
    model_loader,
    load_layer_config,
    check_and_create,
    str2bool,
)
from new_TrainValTest import CVTrainValTest

# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
# torch.set_deterministic(True)
np.set_printoptions(threshold=False)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument(
    "--logger", action="store_true", default=True, help="whether to use logger"
)
parser.add_argument(
    "--arch", type=str, default="flashnet", help="[vgg, resnet, convnet, alexnet]"
)
parser.add_argument(
    "--depth",
    default=10,
    type=int,
    help="depth of the neural network, 16,19 for vgg; 18, 50 for resnet",
)
parser.add_argument(
    "--s", type=float, default=0.0001, help="scale sparse rate (default: 0.0001)"
)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--multi-gpu", action="store_true", default=False, help="for multi-gpu training"
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=256,
    metavar="N",
    help="input batch size for testing (default: 256)",
)
parser.add_argument(
    "--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.1)"
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=100,
    metavar="N",
    help="how many batches to wait before logging training status",
)

parser.add_argument(
    "--verbose",
    action="store_true",
    default=True,
    help="whether to report admm convergence condition",
)

# For Dataset
parser.add_argument(
    "--dataset",
    default="cifar",
    type=str,
    help="Specify the dataset type [cifar;mnist]",
)
parser.add_argument(
    "--exp_name", default="exp1", type=str, help="Specify the experiment name"
)
parser.add_argument(
    "--base_path",
    default="flash_GPS_Image_LiDAR",
    type=str,
    help="Specify the data path",
)
parser.add_argument(
    "--load_path", default="./centralized/", type=str, help="Specify the load path"
)
parser.add_argument("--input_size", default=32, type=int, help="Specify the input size")
parser.add_argument(
    "--classes", default=64, type=int, help="Specify the number of classes"
)
# Life Long Learning
parser.add_argument("--tasks", type=int, default=1, help="number of tasks")
parser.add_argument(
    "--config-setting",
    metavar="N",
    default="1",
    help="If use manually setting, please set prune ratio for each task. Ex, for 5 tasks --config-setting 2,2,2,2,2",
)
parser.add_argument(
    "--config-shrink",
    type=float,
    default=1,
    help="set the ratio of total model capacity to use",
)
parser.add_argument(
    "--adaptive-mask", default=False, type=str2bool, help="adaptive learning the mask"
)

parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    metavar="N",
    help="number of epochs to train base model (default: 160)",
)

parser.add_argument(
    "--results-path",
    type=str,
    default="./experiments/re_evaluation/",
    help="path to save results",
)

parser.add_argument(
    "--no-tricks",
    action="store_true",
    default=False,
    help="disable all training tricks and restore original classic training process",
)

args = parser.parse_args()

# device setting
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.manual_seed(args.seed)
use_cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")
kwargs = {"num_workers": args.workers, "pin_memory": True} if args.cuda else {}
print("Using CUDA: {}".format(use_cuda))

""" disable all bag of tricks"""
if args.no_tricks:
    # disable all trick even if they are set to some value
    args.lr_scheduler = "default"
    args.warmup = False
    args.mixup = False
    args.smooth = False
    args.alpha = 0.0
    args.smooth_eps = 0.0

if __name__ == "__main__":
    """
    Parallelly Training model with tasks of data and shared model.
    """

    start_time = time.time()
    # Load data
    base_path = os.path.join(args.base_path, "task" + str(3))

    # add common data
    base_common_path = os.path.join(args.base_path, "task_common")

    # create data pipeline
    pipeline = CVTrainValTest(
        base_path=base_path,
        base_common_path=base_common_path,
    )

    # create data loader
    if args.dataset == "cifar":
        train_loader = pipeline.load_data_cifar(args.batch_size)
    elif args.dataset == "mnist":
        train_loader = pipeline.load_data_mnist(args.batch_size)
    elif args.dataset == "mixture":
        args, train_loader = pipeline.load_data_mixture(args.batch_size)
    elif args.dataset == "flash":
        _, _, test_common_loader = pipeline.load_data_common_flash(args.test_batch_size)

    # create genetic cummu_model
    cummu_model = model_loader(args, 0)
    cummu_model.cuda()

    args = load_layer_config(args, cummu_model, 0)

    # load lidar state dict
    lidar_save_path = os.path.join(
        args.load_path, args.exp_name, "flash", "task" + str(0)
    )
    lidar_state_dict = torch.load(
        lidar_save_path + "/retrained_{}{}.pt".format(args.arch, args.depth)
    )

    # create lidar model
    lidar_model = copy.deepcopy(cummu_model)
    lidar_model.load_state_dict(lidar_state_dict)
    lidar_model.cuda()

    # load lidar state dict
    img_save_path = os.path.join(
        args.load_path, args.exp_name, "flash", "task" + str(1)
    )
    img_state_dict = torch.load(
        img_save_path + "/retrained_{}{}.pt".format(args.arch, args.depth)
    )

    # create img model
    img_model = copy.deepcopy(cummu_model)
    img_model.load_state_dict(img_state_dict)
    img_model.cuda()

    # load gps state dict
    gps_save_path = os.path.join(
        args.load_path, args.exp_name, "flash", "task" + str(2)
    )
    gps_state_dict = torch.load(
        gps_save_path + "/retrained_{}{}.pt".format(args.arch, args.depth)
    )

    # create gps model
    gps_model = copy.deepcopy(cummu_model)
    gps_model.load_state_dict(gps_state_dict)
    gps_model.cuda()

    # create common model
    model_common = model_loader(args, 0, common=True)

    # load common state dict
    common_save_path = os.path.join(
        args.load_path, args.exp_name, "flash", "task_common"
    )
    common_state_dict = torch.load(
        common_save_path + "/last_{}{}.pt".format(args.arch, args.depth)
    )
    model_common.load_state_dict(common_state_dict)
    model_common.cuda()

    acc_LIG = pipeline.validate_model(
        args,
        lidar_model,
        img_model,
        gps_model,
        model_common,
        test_common_loader,
        [0, 1, 2],  # LIG
    )

    acc_LI = pipeline.validate_model(
        args,
        lidar_model,
        img_model,
        gps_model,
        model_common,
        test_common_loader,
        [0, 1],  # LI
    )
    acc_LG = pipeline.validate_model(
        args,
        lidar_model,
        img_model,
        gps_model,
        model_common,
        test_common_loader,
        [0, 2],  # LG
    )
    acc_IG = pipeline.validate_model(
        args,
        lidar_model,
        img_model,
        gps_model,
        model_common,
        test_common_loader,
        [1, 2],  # IG
    )
    acc_L = pipeline.validate_model(
        args,
        lidar_model,
        img_model,
        gps_model,
        model_common,
        test_common_loader,
        [0],  # L
    )
    acc_I = pipeline.validate_model(
        args,
        lidar_model,
        img_model,
        gps_model,
        model_common,
        test_common_loader,
        [1],  # I
    )
    acc_G = pipeline.validate_model(
        args,
        lidar_model,
        img_model,
        gps_model,
        model_common,
        test_common_loader,
        [2],  # G
    )

    results = {
        "acc_LIG": acc_LIG.item(),
        "acc_LI": acc_LI.item(),
        "acc_LG": acc_LG.item(),
        "acc_IG": acc_IG.item(),
        "acc_G": acc_G.item(),
        "acc_I": acc_I.item(),
        "acc_L": acc_L.item(),
    }

    result_save_path = os.path.join(args.results_path, args.exp_name)
    check_and_create(result_save_path)
    with open(
        os.path.join(
            result_save_path,
            f"results_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.json",
        ),
        "w",
    ) as f:
        json.dump(results, f)

# CUDA_VISIBLE_DEVICES=0 python khoi_check.py --dataset flash --exp_name ILG_S1
