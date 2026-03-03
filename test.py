from __future__ import print_function
import os
import sys

import logging
import argparse
import time
import yaml
import copy
from time import strftime
import torch
import torch.optim as optim
from torchvision import datasets, transforms

# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
# torch.set_deterministic(True)
import models
import admm
from admm import GradualWarmupScheduler
from admm import CrossEntropyLossMaybeSmooth
from admm import mixup_data, mixup_criterion
from testers import *
from utils import *
from new_TrainValTest import CVTrainValTest

from torch.utils.tensorboard import SummaryWriter

# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
# torch.set_deterministic(True)
np.set_printoptions(threshold=False)
from Server import federated_train

WRITER = SummaryWriter()


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def check_and_create(dir_path):
    if os.path.exists(dir_path):
        return True
    else:
        os.makedirs(dir_path)
        return False


# Training settings
parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument(
    "--logger", action="store_true", default=True, help="whether to use logger"
)
parser.add_argument(
    "--arch", type=str, default=None, help="[vgg, resnet, convnet, alexnet]"
)
parser.add_argument(
    "--depth",
    default=None,
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
    "--batch-size",
    type=int,
    default=64,
    metavar="N",
    help="input batch size for training (default: 32)",
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
    "--lr-decay",
    type=int,
    default=30,
    metavar="LR_decay",
    help="how many every epoch before lr drop (default: 30)",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
    help="SGD momentum (default: 0.9)",
)
parser.add_argument(
    "--weight-decay",
    "--wd",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
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
    "--save-model", type=str, default="", help="For Saving the current Model"
)
parser.add_argument(
    "--save-comon-model", type=str, default="", help="For Saving the common Model"
)
parser.add_argument(
    "--verbose",
    action="store_true",
    default=True,
    help="whether to report admm convergence condition",
)

parser.add_argument(
    "--lr-scheduler", type=str, default="default", help="define lr scheduler"
)
parser.add_argument(
    "--warmup", action="store_true", default=False, help="warm-up scheduler"
)
parser.add_argument(
    "--warmup-lr",
    type=float,
    default=0.0001,
    metavar="M",
    help="warmup-lr, smaller than original lr",
)
parser.add_argument(
    "--warmup-epochs",
    type=int,
    default=0,
    metavar="M",
    help="number of epochs for lr warmup",
)
parser.add_argument("--mixup", action="store_true", default=False, help="ce mixup")
parser.add_argument(
    "--alpha",
    type=float,
    default=0.0,
    metavar="M",
    help="for mixup training, lambda = Beta(alpha, alpha) distribution. Set to 0.0 to disable",
)
parser.add_argument("--smooth", action="store_true", default=False, help="lable smooth")
parser.add_argument(
    "--smooth-eps",
    type=float,
    default=0.0,
    metavar="M",
    help="smoothing rate [0.0, 1.0], set to 0.0 to disable",
)
parser.add_argument(
    "--no-tricks",
    action="store_true",
    default=False,
    help="disable all training tricks and restore original classic training process",
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
parser.add_argument("--base_path", default="", type=str, help="Specify the data path")
parser.add_argument("--save_path", default="", type=str, help="Specify the save path")
parser.add_argument("--input_size", default=32, type=int, help="Specify the input size")
parser.add_argument(
    "--classes", default=10, type=int, help="Specify the number of classes"
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
    "--load-cummu-model",
    type=str,
    default="",
    help="For loading exist pure trained Model",
)

####
parser.add_argument(
    "--sensetivity",
    default=0.02,
    type=float,
    help="sensetivity for stopping and pruning ratio",
)
parser.add_argument(
    "--dynamic", default=False, type=str2bool, help="dynamic pruning ration"
)
parser.add_argument(
    "--comms-round",
    type=int,
    default=100,
    metavar="N",
    help="number of communication round (default: 100)",
)
parser.add_argument(
    "--learning-mode",
    type=str,
    default="centralized",
    help="centralized or federated",
)
parser.add_argument(
    "--clients",
    nargs="*",
    default=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    help="clients included",
    choices=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
)

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.manual_seed(args.seed)
use_cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")
kwargs = {"num_workers": args.workers, "pin_memory": True} if args.cuda else {}
print("Using CUDA: {}".format(use_cuda))
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Save path
args.save_path_exp = os.path.join(args.save_path, args.exp_name)
check_and_create(args.save_path_exp)
setting_file = os.path.join(args.save_path_exp, args.exp_name + "_proposal.config")

print("*************** Configuration ***************")
with open(setting_file, "w") as f:
    args_dic = vars(args)
    for arg, value in args_dic.items():
        line = arg + " : " + str(value)
        print(line)
        f.write(line + "\n")

# set up model archetecture
print("args", args)

""" disable all bag of tricks"""
if args.no_tricks:
    # disable all trick even if they are set to some value
    args.lr_scheduler = "default"
    args.warmup = False
    args.mixup = False
    args.smooth = False
    args.alpha = 0.0
    args.smooth_eps = 0.0

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
if __name__ == "__main__":
    """
    Parallelly Training model with tasks of data and shared model.
    """
    start_time = time.time()
    # Load data
    base_path = None
    save_path = None
    # check_and_create(save_path)
    # Todo : add common data
    base_common_path = os.path.join(args.base_path, "task_common")
    save_common_path = os.path.join(args.save_path_exp, "task_common")
    check_and_create(save_common_path)
    pipeline = CVTrainValTest(
        base_path=base_path,
        save_path=save_path,
        base_common_path=base_common_path,
        save_common_path=save_common_path,
    )
    if args.dataset == "cifar":
        train_loader = pipeline.load_data_cifar(args.batch_size)
    elif args.dataset == "mnist":
        train_loader = pipeline.load_data_mnist(args.batch_size)
    elif args.dataset == "mixture":
        args, train_loader = pipeline.load_data_mixture(args.batch_size)
    elif args.dataset == "flash":
        # train_loader, test_loader = pipeline.load_data_flash(args.batch_size)
        # TODO: add common data loader
        train_common_loader, val_common_loader, test_common_loader = (
            pipeline.load_data_common_flash(args.batch_size)
        )

    # Load model
    cummu_model = model_loader(
        args, 0
    )  # if args.adaptive_mask import from masknet otherwise models/flash_net
    cummu_model.cuda()
    print(cummu_model)

    args = load_layer_config(args, cummu_model, 0)

    """
    Loading cummu_model from the previous task
    """
    if not args.load_cummu_model:
        raise Exception("No model to load")

    # Loading mask for each task from the previous task
    task0_save_path = os.path.join(args.save_path_exp, "task" + str(0))
    task0_mask = pickle.load(open(task0_save_path + "/mask.pkl", "rb"))
    test_sparsity_mask(args, task0_mask)
    task1_save_path = os.path.join(args.save_path_exp, "task" + str(1))
    task1_mask = pickle.load(open(task1_save_path + "/mask.pkl", "rb"))
    test_sparsity_mask(args, task1_mask)
    prev_cummu_mask = pickle.load(open(task1_save_path + "/cumu_mask.pkl", "rb"))
    task2_mask = mask_reverse(args, prev_cummu_mask)

    """
    Test the model
    """
    model_common = model_loader(
        args, 0, common=True
    )  # if args.adaptive_mask import from masknet otherwise models/flash_net
    model_common.cuda()
    """
    Loading common model
    """
    print("*************** Testing ***************")
    cummu_model.load_state_dict(torch.load(args.load_cummu_model))
    task0_model = copy.deepcopy(cummu_model)
    set_model_mask(task0_model, task0_mask)
    task1_model = copy.deepcopy(cummu_model)
    set_model_mask(task1_model, task1_mask)
    task2_model = copy.deepcopy(cummu_model)
    set_model_mask(task2_model, task2_mask)

    """
    Loading common model and individual retrained models
    """
    model_common = model_loader(
        args, 0, common=True
    )  # if args.adaptive_mask import from masknet otherwise models/flash_net
    model_common.cuda()
    print(model_common)
    common_save_path = os.path.join(args.save_path_exp, "task_common")
    model_common.load_state_dict(
        torch.load(common_save_path + "/{}{}.pt".format(args.arch, args.depth))
    )

    ## Start loading individual retrained models
    task0_save_path = os.path.join(args.save_path_exp, "task" + str(0))
    task0_model.load_state_dict(
        torch.load(
            task0_save_path + "/retrained.pt"
        )
    )
    task1_save_path = os.path.join(args.save_path_exp, "task" + str(1))
    task1_model.load_state_dict(
        torch.load(
            task1_save_path + "/retrained.pt"
        )
    )
    task2_save_path = os.path.join(args.save_path_exp, "task" + str(2))
    task2_model.load_state_dict(
        torch.load(
            task2_save_path + "/{}{}.pt".format(args.arch, args.depth)
        )
    )

    """
    Testing the common model
    """
    common_prec1 = pipeline.validate_model(
        args,
        task0_model,
        task1_model,
        task2_model,
        model_common,
        test_common_loader,
        [0, 1, 2],
    )

    _prec1 = pipeline.validate_model(
        args,
        task0_model,
        task1_model,
        task2_model,
        model_common,
        test_common_loader,
        [0, 1],
    )
    _prec1 = pipeline.validate_model(
        args,
        task0_model,
        task1_model,
        task2_model,
        model_common,
        test_common_loader,
        [0, 2],
    )
    _prec1 = pipeline.validate_model(
        args,
        task0_model,
        task1_model,
        task2_model,
        model_common,
        test_common_loader,
        [1, 2],
    )
    _prec1 = pipeline.validate_model(
        args,
        task0_model,
        task1_model,
        task2_model,
        model_common,
        test_common_loader,
        [0],
    )
    _prec1 = pipeline.validate_model(
        args,
        task0_model,
        task1_model,
        task2_model,
        model_common,
        test_common_loader,
        [1],
    )
    _prec1 = pipeline.validate_model(
        args,
        task0_model,
        task1_model,
        task2_model,
        model_common,
        test_common_loader,
        [2],
    )
