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

np.set_printoptions(threshold=False)


# python main.py --dataset cifar --exp_name cifar --base_path /home/batool/LPSforLifelong/datasets/cifar/ --save_path /home/batool/LPSforLifelong/datasets/ --arch cifarnet --depth 10 --tasks 3 --epochs 5 --sparsity-type irregular  --epochs-prune 2 --epochs-mask-retrain 3
# python main.py --dataset flash --exp_name flash --base_path /home/batool/flash_LPSforLifelong/datasets/flash/ --save_path /home/batool/flash_LPSforLifelong/datasets/ --arch flashnet --depth 10 --tasks 3 --epochs 5 --sparsity-type irregular  --epochs-prune 2 --epochs-mask-retrain 3
# CUDA_VISIBLE_DEVICES=2 python main.py --dataset flash --exp_name flash --base_path /home/salehihikoueib/flash_LPSforLifelong/datasets/flash/ --save_path /home/salehihikoueib/flash_LPSforLifelong/experiments/testing/ --load-model '' --load-model-pruned '' --classes 64  --sparsity-type irregular --epochs 3 --epochs-prune 3 --epochs-mask-retrain 3 --admm-epochs 1 --mask-admm-epochs 3  --rho 0.01  --mixup --alpha 0 --smooth --smooth-eps 0.1 --config-setting 3,5,2 --adaptive-mask False --adaptive-ratio 0 --arch flashnet --depth 10 --tasks 3
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
parser = argparse.ArgumentParser(description="PyTorch CIFAR10 admm training")
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
    default=32,
    metavar="N",
    help="input batch size for training (default: 64)",
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=256,
    metavar="N",
    help="input batch size for testing (default: 256)",
)
parser.add_argument(
    "--admm-epochs",
    type=int,
    default=1,
    metavar="N",
    help="number of interval epochs to update admm (default: 1)",
)
parser.add_argument(
    "--optmzr",
    type=str,
    default="sgd",
    metavar="OPTMZR",
    help="optimizer used (default: adam)",
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
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
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
    "--masked-retrain", action="store_true", default=False, help="for masked retrain"
)
parser.add_argument(
    "--verbose",
    action="store_true",
    default=True,
    help="whether to report admm convergence condition",
)
parser.add_argument(
    "--admm", action="store_true", default=False, help="for admm training"
)
parser.add_argument("--rho", type=float, default=0.0001, help="define rho for ADMM")
parser.add_argument(
    "--rho-num", type=int, default=3, help="define how many rohs for ADMM training"
)
parser.add_argument(
    "--multi_rho",
    type=bool,
    default=True,
    help="define how many rohs for ADMM training/updated",
)
parser.add_argument(
    "--sparsity-type",
    type=str,
    default="random-pattern",
    help="define sparsity_type: [irregular,column,filter,pattern,random-pattern]",
)
parser.add_argument(
    "--combine-progressive",
    default=False,
    type=str2bool,
    help="for filter pruning after column pruning",
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

########### For Dataset
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
################## Lifelong Learning #####################
parser.add_argument("--tasks", type=int, default=1, help="number of tasks")
parser.add_argument(
    "--mask", type=str, default="", help="loading cumulative Mask"
)  # ???
parser.add_argument("--config-file", type=str, default="", help="config file name")
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
    "--heritage-weight",
    type=str2bool,
    default=False,
    help="use previous weights for current tasks",
)
parser.add_argument(
    "--adaptive-mask", default=False, type=str2bool, help="adaptive learning the mask"
)
parser.add_argument(
    "--admm-mask", default=False, type=str2bool, help="adaptive learning the mask"
)
parser.add_argument(
    "--adaptive-ratio", default=1.0, type=float, help="adaptive learning the mask"
)

parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    metavar="N",
    help="number of epochs to train base model (default: 160)",
)
parser.add_argument(
    "--epochs-prune",
    type=int,
    default=100,
    metavar="N",
    help="number of epochs to train admm (default: 160)",
)
parser.add_argument(
    "--epochs-mask-retrain",
    type=int,
    default=100,
    metavar="N",
    help="number of epochs to train mask (default: 160)",
)
parser.add_argument(
    "--mask-admm-epochs",
    type=int,
    default=1,
    metavar="N",
    help="number of interval epochs to update mask admm ",
)

parser.add_argument(
    "--load-model", type=str, default="", help="For loading exist pure trained Model"
)

parser.add_argument(
    "--load-common-model",
    type=str,
    default="",
    help="For loading exist pure trained Model",
)

parser.add_argument(
    "--load-model-pruned", type=str, default="", help="For loading exist pruned Model"
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


args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.manual_seed(args.seed)
use_cuda = not args.no_cuda and torch.cuda.is_available()


device = torch.device("cuda:0" if use_cuda else "cpu")
kwargs = {"num_workers": args.workers, "pin_memory": True} if args.cuda else {}
WRITER = SummaryWriter()
print("Use Cuda:", use_cuda)
# ------------------ save path ----------------------------------------------
args.save_path_exp = os.path.join(args.save_path, args.exp_name)
check_and_create(args.save_path_exp)
setting_file = os.path.join(args.save_path_exp, args.exp_name + ".config")

print("*************** Configuration ***************")
with open(setting_file, "w") as f:
    args_dic = vars(args)
    for arg, value in args_dic.items():
        line = arg + " : " + str(value)
        print(line)
        f.write(line + "\n")

# set up model archetecture
print("args", args)
# model = model_loader(args)
# model.cuda()
# print(model)
""" disable all bag of tricks"""
if args.no_tricks:
    # disable all trick even if they are set to some value
    args.lr_scheduler = "default"
    args.warmup = False
    args.mixup = False
    args.smooth = False
    args.alpha = 0.0
    args.smooth_eps = 0.0


def save_best_model(
    model_dict,
    task,
):
    save_common_path = os.path.join(args.save_path_exp, "task_common")
    # print('check',args.arch, args.depth)
    str_task = "best_task" + "".join([str(i) for i in task])
    torch.save(
        model_dict["model_common"].state_dict(),
        save_common_path + "/{}.pt".format(str_task),
    )
    for i in range(3):
        save_path = os.path.join(args.save_path_exp, "task" + str(i))
        torch.save(
            model_dict[f"model{i}"].state_dict(),
            save_path + "/{}.pt".format(str_task),
        )


def train(
    args,
    pipeline,
    task,
    train_common_loader,
    val_common_loader,
    test_common_loader,
    model_dict,
    mask_dict,
    mode_split,
):  # 1
    print("*************** Training Model ***************")
    optimizer_init_lr = args.lr
    best_acc = 0
    best_acc_dict = {
        "task0": 0,
        "task1": 0,
        "task2": 0,
        "task012": 0,
        "task01": 0,
        "task02": 0,
        "task12": 0,
    }

    # TODO: Creating optimizer for both model and model_common
    compined_params = []
    for key in model_dict.keys():
        if model_dict[key] is not None:
            for param in model_dict[key].parameters():
                param.requires_grad = True
            compined_params += list(model_dict[key].parameters())

    optimizer = torch.optim.Adam(
        compined_params, optimizer_init_lr, weight_decay=0.0005
    )
    # optimizer = torch.optim.Adam(model.parameters(), optimizer_init_lr,betas=(0.9,0.999),weight_decay=0,amsgrad=False)  #2
    criterion = torch.nn.CrossEntropyLoss()
    distribution_loss = torch.nn.L1Loss()

    """
    Set learning rate
    """
    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs * len(train_common_loader), eta_min=4e-08
        )
    elif args.lr_scheduler == "default":
        # my learning rate scheduler for cifar, following https://github.com/kuangliu/pytorch-cifar
        # epoch_milestones = [65, 100, 130, 190, 220, 250, 280]
        epoch_milestones = [50, 75, 95, 115, 140, 165, 190, 220, 250, 280]

        """
        Set the learning rate of each parameter task to the initial lr decayed
        by gamma once the number of epoch reaches one of the milestones
        """
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[i * len(train_common_loader) for i in epoch_milestones],
            gamma=0.5,
        )
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max")
    else:
        # adjust learning rate
        lr = optimizer_init_lr * (0.1 ** (epoch // 25))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    # start training
    for epoch in range(0, args.epochs):
        start = time.time()

        # check availability of the current mask
        model_dict = pipeline.train_model(
            args,
            train_common_loader,
            optimizer,
            criterion,
            distribution_loss,
            epoch,
            scheduler,
            WRITER,
            model_dict["model_common"],
            model_dict["model0"],
            mask_dict["mask0"],
            model_dict["model1"],
            mask_dict["mask1"],
            model_dict["model2"],
            mask_dict["mask2"],
            mode_split,
        )  # for task0 args.mask is none

        end_train = time.time()
        print("***********test accuracy after each epoch*******")
        prec1 = pipeline.validate_model(
            args,
            model_dict["model0"],
            model_dict["model1"],
            model_dict["model2"],
            model_dict["model_common"],
            val_common_loader,
            [task],
        )
        # scheduler.step()
        WRITER.add_scalar(f"test/acc_task{task}", prec1, epoch)
        end_test = time.time()
        print(
            "Training time: {:.3f}; Testing time: {:.3f}.".format(
                end_train - start, end_test - end_train
            )
        )
        print(
            "#############Check if weighst are zeror during training###########",
            len(
                np.nonzero(
                    model_dict[f"model{task}"]
                    .conv1.weight.reshape(32 * 45 * 3 * 3)
                    .cpu()
                    .detach()
                    .numpy()
                )[0]
            ),
        )
        if prec1 > best_acc:  # saving best model
            best_acc = prec1
            save_common_path = os.path.join(args.save_path_exp, "task_common")
            # print('check',args.arch, args.depth)
            torch.save(
                model_dict["model_common"].state_dict(),
                save_common_path + "/{}{}.pt".format(args.arch, args.depth),
            )
            for i in range(task):
                save_path = os.path.join(args.save_path_exp, "task" + str(i))
                torch.save(
                    model_dict[f"model{i}"].state_dict(),
                    save_path + "/retrained.pt".format(args.arch, args.depth),
                )
            save_path = os.path.join(args.save_path_exp, "task" + str(task))
            torch.save(
                model_dict[f"model{task}"].state_dict(),
                save_path + "/{}{}.pt".format(args.arch, args.depth),
            )
        if task == 2:
            prec1 = pipeline.validate_model(
                args,
                model_dict["model0"],
                model_dict["model1"],
                model_dict["model2"],
                model_dict["model_common"],
                val_common_loader,
                [0, 1, 2],
            )
            if prec1 > best_acc_dict["task012"]:
                best_acc_dict["task012"] = prec1
                save_best_model(model_dict, [0, 1, 2])
            prec1 = pipeline.validate_model(
                args,
                model_dict["model0"],
                model_dict["model1"],
                model_dict["model2"],
                model_dict["model_common"],
                val_common_loader,
                [0, 1],
            )
            if prec1 > best_acc_dict["task01"]:
                best_acc_dict["task01"] = prec1
                save_best_model(model_dict, [0, 1])
            prec1 = pipeline.validate_model(
                args,
                model_dict["model0"],
                model_dict["model1"],
                model_dict["model2"],
                model_dict["model_common"],
                val_common_loader,
                [0, 2],
            )
            if prec1 > best_acc_dict["task02"]:
                best_acc_dict["task02"] = prec1
                save_best_model(model_dict, [0, 2])
            prec1 = pipeline.validate_model(
                args,
                model_dict["model0"],
                model_dict["model1"],
                model_dict["model2"],
                model_dict["model_common"],
                val_common_loader,
                [1, 2],
            )
            if prec1 > best_acc_dict["task12"]:
                best_acc_dict["task12"] = prec1
                save_best_model(model_dict, [1, 2])
            prec1 = pipeline.validate_model(
                args,
                model_dict["model0"],
                model_dict["model1"],
                model_dict["model2"],
                model_dict["model_common"],
                val_common_loader,
                [0],
            )
            if prec1 > best_acc_dict["task0"]:
                best_acc_dict["task0"] = prec1
                save_best_model(model_dict, [0])
            prec1 = pipeline.validate_model(
                args,
                model_dict["model0"],
                model_dict["model1"],
                model_dict["model2"],
                model_dict["model_common"],
                val_common_loader,
                [1],
            )
            if prec1 > best_acc_dict["task1"]:
                best_acc_dict["task1"] = prec1
                save_best_model(model_dict, [1])
            prec1 = pipeline.validate_model(
                args,
                model_dict["model0"],
                model_dict["model1"],
                model_dict["model2"],
                model_dict["model_common"],
                val_common_loader,
                [2],
            )
            if prec1 > best_acc_dict["task2"]:
                best_acc_dict["task2"] = prec1
                save_best_model(model_dict, [2])

    print("Best Acc after training: {:.4f}%".format(best_acc))
    print("Best Acc after training dict: ", best_acc_dict)


def prune(args, task, train_common_loader, val_common_loader, mode_split):  # 2
    print("*************** prune Model ***************")

    """====================="""
    """ Initialize submask"""
    """====================="""

    # initialize mask for the first task but args.mask is None for the first task
    if args.adaptive_mask:
        if task > 0:
            set_adaptive_mask(
                model_dict[f"model{task}"], reset=True, requires_grad=True
            )  # set initial as 1
            args.admm_mask = True
        else:
            set_adaptive_mask(
                model_dict[f"model{task}"], reset=True, requires_grad=False
            )

    """====================="""
    """ multi-rho admm train"""
    """====================="""

    args.admm, args.masked_retrain = True, False

    # Trigger for experiment [leave space for future learning]
    if args.admm_mask and task == args.tasks - 1:
        args.admm = False
    admm_prune(
        args,
        args.mask,
        task,
        train_common_loader,
        val_common_loader,
        [task],
    )

    """=============="""
    """masked retrain"""
    """=============="""
    #########fine-tune
    args.admm, args.admm_mask, args.masked_retrain = False, False, True
    return masked_retrain(
        args,
        args.mask,
        task,
        train_common_loader,
        val_common_loader,
        [task],
    )


def admm_prune(
    args, pre_mask, task, train_common_loader, val_common_loader, mode_split
):  # 3
    print("***************admm prune Model ***************")

    """
    bag of tricks set-ups
    """
    initial_rho = args.rho
    criterion = CrossEntropyLossMaybeSmooth(smooth_eps=args.smooth_eps).cuda()
    distribution_loss = torch.nn.L1Loss().cuda()
    args.smooth = args.smooth_eps > 0.0
    args.mixup = args.alpha > 0.0

    optimizer_init_lr = args.warmup_lr if args.warmup else args.lr
    optimizer = None
    # # TODO: Not update the parameters for model_common
    # print("Fixed model_common and other models weights")
    # for key in model_dict.keys():
    #     if key != f"model{task}" and model_dict[key] is not None:
    #         for param in model_dict[key].parameters():
    #             param.requires_grad = False

    compined_params = model_dict[
        f"model{task}"
    ].parameters()  # Test with only one model

    if args.optmzr == "sgd":
        optimizer = torch.optim.SGD(
            compined_params,
            optimizer_init_lr,
            momentum=0.9,
            weight_decay=1e-4,
        )
    elif args.optmzr == "adam":
        optimizer = torch.optim.Adam(
            compined_params,
            optimizer_init_lr,
            betas=(0.9, 0.999),
            weight_decay=0,
            amsgrad=False,
        )  # 2

    """
    Set learning rate
    """
    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs_prune * len(train_common_loader), eta_min=4e-08
        )
    elif args.lr_scheduler == "default":
        # my learning rate scheduler for cifar, following https://github.com/kuangliu/pytorch-cifar
        epoch_milestones = [65, 100, 130, 190, 220, 250, 280]

        """
        Set the learning rate of each parameter task to the initial lr decayed
        by gamma once the number of epoch reaches one of the milestones
        """
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[i * len(train_common_loader) for i in epoch_milestones],
            gamma=0.5,
        )
    else:
        raise Exception("unknown lr scheduler")

    if args.warmup:
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=args.lr / args.warmup_lr,
            total_iter=args.warmup_epochs * len(train_common_loader),
            after_scheduler=scheduler,
        )

    # backup model weights
    if args.heritage_weight or args.adaptive_mask:
        model_backup = copy.deepcopy(model_dict[f"model{task}"].state_dict())

    # get mask for training & set pre-trained (for previous tasks) weights to be zero
    if pre_mask:
        pre_mask = mask_reverse(args, pre_mask)
        set_model_mask(model_dict[f"model{task}"], pre_mask)

    """
    if heritage or adaptive, copy weights back to model
    not for first task
    """
    if args.heritage_weight or args.adaptive_mask:
        if args.mask:
            with torch.no_grad():
                for name, W in model_dict[f"model{task}"].named_parameters():
                    if name in args.pruned_layer:
                        W.data += model_backup[name].data * args.mask[name].cuda()

    ####new_rho
    """
    Start Pruning...
    """
    # for i in range(args.rho_num):    #rho_num 3 and initial_rho =0.01 in my experiments and
    #     current_rho = initial_rho * 10 ** i
    # print('args.config_file',args.config_file,args.prune_ratios,args.config_setting)
    if args.config_file:
        config = "./profile/" + args.config_file + ".yaml"
    elif args.config_setting:
        # Implement the config setting
        config = args.prune_ratios
        # print('config',config)
    else:
        raise Exception("must provide a config setting.")
    ADMM = admm.ADMM(args, model_dict[f"model{task}"], config=config, rho=initial_rho)
    admm.admm_initialization(
        args, ADMM=ADMM, model=model_dict[f"model{task}"]
    )  # intialize Z variable

    # admm train
    best_prec1 = 0.0

    for epoch in range(1, args.epochs_prune + 1):
        # print("current rho: {}".format(current_rho))
        print("config main", config)
        # print('pre_mask',pre_mask)
        if task != 0:
            pre_mask_previous_it = pre_mask
        prune_train(
            args,
            task,
            pre_mask,
            ADMM,
            train_common_loader,
            criterion,
            distribution_loss,
            optimizer,
            scheduler,
            epoch,
            model_dict,
            mode_split,
        )
        if task != 0:
            pre_mask_current_it = pre_mask
            for k in pre_mask_current_it.keys():
                print(
                    "(A==B).all()",
                    k,
                    (pre_mask_previous_it[k] == pre_mask_current_it[k]).all(),
                )
        prec1 = pipeline.validate_model(
            args,
            model_dict["model0"],
            model_dict["model1"],
            model_dict["model2"],
            model_dict["model_common"],
            val_common_loader,
            mode_split,
        )
        best_prec1 = max(prec1, best_prec1)
        admm.admm_adjust_learning_rate(optimizer, epoch, args)  ######added

    print("Best Acc after pruning: {:.4f}%".format(best_prec1))
    save_path = os.path.join(args.save_path_exp, "task" + str(task))
    torch.save(
        model_dict[f"model{task}"].state_dict(),
        save_path
        + "/prunned_{}{}_{}_{}_{}.pt".format(
            args.arch, args.depth, args.config_file, args.optmzr, args.sparsity_type
        ),
    )


def prune_train(
    args,
    task,
    pre_mask,
    ADMM,
    train_common_loader,
    criterion,
    distribution_loss,
    optimizer,
    scheduler,
    epoch,
    model_dict,
    mode_split=None,
):  # 4
    print("***************prune_train function ***************")

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    idx_loss_dict = {}

    for key in model_dict.keys():
        if key == f"model{task}":
            # switch to train mode
            model_dict[key].train()
        elif model_dict[key] is not None:
            # switch to evaluate mode
            model_dict[key].eval()

    end = time.time()
    for i, (input_common, target) in enumerate(train_common_loader):
        # TODO: Implement the training loop for common model
        specific_features = torch.zeros((input_common.shape[0], 3, 64)).float().cuda()
        _input_common = input_common.float().cuda()
        target = target.long().cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate
        if args.admm:
            admm.admm_adjust_learning_rate(optimizer, epoch, args)
        else:
            scheduler.step()

        if args.mixup:
            _input_common, target_a, target_b, lam = mixup_data(
                _input_common, target, args.alpha
            )

        # compute output
        for m in [0, 1, 2]:
            if m not in mode_split:
                _input_common[:, m, :, :, :] = 0
                specific_features[:, m, :] = 0
            elif m == 0:
                d0 = _input_common[:, 0, :, :, :]
                specific_features[:, m, :] = model_dict["model0"](d0)
            elif m == 1:
                d1 = _input_common[:, 1, :, :, :]
                specific_features[:, m, :] = model_dict["model1"](d1)
            elif m == 2:
                d2 = _input_common[:, 2, :, :, :]
                specific_features[:, m, :] = model_dict["model2"](d2)

        input = _input_common
        # TODO: Compute the output for the common model
        input = input.reshape(-1, input.shape[2], input.shape[3], input.shape[4])
        specific_features = specific_features.reshape(
            -1,
            specific_features.shape[2],
        )
        model_dict["model_common"].set_mode_split(mode_split)
        shared_features, final_output = model_dict["model_common"](
            input, specific_features
        )

        # TODO: Compute more loss functions, including cross entropy loss, similarity loss, and sparsity loss
        shared_loss = (
            distribution_loss(
                shared_features[0 : len(shared_features) + 1 : 3],
                shared_features[1 : len(shared_features) + 1 : 3],
            )
            + distribution_loss(
                shared_features[0 : len(shared_features) + 1 : 3],
                shared_features[2 : len(shared_features) + 1 : 3],
            )
            + distribution_loss(
                shared_features[1 : len(shared_features) + 1 : 3],
                shared_features[2 : len(shared_features) + 1 : 3],
            )
        )

        if args.mixup:
            ce_loss = mixup_criterion(
                criterion, final_output, target_a, target_b, lam, args.smooth
            )
        else:
            ce_loss = criterion(final_output, target[:, 0], smooth=args.smooth)

        # Parameters of the model_common are not updated
        # ce_loss += 0.3 * shared_loss
        mixed_loss = ce_loss

        if args.admm:
            admm.z_u_update(
                args,
                task,
                ADMM,
                model_dict[f"model{task}"],
                epoch,
                i,
                WRITER,
            )  # update Z and U
            ce_loss, admm_loss, mixed_loss = admm.append_admm_loss(
                args, ADMM, model_dict[f"model{task}"], ce_loss
            )  # append admm loss
        if args.admm_mask:
            admm.y_k_update(
                args,
                task,
                ADMM,
                model_dict[f"model{task}"],
                epoch,
                i,
                WRITER,
            )  # update Y\K
            ce_loss, admm_loss, mixed_loss = admm.append_mask_loss(
                args, ADMM, model_dict[f"model{task}"], mixed_loss
            )

        # measure accuracy and record loss
        acc1, _ = accuracy(final_output, target, topk=(1, 5))

        losses.update(ce_loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if args.admm or args.admm_mask:
            mixed_loss.backward(retain_graph=True)
        else:
            ce_loss.backward()

        if pre_mask:
            with torch.no_grad():
                for name, W in model_dict[f"model{task}"].named_parameters():
                    # shared layers
                    if name in args.fixed_layer:
                        W.grad *= 0
                        continue

                    # pruned weight layers: fix weight for previous task
                    if name in args.pruned_layer and name in pre_mask:
                        W.grad *= pre_mask[name].cuda()

                    # adaptively learn the mask: fix mask for trainable weight part
                    if args.adaptive_mask and "mask" in name and args.admm:
                        W.grad *= args.mask[name.replace("w_mask", "weight")].cuda()

                        # W.grad *= 100
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0:
            for param_group in optimizer.param_groups:
                current_lr = param_group["lr"]
            print(
                "({0}) lr:[{1:.5f}]  "
                "Epoch: [{2}][{3}/{4}]\t"
                "Status: admm-[{5}] retrain-[{6}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Acc@1 {top1.val:.3f}% ({top1.avg:.3f}%)\t".format(
                    args.optmzr,
                    current_lr,
                    epoch,
                    i,
                    len(train_common_loader),
                    args.admm,
                    args.masked_retrain,
                    batch_time=data_time,
                    loss=losses,
                    top1=top1,
                )
            )
        if i % 100 == 0:
            idx_loss_dict[i] = losses.avg
    # return masks, idx_loss_dict


def masked_retrain(
    args, pre_mask, task, train_common_loader, val_common_loader, mode_split
):  # 5
    print("*************** masked retrain ***************")

    """
    bag of tricks set-ups
    """
    initial_rho = args.rho
    criterion = CrossEntropyLossMaybeSmooth(smooth_eps=args.smooth_eps).cuda()
    distribution_loss = torch.nn.L1Loss().cuda()
    args.smooth = args.smooth_eps > 0.0
    args.mixup = args.alpha > 0.0

    optimizer_init_lr = args.warmup_lr if args.warmup else args.lr
    optimizer = None
    # # TODO: Not update the parameters for model_common
    # print("Fixed model_common and other models weights")
    # for key in model_dict.keys():
    #     if key != f"model{task}" and model_dict[key] is not None:
    #         for param in model_dict[key].parameters():
    #             param.requires_grad = False

    compined_params = model_dict[
        f"model{task}"
    ].parameters()  # Test with only one model

    if args.optmzr == "sgd":
        optimizer = torch.optim.SGD(
            compined_params,
            optimizer_init_lr,
            momentum=0.9,
            weight_decay=1e-4,
        )
    elif args.optmzr == "adam":
        optimizer = torch.optim.Adam(
            compined_params,
            optimizer_init_lr,
            betas=(0.9, 0.999),
            weight_decay=0,
            amsgrad=False,
        )  # 2

    """
    Set learning rate
    """
    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs_mask_retrain * len(train_common_loader),
            eta_min=4e-08,
        )
    elif args.lr_scheduler == "default":
        # my learning rate scheduler for cifar, following https://github.com/kuangliu/pytorch-cifar
        epoch_milestones = [65, 100, 130, 190, 220, 250, 280]

        """
        Set the learning rate of each parameter task to the initial lr decayed
        by gamma once the number of epoch reaches one of the milestones
        """
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[i * len(train_common_loader) for i in epoch_milestones],
            gamma=0.5,
        )
    else:
        raise Exception("unknown lr scheduler")

    if args.warmup:
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=args.lr / args.warmup_lr,
            total_iter=args.warmup_epochs * len(train_common_loader),
            after_scheduler=scheduler,
        )

    """
    load admm trained model
    """
    save_path = os.path.join(args.save_path_exp, "task" + str(task))
    print(
        "Loading prunned model file: "
        + save_path
        + "/prunned_{}{}_{}_{}_{}.pt".format(  ##upadted removed rho
            args.arch, args.depth, args.config_file, args.optmzr, args.sparsity_type
        )
    )
    model_dict[f"model{task}"].load_state_dict(
        torch.load(
            save_path
            + "/prunned_{}{}_{}_{}_{}.pt".format(
                args.arch, args.depth, args.config_file, args.optmzr, args.sparsity_type
            )
        )
    )

    if args.config_file:
        config = "./profile/" + args.config_file + ".yaml"
    elif args.config_setting:
        # Implement the config setting
        config = args.prune_ratios
    else:
        raise Exception("must provide a config setting.")

    ADMM = admm.ADMM(args, model_dict[f"model{task}"], config=config, rho=initial_rho)
    best_prec1 = [0]
    best_mask = ""

    """
    Deal with masks
    """
    if args.heritage_weight or args.adaptive_mask:
        model_backup = copy.deepcopy(model_dict[f"model{task}"].state_dict())

    if pre_mask:
        pre_mask = mask_reverse(args, pre_mask)
        # test_column_sparsity_mask(pre_mask)
        set_model_mask(model_dict[f"model{task}"], pre_mask)

    # Trigger for experiment [leave space for future learning]
    if task != args.tasks - 1:
        admm.hard_prune(args, ADMM, model_dict[f"model{task}"])  # prune weights

    if args.adaptive_mask and args.mask:
        admm.hard_prune_mask(args, ADMM, model_dict[f"model{task}"])  # set submasks

    current_trainable_mask = get_model_mask(model=model_dict[f"model{task}"])
    current_mask = copy.deepcopy(current_trainable_mask)
    submask = {}

    # if heritage, copy weights back to model
    if args.heritage_weight and args.mask:
        with torch.no_grad():
            for name, W in model_dict[f"model{task}"].named_parameters():
                if name in args.pruned_layer:
                    W.data += model_backup[name].data * args.mask[name].cuda()

    # if adaptive learning, copy selected weights back to model
    if args.adaptive_mask and args.mask:
        with torch.no_grad():

            # mask layer: previous tasks part {0,1}; remaining {0}
            for name, M in model_dict[f"model{task}"].named_parameters():
                if "mask" in name:
                    weight_name = name.replace("w_mask", "weight")
                    submask[weight_name] = M.cpu().detach()

            # copy selected weights back to model
            for name, W in model_dict[f"model{task}"].named_parameters():
                if name in args.pruned_layer:

                    """
                    Reason why use args.mask instead of submask
                    1. easy to cumulate model weights, if use submask, then need to backup weights belong to args.mask-submask
                    2. weights 'selective' already achieved by mask layer (fixed during mask retrain)
                    """
                    W.data += model_backup[name].data * args.mask[name].cuda()

            # combine submask and current trainable mask
            for name in submask:
                current_mask[name] += submask[name]

            # mask layer: previous tasks part {0,1}; remaining {1}
            for name, M in model_dict[f"model{task}"].named_parameters():
                if "mask" in name:
                    M.data = current_mask[name.replace("w_mask", "weight")].cuda()

        set_adaptive_mask(model_dict[f"model{task}"], requires_grad=False)

    epoch_loss_dict = {}
    testAcc = []

    """
    Start prunning
    """
    for epoch in range(1, args.epochs_mask_retrain + 1):
        prune_train(
            args,
            task,
            current_trainable_mask,
            ADMM,
            train_common_loader,
            criterion,
            distribution_loss,
            optimizer,
            scheduler,
            epoch,
            model_dict,
            mode_split,
        )
        prec1 = pipeline.validate_model(
            args,
            model_dict["model0"],
            model_dict["model1"],
            model_dict["model2"],
            model_dict["model_common"],
            val_common_loader,
            mode_split,
        )

        if prec1 > max(best_prec1):
            # print("\n>_ Got better accuracy, saving model with accuracy {:.3f}% now...\n".format(prec1))
            torch.save(
                model_dict[f"model{task}"].state_dict(), save_path + "/retrained.pt"
            )

            # save_common_path = os.path.join(args.save_path_exp, "task_common")
            # torch.save(
            #     model_dict["model_common"].state_dict(),
            #     save_common_path + "/{}{}.pt".format(args.arch, args.depth),
            # )

        testAcc.append(prec1)

        best_prec1.append(prec1)
        # print("current best acc is: {:.4f}".format(max(best_prec1)))

    print("Best Acc: {:.4f}%".format(max(best_prec1)))
    print("Pruned Mask sparsity")
    test_sparsity_mask(args, current_trainable_mask)
    print("*****end of masked retrain*******")
    return current_mask


if __name__ == "__main__":

    """
    Consecutively train a model with tasks of data.
    """
    print("**********star runing******************")
    start_time = time.time()
    num_tasks = args.tasks
    model_dict = {"model_common": None, "model0": None, "model1": None, "model2": None}
    mask_dict = {"mask0": None, "mask1": None, "mask2": None}
    for task in range(num_tasks):
        print("\n\n*************** Training task {} ***************".format(task))
        """
        load config (pruning) setting
        """
        model_dict[f"model{task}"] = model_loader(args, task)
        model_dict[f"model{task}"].cuda()
        # TODO: add model_common for common data
        model_dict["model_common"] = model_loader(
            args, task, common=True
        )  # if args.adaptive_mask import from masknet otherwise models/flash_net
        model_dict["model_common"].cuda()
        print(model_dict[f"model{task}"])
        if id(model_dict[f"model{task}"]) == id(model_dict["model_common"]):
            raise Exception("model and model_common are the same object")
        args = load_layer_config(
            args, model_dict[f"model{task}"], task
        )  # returns pruning ratio for each partrio based on config_setting [2,4,4] [1-2/10, 1-4/10,1-4/10]
        mode_split = [i for i in range(task + 1)]
        # mode_split = [task]
        """
        load-data
        """
        base_path = os.path.join(args.base_path, "task" + str(task))
        save_path = os.path.join(args.save_path_exp, "task" + str(task))
        check_and_create(save_path)
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

        """
        Load model
        """
        args.load_common_model = save_common_path + "/{}{}.pt".format(
            args.arch, args.depth
        )
        if os.path.exists(args.load_common_model):
            print("Loading pretrained model from: ", args.load_common_model)
            model_dict["model_common"].load_state_dict(
                torch.load(args.load_common_model)
            )
        else:
            print("Training model_common from scratch")

        for j in range(task + 1):
            s_p = os.path.join(args.save_path_exp, "task" + str(j))
            if os.path.exists(s_p + "/retrained.pt") and os.path.exists(
                s_p + "/mask.pkl"
            ):
                print("Loading pre-trained model from: ", s_p + "/retrained.pt")
                load_state_dict(
                    args,
                    model_dict[f"model{j}"],
                    torch.load(s_p + "/retrained.pt", weights_only=False),
                    target_keys=args.output_layer,
                )
                with open(s_p + "/mask.pkl", "rb") as handle:
                    mask_dict[f"mask{j}"] = pickle.load(handle)
                with open(s_p + "/cumu_mask.pkl", "rb") as handle:
                    args.mask = pickle.load(handle)

        """
        Pure train
        """
        train(
            args,
            pipeline,
            task,
            train_common_loader,
            val_common_loader,
            test_common_loader,
            model_dict,
            mask_dict,
            mode_split,
        )
        # print('check',args.arch, args.depth)
        WRITER.flush()

        model_dict["model_common"].load_state_dict(torch.load(args.load_common_model))

        # load best model of individual task
        model_path = save_path + "/{}{}.pt".format(args.arch, args.depth)
        load_state_dict(args, model_dict[f"model{task}"], torch.load(model_path))
        print(f"Succesfully loaded model{task} from {model_path}")
        for i in range(task):
            _save_path = os.path.join(args.save_path_exp, "task" + str(i))
            _model_path = _save_path + "/retrained.pt".format(args.arch, args.depth)
            model_dict[f"model{i}"].load_state_dict(torch.load(_model_path))
            print(f"Succesfully loaded model{i} from {_model_path}")

        """
        Prune
        """
        # Trigger for experiment [leave space for future learning]
        if task != num_tasks - 1:
            """
            admm prunning based on basic model
            mask_for_current_task: pruned mask for current task i
            if adaptive_mask: mask_for_current_task = pruned + subset of cumulative mask
            """
            if task == 0 and args.load_model_pruned:
                print(f"Loading pre-pruned model from: {args.load_model_pruned}")
                load_state_dict(
                    args, model_dict[f"model{task}"], torch.load(args.load_model_pruned)
                )  # this will be saved as retrained
                mask_for_current_task = get_model_mask(model=model_dict[f"model{task}"])
                torch.save(
                    model_dict[f"model{task}"].state_dict(), save_path + "/retrained.pt"
                )
            else:
                print(f"...............Pruning task {task}................")
                # TODO: Prune model and get mask for current task and retrained both model and model_common
                mask_for_current_task = prune(
                    args, task, train_common_loader, val_common_loader, mode_split
                )
                # print('mask_for_current_task',mask_for_current_task)

            """
            Get mask for this specific task
            """
            print("Total Mask sparsity for task ", str(task))
            test_sparsity_mask(
                args, mask_for_current_task
            )  # returns zero/non-zero statistics after pruning

            cumulative_mask = mask_joint(args, mask_for_current_task, args.mask)
            args.mask = copy.deepcopy(cumulative_mask)

            mask_dict[f"mask{task}"] = copy.deepcopy(mask_for_current_task)
            with open(os.path.join(save_path, "mask.pkl"), "wb") as handle:
                pickle.dump(
                    mask_for_current_task, handle, protocol=pickle.HIGHEST_PROTOCOL
                )
            with open(os.path.join(save_path, "cumu_mask.pkl"), "wb") as handle:
                pickle.dump(cumulative_mask, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Trigger for experiment [leave space for future learning]
        else:
            if args.adaptive_mask:
                print("Total adaptive Mask sparsity for task ", str(task))
                mask_for_current_task = prune(
                    args, task, train_common_loader, val_common_loader, mode_split
                )
                test_sparsity_mask(args, mask_for_current_task)
                with open(os.path.join(save_path, "mask.pkl"), "wb") as handle:
                    pickle.dump(
                        mask_for_current_task, handle, protocol=pickle.HIGHEST_PROTOCOL
                    )
            else:
                print(
                    f"Because args.adaptive_mask is {args.adaptive_mask}, skipping pruning for task {task}"
                )

        """
        Combine & Save Model
        """

        """
        save the best model to be cumulated model
        for the last task, there is no pruning requirement, then save the best purely trained model
        """
        if args.heritage_weight or args.adaptive_mask:
            print(
                f"args.heritage_weight is {args.heritage_weight} or args.adaptive_mask is {args.adaptive_mask}, cumulating model"
            )
            if (
                task != num_tasks - 1
            ):  # Trigger for experiment [leave space for future learning]
                torch.save(
                    torch.load(save_path + "/retrained.pt"),
                    save_path + "/cumu_model.pt",
                )
            else:  # Trigger for experiment [leave space for future learning]
                torch.save(
                    torch.load(save_path + "/{}{}.pt".format(args.arch, args.depth)),
                    save_path + "/cumu_model.pt",
                )
        else:
            print(
                f"args.heritage_weight is {args.heritage_weight} and args.adaptive_mask is {args.adaptive_mask}, saving model"
            )
            cumulate_model(args, task)  # cumulate pruned layers

        """
        Test
        """
        print("*************** Testing ***************")
        try:
            state_dict = torch.load(save_path + "/retrained.pt")
            load_state_dict(
                args,
                model_dict[f"model{task}"],
                state_dict,
                target_keys=args.output_layer,
            )
        except:
            print("No retrained model found, loading cumulated model")
            state_dict = torch.load(save_path + "/cumu_model.pt")
            load_state_dict(
                args,
                model_dict[f"model{task}"],
                state_dict,
                target_keys=args.output_layer,
            )

    duration = time.time() - start_time
    need_hour, need_mins, need_secs = convert_secs2time(duration)
    print("total runtime: {:02d}:{:02d}:{:02d}".format(need_hour, need_mins, need_secs))
