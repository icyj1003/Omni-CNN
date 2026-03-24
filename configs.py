import argparse
from utils import *


def default_configs():
    parser = argparse.ArgumentParser()
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
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.1)",
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
        "--masked-retrain",
        action="store_true",
        default=False,
        help="for masked retrain",
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
    parser.add_argument(
        "--smooth", action="store_true", default=False, help="lable smooth"
    )
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
    parser.add_argument(
        "--base_path", default="", type=str, help="Specify the data path"
    )
    parser.add_argument(
        "--save_path", default="", type=str, help="Specify the save path"
    )
    parser.add_argument(
        "--input_size", default=32, type=int, help="Specify the input size"
    )
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
        "--adaptive-mask",
        default=False,
        type=str2bool,
        help="adaptive learning the mask",
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
        "--load-model",
        type=str,
        default="",
        help="For loading exist pure trained Model",
    )

    parser.add_argument(
        "--load-common-model",
        type=str,
        default="",
        help="For loading exist pure trained Model",
    )

    parser.add_argument(
        "--load-model-pruned",
        type=str,
        default="",
        help="For loading exist pruned Model",
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
