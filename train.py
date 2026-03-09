from __future__ import print_function
import os
import argparse
import time
import copy
import pickle

import torch
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from testers import test_sparsity_mask
from utils import model_loader, load_layer_config, mask_reverse
from new_TrainValTest import CVTrainValTest
from Server import federated_train

# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
# torch.set_deterministic(True)
np.set_printoptions(threshold=False)

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
    "--use_tfed",
    action="store_true",
    default=False,
    help="use tfed or not",
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

# device setting
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


def train_common_model(args, pipeline, train_common_loader, test_common_loader):
    """
    Train the common model with all tasks
    """
    print("*************** Training Common Model ***************")
    """ Set up no trainables for the lidar, img and gps models"""

    optimizer_init_lr = args.lr
    best_acc = 0
    compined_params = (
        list(model_common.parameters())
        + list(lidar_model.parameters())
        + list(img_model.parameters())
        + list(gps_model.parameters())
    )
    optimizer = torch.optim.Adam(compined_params, optimizer_init_lr, weight_decay=0.001)
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
    else:
        # adjust learning rate
        lr = optimizer_init_lr * (0.1 ** (arg.epochs // 25))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    for epoch in range(args.epochs):
        start = time.time()

        pipeline.train_model(
            args,
            train_common_loader,
            optimizer,
            criterion,
            distribution_loss,
            epoch,
            scheduler,
            WRITER,
            model_common,
            lidar_model,
            lidar_mask,
            img_model,
            img_mask,
            gps_model,
            gps_mask,
        )

        end_train = time.time()
        print("***********val accuracy after each epoch*******")
        prec1 = pipeline.validate_model(
            args,
            lidar_model,
            img_model,
            gps_model,
            model_common,
            test_common_loader,
            [0, 1, 2],
        )
        end_test = time.time()
        WRITER.add_scalar("val/acc_task_lidar_img_gps", prec1, epoch)
        _prec1 = pipeline.validate_model(
            args,
            lidar_model,
            img_model,
            gps_model,
            model_common,
            test_common_loader,
            [0, 1],
        )
        WRITER.add_scalar("val/acc_task_lidar_img", _prec1, epoch)
        _prec1 = pipeline.validate_model(
            args,
            lidar_model,
            img_model,
            gps_model,
            model_common,
            test_common_loader,
            [0, 2],
        )
        WRITER.add_scalar("val/acc_task_lidar_gps", _prec1, epoch)
        _prec1 = pipeline.validate_model(
            args,
            lidar_model,
            img_model,
            gps_model,
            model_common,
            test_common_loader,
            [1, 2],
        )
        WRITER.add_scalar("val/acc_task_img_gps", _prec1, epoch)
        _prec1 = pipeline.validate_model(
            args,
            lidar_model,
            img_model,
            gps_model,
            model_common,
            test_common_loader,
            [0],
        )
        WRITER.add_scalar("val/acc_task_lidar", _prec1, epoch)
        _prec1 = pipeline.validate_model(
            args,
            lidar_model,
            img_model,
            gps_model,
            model_common,
            test_common_loader,
            [1],
        )
        WRITER.add_scalar("val/acc_task_img", _prec1, epoch)
        _prec1 = pipeline.validate_model(
            args,
            lidar_model,
            img_model,
            gps_model,
            model_common,
            test_common_loader,
            [2],
        )
        WRITER.add_scalar("val/acc_task_gps", _prec1, epoch)
        print(
            "Training time: {:.3f}; Testing time: {:.3f}.".format(
                end_train - start, end_test - end_train
            )
        )
        if prec1 > best_acc:  # saving best model
            best_acc = prec1
            save_common_path = os.path.join(args.save_path_exp, "task_common")
            check_and_create(save_common_path)
            torch.save(
                model_common.state_dict(),
                save_common_path + "/best_{}{}.pt".format(args.arch, args.depth),
            )
            lidar_save_path = os.path.join(args.save_path_exp, "task" + str(0))
            img_save_path = os.path.join(args.save_path_exp, "task" + str(1))
            gps_save_path = os.path.join(args.save_path_exp, "task" + str(2))
            check_and_create(lidar_save_path)
            check_and_create(img_save_path)
            check_and_create(gps_save_path)
            torch.save(
                lidar_model.state_dict(),
                lidar_save_path
                + "/best_retrained_{}{}.pt".format(args.arch, args.depth),
            )
            torch.save(
                img_model.state_dict(),
                img_save_path + "/best_retrained_{}{}.pt".format(args.arch, args.depth),
            )
            torch.save(
                gps_model.state_dict(),
                gps_save_path + "/best_retrained_{}{}.pt".format(args.arch, args.depth),
            )

        print("Best Acc after training: {:.4f}%".format(best_acc))
    save_common_path = os.path.join(args.save_path_exp, "task_common")
    check_and_create(save_common_path)
    torch.save(
        model_common.state_dict(),
        save_common_path + "/last_{}{}.pt".format(args.arch, args.depth),
    )
    lidar_save_path = os.path.join(args.save_path_exp, "task" + str(0))
    img_save_path = os.path.join(args.save_path_exp, "task" + str(1))
    gps_save_path = os.path.join(args.save_path_exp, "task" + str(2))
    check_and_create(lidar_save_path)
    check_and_create(img_save_path)
    check_and_create(gps_save_path)
    torch.save(
        lidar_model.state_dict(),
        lidar_save_path + "/last_retrained_{}{}.pt".format(args.arch, args.depth),
    )
    torch.save(
        img_model.state_dict(),
        img_save_path + "/last_retrained_{}{}.pt".format(args.arch, args.depth),
    )
    torch.save(
        gps_model.state_dict(),
        gps_save_path + "/last_retrained_{}{}.pt".format(args.arch, args.depth),
    )


if __name__ == "__main__":
    """
    Parallelly Training model with tasks of data and shared model.
    """
    start_time = time.time()
    # Load data
    base_path = os.path.join(args.base_path, "task" + str(3))
    save_path = os.path.join(args.save_path_exp, "task" + str(3))
    check_and_create(save_path)

    # add common data
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

    # Loading mask for each task from the previous task:
    # LIG -> Load mask L, mask I, mask G is loaded from cummu mask of the previous task
    # load lidar mask
    lidar_save_path = os.path.join(args.save_path_exp, "task" + str(0))
    lidar_mask = pickle.load(open(lidar_save_path + "/mask.pkl", "rb"))
    test_sparsity_mask(args, lidar_mask)

    # load image mask
    img_save_path = os.path.join(args.save_path_exp, "task" + str(1))
    img_mask = pickle.load(open(img_save_path + "/mask.pkl", "rb"))
    test_sparsity_mask(args, img_mask)

    # load gps mask
    prev_cummu_mask = pickle.load(open(img_save_path + "/cumu_mask.pkl", "rb"))
    gps_mask = mask_reverse(args, prev_cummu_mask)

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
    common_save_path = os.path.join(args.save_path_exp, "task_common")
    # model_common.load_state_dict(
    #     torch.load(common_save_path + "/last_{}{}.pt".format(args.arch, args.depth))
    # )
    print("*************** Testing ***************")
    cummu_model.load_state_dict(torch.load(args.load_cummu_model))
    lidar_prec1 = pipeline.test_model(
        args,
        0,
        cummu_model,
        model_common,
        val_common_loader,
        lidar_mask,
    )
    lidar_model = copy.deepcopy(cummu_model)
    cummu_model.load_state_dict(torch.load(args.load_cummu_model))
    img_prec1 = pipeline.test_model(
        args,
        1,
        cummu_model,
        model_common,
        val_common_loader,
        img_mask,
    )
    img_model = copy.deepcopy(cummu_model)
    cummu_model.load_state_dict(torch.load(args.load_cummu_model))
    gps_prec1 = pipeline.test_model(
        args,
        2,
        cummu_model,
        model_common,
        val_common_loader,
        gps_mask,
    )
    gps_model = copy.deepcopy(cummu_model)

    """
    # Train common model
    # """
    model_common = model_loader(
        args, 0, common=True
    )  # if args.adaptive_mask import from masknet otherwise models/flash_net
    model_common.cuda()
    print(model_common)
    # model_common.load_state_dict(
    #     torch.load(common_save_path + "/last_{}{}.pt".format(args.arch, args.depth))
    # )
    # ## Start loading individual retrained models
    # lidar_save_path = os.path.join(args.save_path_exp, "task" + str(0))
    # lidar_model.load_state_dict(
    #     torch.load(
    #         lidar_save_path + "/last_retrained_{}{}.pt".format(args.arch, args.depth)
    #     )
    # )
    # img_save_path = os.path.join(args.save_path_exp, "task" + str(1))
    # img_model.load_state_dict(
    #     torch.load(
    #         img_save_path + "/last_retrained_{}{}.pt".format(args.arch, args.depth)
    #     )
    # )
    # gps_save_path = os.path.join(args.save_path_exp, "task" + str(2))
    # gps_model.load_state_dict(
    #     torch.load(
    #         gps_save_path + "/last_retrained_{}{}.pt".format(args.arch, args.depth)
    #     )
    # )
    if id(cummu_model) == id(model_common):
        raise Exception("cummu_model and model_common are the same object")

    if args.learning_mode == "centralized":
        print(
            "*************** Training Common Model on Centralized mode ***************"
        )
        model_common = train_common_model(
            args, pipeline, train_common_loader, val_common_loader
        )
    elif args.learning_mode == "federated":
        print("*************** Training Common Model on Federated mode ***************")
        model_common = federated_train(
            args,
            model_common,
            lidar_model,
            lidar_mask,
            img_model,
            img_mask,
            gps_model,
            gps_mask,
            WRITER,
        )
    else:
        raise Exception("Learning mode not supported")

    """
    Loading common model and individual retrained models
    """
    # model_common = model_loader(
    #     args, 0, common=True
    # )  # if args.adaptive_mask import from masknet otherwise models/flash_net
    # model_common.cuda()
    # print(model_common)
    # common_save_path = os.path.join(args.save_path_exp, "task_common")
    # model_common.load_state_dict(
    #     torch.load("experiments/LIG_S1/flash/Client_0/model_common.pt")
    # )

    # ## Start loading individual retrained models
    # lidar_save_path = os.path.join(args.save_path_exp, "task" + str(0))
    # lidar_model.load_state_dict(
    #     torch.load("experiments/LIG_S1/flash/Client_0/lidar_model.pt")
    # )
    # img_save_path = os.path.join(args.save_path_exp, "task" + str(1))
    # img_model.load_state_dict(
    #     torch.load("experiments/LIG_S1/flash/Client_0/img_model.pt")
    # )
    # gps_save_path = os.path.join(args.save_path_exp, "task" + str(2))
    # gps_model.load_state_dict(
    #     torch.load("experiments/LIG_S1/flash/Client_0/gps_model.pt")
    # )

    # """
    # Testing the common model
    # """
    # common_prec1 = pipeline.validate_model(
    #     args,
    #     lidar_model,
    #     img_model,
    #     gps_model,
    #     model_common,
    #     test_common_loader,
    #     [0, 1, 2],
    # )

    # _prec1 = pipeline.validate_model(
    #     args,
    #     lidar_model,
    #     img_model,
    #     gps_model,
    #     model_common,
    #     test_common_loader,
    #     [0, 1],
    # )
    # _prec1 = pipeline.validate_model(
    #     args,
    #     lidar_model,
    #     img_model,
    #     gps_model,
    #     model_common,
    #     test_common_loader,
    #     [0, 2],
    # )
    # _prec1 = pipeline.validate_model(
    #     args,
    #     lidar_model,
    #     img_model,
    #     gps_model,
    #     model_common,
    #     test_common_loader,
    #     [1, 2],
    # )
    # _prec1 = pipeline.validate_model(
    #     args,
    #     lidar_model,
    #     img_model,
    #     gps_model,
    #     model_common,
    #     test_common_loader,
    #     [0],
    # )
    # _prec1 = pipeline.validate_model(
    #     args,
    #     lidar_model,
    #     img_model,
    #     gps_model,
    #     model_common,
    #     test_common_loader,
    #     [1],
    # )
    # _prec1 = pipeline.validate_model(
    #     args,
    #     lidar_model,
    #     img_model,
    #     gps_model,
    #     model_common,
    #     test_common_loader,
    #     [2],
    # )
