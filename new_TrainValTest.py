from __future__ import division

import random

random.seed(0)
import pickle
import math
import os
import sys
import numpy as np
import torch

torch.manual_seed(1)
# torch.use_deterministic_algorithms(True)
import time

np.set_printoptions(threshold=sys.maxsize)

from testers import *
from utils import *
import DataGenerator as DG
from torch.utils.data import DataLoader
from control_module import ControlModule
import torchvision.transforms as transforms

np.random.seed(0)
# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True


# torch.set_deterministic(True)
# DATA LOADER FOR SINGLE MODALITY
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def freeze_weights(args, model, mask):
    with torch.no_grad():
        for name, W in model.named_parameters():
            if name in args.fixed_layer:
                W.grad *= 0
                continue
            if name in args.pruned_layer and name in mask:
                # print("name", name)
                # print("mask", mask[name].shape)
                # print("W.grad", W.grad.shape)
                W.grad *= mask[name].cuda()


class data_loader(object):
    def __init__(self, train_val_test, X_lidar_train, X_lidar_test, y_train, y_test):
        if train_val_test == "train":
            self.feat = X_lidar_train
            self.label = y_train
        elif train_val_test == "val":
            self.feat = X_lidar_validation
            self.label = y_validation
        elif train_val_test == "test":
            self.feat = X_lidar_test
            self.label = y_test
        print(train_val_test)

    def __len__(self):
        return self.feat.shape[0]

    def __getitem__(self, index):
        feat = self.feat[index]  #
        label = self.label[index]  # change
        return torch.from_numpy(feat).type(torch.FloatTensor), torch.from_numpy(
            label
        ).type(torch.FloatTensor)


def process_data(X, scale, gps=False):
    if gps:
        X = X.reshape(X.shape[0], X.shape[1], 1, 1)
        X = X / scale
        print("############GPS common scale##########", scale)
    else:
        X = X / scale
    return X


class data_common_loader(object):
    def __init__(
        self,
        train_val_test,
        base_common_path,
    ):
        self.x_gps = np.asarray(
            np.load(os.path.join(base_common_path, f"{train_val_test}/X_gps.npy"))
        )
        self.x_img = np.asarray(
            np.load(os.path.join(base_common_path, f"{train_val_test}/X_img.npy"))
        )
        self.x_lidar = np.asarray(
            np.load(os.path.join(base_common_path, f"{train_val_test}/X_lidar.npy"))
        )
        self.y = np.asarray(
            np.argmax(
                np.load(os.path.join(base_common_path, f"{train_val_test}/y.npy")),
                axis=1,
            )
        )
        self.label = self.y.reshape(self.y.shape[0], 1)
        self.feat_gps = process_data(self.x_gps, scale=1, gps=True)
        self.feat_img = process_data(self.x_img, scale=255)
        self.feat_lidar = process_data(self.x_lidar, scale=1)
        print(
            f"...............{train_val_test} data common shapes............",
            self.feat_gps.shape,
            self.feat_img.shape,
            self.feat_lidar.shape,
            self.label.shape,
        )

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, index):
        feat_gps = self.feat_gps[index]  #
        feat_img = self.feat_img[index]  #
        feat_lidar = self.feat_lidar[index]  #

        # print(f"feat_gps: {feat_gps.shape}, feat_img: {feat_img.shape}, feat_lidar: {feat_lidar.shape}")

        # print('90-feat_gps.shape[0]',feat_gps.shape[0],feat_gps.shape[1],feat_gps.shape[2],90-feat_gps.shape[0],160-feat_gps.shape[1],20-feat_gps.shape[2])
        feat_gps = np.pad(
            feat_gps,
            [
                (0, 45 - feat_gps.shape[0]),
                (0, 80 - feat_gps.shape[1]),
                (0, 20 - feat_gps.shape[2]),
            ],
            mode="constant",
            constant_values=0,
        )
        # self.x_test_gps = np.pad(self.x_test_gps, [(0, 0), (0, 45-self.x_test_gps.shape[0]),(0, 80-self.x_test_gps.shape[1]),(0, 20-self.x_test_gps.shape[2])], mode='constant', constant_values=0)

        # print('90-feat_img.shape[0]',feat_img.shape[0],feat_img.shape[1],feat_img.shape[2],90-feat_img.shape[0],160-feat_img.shape[1],20-feat_img.shape[2])
        feat_img = np.pad(
            feat_img,
            [
                (0, 45 - feat_img.shape[0]),
                (0, 80 - feat_img.shape[1]),
                (0, 20 - feat_img.shape[2]),
            ],
            mode="constant",
            constant_values=0,
        )
        # self.x_test_img = np.pad(self.x_test_img, [(0, 0), (0, 45-self.x_test_img.shape[0]),(0, 80-self.x_test_img.shape[1]),(0, 20-self.x_test_img.shape[2])], mode='constant', constant_values=0)

        # print('90-feat_lidar.shape[0]',feat_lidar.shape[0],feat_lidar.shape[1],feat_lidar.shape[2],90-feat_lidar.shape[0],160-feat_lidar.shape[1],20-feat_lidar.shape[2])
        feat_lidar = np.pad(
            feat_lidar,
            [
                (0, 45 - feat_lidar.shape[0]),
                (0, 80 - feat_lidar.shape[1]),
                (0, 20 - feat_lidar.shape[2]),
            ],
            mode="constant",
            constant_values=0,
        )
        # self.x_test_lidar = np.pad(self.x_test_lidar, [(0, 0), (0, 45-self.x_test_lidar.shape[0]),(0, 80-self.x_test_lidar.shape[1]),(0, 20-self.x_test_lidar.shape[2])], mode='constant', constant_values=0)

        feat_common = np.stack((feat_img, feat_lidar, feat_gps), axis=0)
        label = self.label[index]  # change
        return torch.from_numpy(feat_common).type(torch.FloatTensor), torch.from_numpy(
            label
        ).type(torch.FloatTensor)


class CVTrainValTest:
    def __init__(self, base_path, base_common_path):

        self.base_path = base_path
        self.base_common_path = base_common_path

    def load_data_cifar(self, batch_size):
        self.x_train = np.asarray(np.load(os.path.join(self.base_path, "train/X.npy")))
        self.y_train = np.asarray(np.load(os.path.join(self.base_path, "train/y.npy")))
        self.x_test = np.asarray(np.load(os.path.join(self.base_path, "test/X.npy")))
        self.y_test = np.asarray(np.load(os.path.join(self.base_path, "test/y.npy")))

        # map label to 0-9
        max_label = np.max(self.y_train)
        if max_label > 9:
            self.y_train = self.y_train - (max_label - 9)
            self.y_test = self.y_test - (max_label - 9)
        print(
            "# of training exp:%d, testing exp:%d"
            % (len(self.x_train), len(self.x_test))
        )

        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        self.training_set = DG.CifarDataGenerator(self.x_train, self.y_train)
        DataParams = {"batch_size": batch_size, "shuffle": True, "num_workers": 0}
        self.train_generator = DataLoader(self.training_set, **DataParams)

        self.test_set = DG.CifarDataGenerator(self.x_test, self.y_test)
        DataParams = {"batch_size": batch_size, "shuffle": False, "num_workers": 0}
        self.test_generator = DataLoader(self.test_set, **DataParams)

        return self.train_generator

    def load_data_mnist(self, batch_size):
        self.x_train = np.asarray(np.load(os.path.join(self.base_path, "train/X.npy")))
        self.y_train = np.asarray(np.load(os.path.join(self.base_path, "train/y.npy")))
        self.x_test = np.asarray(np.load(os.path.join(self.base_path, "test/X.npy")))
        self.y_test = np.asarray(np.load(os.path.join(self.base_path, "test/y.npy")))

        # map label to 0-9
        max_label = np.max(self.y_train)
        if max_label > 9:
            self.y_train = self.y_train - (max_label - 9)
            self.y_test = self.y_test - (max_label - 9)
        print(
            "# of training exp:%d, testing exp:%d"
            % (len(self.x_train), len(self.x_test))
        )

        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        self.training_set = DG.MnistDataGenerator(self.x_train, self.y_train)
        DataParams = {"batch_size": batch_size, "shuffle": True, "num_workers": 0}
        self.train_generator = DataLoader(self.training_set, **DataParams)

        self.test_set = DG.MnistDataGenerator(self.x_test, self.y_test)
        DataParams = {"batch_size": batch_size, "shuffle": False, "num_workers": 0}
        self.test_generator = DataLoader(self.test_set, **DataParams)

        return self.train_generator

    def load_data_mixture(self, params):
        """
        Mixture dataset contains 5 tasks, [mnist,cifar,mnist,cifar,mnist]
        Mnist > Cifar => subsample mnist
        # Mnist: 60000
        # Cifar: 5000
        """
        self.x_train = np.asarray(np.load(os.path.join(self.base_path, "train/X.npy")))
        self.y_train = np.asarray(np.load(os.path.join(self.base_path, "train/y.npy")))
        self.x_test = np.asarray(np.load(os.path.join(self.base_path, "test/X.npy")))
        self.y_test = np.asarray(np.load(os.path.join(self.base_path, "test/y.npy")))

        # map label to 0-9
        max_label = np.max(self.y_train)
        if max_label > 9:
            self.y_train = self.y_train - (max_label - 9)
            self.y_test = self.y_test - (max_label - 9)
        print(
            "# of training exp:%d, testing exp:%d"
            % (len(self.x_train), len(self.x_test))
        )

        # scale number of training sample
        scale = 1
        trigger = False
        if len(self.y_train) > 5000:
            trigger = True
            params.epochs = 50
            params.epochs_prune = 30
            params.epochs_mask_retrain = 50
            print(
                "Sample {} examples in each training epoch.".format(
                    int(len(self.y_train) * scale)
                )
            )
        else:
            params.epochs = 300
            params.epochs_prune = 200
            params.epochs_mask_retrain = 300

        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        self.training_set = DG.MixtureDataGenerator(
            self.x_train, self.y_train, scale=scale, trigger=trigger
        )
        DataParams = {
            "batch_size": params.batch_size,
            "shuffle": True,
            "num_workers": 0,
        }
        self.train_generator = DataLoader(self.training_set, **DataParams)

        self.test_set = DG.MixtureDataGenerator(
            self.x_test, self.y_test, trigger=trigger
        )
        DataParams = {
            "batch_size": params.batch_size,
            "shuffle": False,
            "num_workers": 0,
        }
        self.test_generator = DataLoader(self.test_set, **DataParams)
        return params, self.train_generator

    # TODO: Implement load_data_common_flash
    def load_data_common_flash(self, batch_size):

        self.common_training_set = data_common_loader(
            "train",
            self.base_common_path,
        )
        self.common_val_set = data_common_loader(
            "val",
            self.base_common_path,
        )
        self.common_test_set = data_common_loader(
            "test",
            self.base_common_path,
        )

        self.common_train_generator = DataLoader(
            self.common_training_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            worker_init_fn=seed_worker,
        )
        self.common_val_generator = DataLoader(
            self.common_val_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            worker_init_fn=seed_worker,
        )
        self.common_test_generator = DataLoader(
            self.common_test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            worker_init_fn=seed_worker,
        )
        print("self.common_train_generator", self.common_train_generator)
        print("self.common_test_generator", self.common_test_generator)

        return (
            self.common_train_generator,
            self.common_val_generator,
            self.common_test_generator,
        )

    def train_model(
        self,
        args,
        train_common_loader,
        optimizer,
        criterion,
        distribution_loss,
        epoch,
        scheduler,
        WRITER,
        model_common,
        model0=None,
        mask0=None,
        model1=None,
        mask1=None,
        model2=None,
        mask2=None,
        mode_split=None,
    ):
        # print('train func params',args, model, masks)
        losses = AverageMeter()
        top1 = AverageMeter()
        idx_loss_dict = {}

        # TODO: Implement the training loop for common model
        model_common.train()
        model0.train()
        if model1 is not None:
            model1.train()
        if model2 is not None:
            model2.train()
        for i, (input_common, target) in enumerate(train_common_loader):
            # TODO: Set mode
            scheduler.step()
            if mode_split is None:
                mode_num = random.randint(1, 3)
                _mode_split = random.sample([0, 1, 2], mode_num)
            else:
                mode_num = random.randint(1, len(mode_split))
                _mode_split = random.sample(mode_split, mode_num)
                # _mode_split = mode_split
            # print("mode_split", mode_split)
            specific_features = (
                torch.zeros((input_common.shape[0], 3, 64)).float().cuda()
            )
            _input_common = input_common.clone().float().cuda()
            for m in [0, 1, 2]:
                if m not in _mode_split:
                    _input_common[:, m, :, :, :] = 0
                    specific_features[:, m, :] = 0
                elif m == 0:
                    d0 = _input_common[:, 0, :, :, :]
                    specific_features[:, m, :] = model0(d0)
                elif m == 1:
                    d1 = _input_common[:, 1, :, :, :]
                    specific_features[:, m, :] = model1(d1)
                elif m == 2:
                    d2 = _input_common[:, 2, :, :, :]
                    specific_features[:, m, :] = model2(d2)

            input = _input_common
            target = target.long().cuda()
            # start_reshape = time.time()
            input = input.reshape(-1, input.shape[2], input.shape[3], input.shape[4])
            specific_features = specific_features.reshape(
                -1,
                specific_features.shape[2],
            )
            model_common.set_mode_split(_mode_split)
            shared_features, final_output = model_common(input, specific_features)

            # TODO: Compute more loss functions, including cross entropy loss, similarity loss, and sparsity loss
            z1 = shared_features[0 : len(shared_features) + 1 : 3]
            z2 = shared_features[1 : len(shared_features) + 1 : 3]
            z3 = shared_features[2 : len(shared_features) + 1 : 3]

            shared_loss = (
                distribution_loss(z1, z2)
                + distribution_loss(z1, z3)
                + distribution_loss(z2, z3)
            )
            ce_loss = criterion(final_output, target[:, 0])

            loss_1 = shared_loss.item()
            loss_2 = ce_loss.item()

            ce_loss += 0.2 * shared_loss

            # measure accuracy and record loss
            prec1, _ = accuracy(final_output, target, topk=(1, 5))
            losses.update(ce_loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))

            WRITER.add_scalar(
                "Loss/distribution_loss",
                loss_1,
                epoch * len(train_common_loader) + i,
            )
            WRITER.add_scalar(
                "Loss/cross_entropy_loss",
                loss_2,
                epoch * len(train_common_loader) + i,
            )
            WRITER.add_scalar(
                "Loss/total_loss",
                ce_loss.item(),
                epoch * len(train_common_loader) + i,
            )
            WRITER.add_scalar(
                "Accuracy/train_task_lidar_img_gps",
                prec1[0],
                epoch * len(train_common_loader) + i,
            )
            # compute gradient and do SGD step
            optimizer.zero_grad()
            ce_loss.backward()

            if (mask0 is not None) and (0 in _mode_split):
                freeze_weights(args, model0, mask0)
            if (mask1 is not None) and (1 in _mode_split):
                freeze_weights(args, model1, mask1)
            if (mask2 is not None) and (2 in _mode_split):
                freeze_weights(args, model2, mask2)
            optimizer.step()
            # print("test 2",model.conv1.weight.grad[0])

            # control = ControlModule(model, config=config)
            # for (key, param), g in zip(model.named_parameters(), list_grad):
            #     assert param.size() == g.size()
            #     control.accumulate(key, g ** 2)

            # print(i)
            if i % 100 == 0:
                for param_group in optimizer.param_groups:
                    current_lr = param_group["lr"]
                print(
                    "({0}) lr:[{1:.5f}]  "
                    "Epoch: [{2}][{3}/{4}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Acc@1 {top1.val:.3f}% ({top1.avg:.3f}%)\t".format(
                        "adam",
                        current_lr,
                        epoch,
                        i,
                        len(train_common_loader),
                        loss=losses,
                        top1=top1,
                    )
                )
            if i % 100 == 0:
                idx_loss_dict[i] = losses.avg
        model_dict = {
            "model0": model0,
            "model1": model1,
            "model2": model2,
            "model_common": model_common,
        }

        return model_dict

    def validate_model(
        self,
        args,
        model0,
        model1,
        model2,
        model_common,
        test_common_loader,
        mode_split,
    ):
        """
        Run evaluation
        """
        batch_time = AverageMeter()
        top1 = AverageMeter()

        # switch to evaluate mode
        model_common.eval()
        model0.eval()
        if model1 is not None:
            model1.eval()
        if model2 is not None:
            model2.eval()

        end = time.time()

        # print('Afterself.test_generator',self.test_generator)
        # for i, (input, target) in enumerate(self.test_generator):
        for i, (input_common, target) in enumerate(test_common_loader):
            # TODO: Set mode
            # mode_num = random.randint(1, 3)
            # mode_split = random.sample([0, 1, 2], 2)
            specific_features = (
                torch.zeros((input_common.shape[0], 3, 64)).float().cuda()
            )
            _input_common = input_common.clone().float().cuda()
            for m in [0, 1, 2]:
                if m not in mode_split:
                    _input_common[:, m, :, :, :] = 0
                    specific_features[:, m, :] = 0
                elif m == 0:
                    d0 = _input_common[:, 0, :, :, :]
                    specific_features[:, m, :] = model0(d0)
                elif m == 1:
                    d1 = _input_common[:, 1, :, :, :]
                    specific_features[:, m, :] = model1(d1)
                elif m == 2:
                    d2 = _input_common[:, 2, :, :, :]
                    specific_features[:, m, :] = model2(d2)

            input = _input_common
            target = target.long().cuda()
            # start_reshape = time.time()
            input = input.reshape(-1, input.shape[2], input.shape[3], input.shape[4])
            specific_features = specific_features.reshape(
                -1,
                specific_features.shape[2],
            )
            # reshaped_time = time.time() - start_reshape
            model_common.set_mode_split(mode_split)
            shared_features, final_output = model_common(input, specific_features)

            # measure accuracy and record loss
            prec1, _ = accuracy(final_output, target, topk=(1, 5))
            top1.update(prec1[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        if len(mode_split) == 1 and 0 in mode_split:
            print(
                "Lidar Average time per batch {batch_time.avg:.5f}".format(
                    batch_time=batch_time
                )
            )
            print("Lidar Testing Prec@1 {top1.avg:.3f}%".format(top1=top1))
        elif len(mode_split) == 1 and 1 in mode_split:
            print(
                "Image Average time per batch {batch_time.avg:.5f}".format(
                    batch_time=batch_time
                )
            )
            print("Image Testing Prec@1 {top1.avg:.3f}%".format(top1=top1))
        elif len(mode_split) == 1 and 2 in mode_split:
            print(
                "GPS Average time per batch {batch_time.avg:.5f}".format(
                    batch_time=batch_time
                )
            )
            print("GPS Testing Prec@1 {top1.avg:.3f}%".format(top1=top1))
        elif len(mode_split) == 2 and 0 in mode_split and 1 in mode_split:
            print(
                "Lidar Image Average time per batch {batch_time.avg:.5f}".format(
                    batch_time=batch_time
                )
            )
            print("Lidar Image Testing Prec@1 {top1.avg:.3f}%".format(top1=top1))
        elif len(mode_split) == 2 and 0 in mode_split and 2 in mode_split:
            print(
                "Lidar GPS Average time per batch {batch_time.avg:.5f}".format(
                    batch_time=batch_time
                )
            )
            print("Lidar GPS Testing Prec@1 {top1.avg:.3f}%".format(top1=top1))
        elif len(mode_split) == 2 and 1 in mode_split and 2 in mode_split:
            print(
                "Image GPS Average time per batch {batch_time.avg:.5f}".format(
                    batch_time=batch_time
                )
            )
            print("Image GPS Testing Prec@1 {top1.avg:.3f}%".format(top1=top1))
        elif len(mode_split) == 3:
            print(
                "Lidar Image GPS Average time per batch {batch_time.avg:.5f}".format(
                    batch_time=batch_time
                )
            )
            print("Lidar Image GPS Testing Prec@1 {top1.avg:.3f}%".format(top1=top1))

        return top1.avg

    def test_model(self, args, task, model, model_common, test_common_loader, mask=""):
        """
        Run evaluation
        """
        batch_time = AverageMeter()
        top1 = AverageMeter()

        if mask:
            set_model_mask(model, mask)
        # switch to evaluate mode
        model.eval()
        # model_common.eval()
        # model_common.set_task(1, task)
        # print("model_common.task", model_common.task)

        end = time.time()
        # print('Afterself.test_generator',self.test_generator)
        # for i, (input, target) in enumerate(self.test_generator):
        for i, (input_common, target) in enumerate(test_common_loader):
            input = input_common[:, task, :, :, :].float().cuda()
            # input_common = input_common.float().cuda()
            target = target.long().cuda()

            # compute output
            output = model(input)
            # shared_features, output = model_common(input, spec_features)

            output = output.float()

            # TODO: Compute the output for the common model
            # shared_features, final_output = model_common(input, output)

            # measure accuracy and record loss
            prec1, _ = accuracy(output, target, topk=(1, 5))
            top1.update(prec1[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print(
            "Average time per batch {batch_time.avg:.5f}".format(batch_time=batch_time)
        )
        print("Testing Prec@1 {top1.avg:.3f}%".format(top1=top1))

        return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
