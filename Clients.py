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
                W.grad *= mask[name].cuda()


def convert(equipment):
    equipment_list = ["lidar", "img", "gps"]
    mode_list = [0, 1, 2]
    equipment_to_mode = dict(zip(equipment_list, mode_list))
    mode_split = [equipment_to_mode[eq] for eq in equipment]
    return mode_split


def check_and_create(dir_path):
    if os.path.exists(dir_path):
        return True
    else:
        os.makedirs(dir_path)
        return False


def show_results(mode_split, top1, batch_time, val_test):
    if len(mode_split) == 1 and 0 in mode_split:
        print(
            "Lidar Average time per batch {batch_time.avg:.5f}".format(
                batch_time=batch_time
            )
        )
        print(val_test + "Lidar Prec@1 {top1.avg:.3f}%".format(top1=top1))
    elif len(mode_split) == 1 and 1 in mode_split:
        print(
            "Image Average time per batch {batch_time.avg:.5f}".format(
                batch_time=batch_time
            )
        )
        print(val_test + "Image Prec@1 {top1.avg:.3f}%".format(top1=top1))
    elif len(mode_split) == 1 and 2 in mode_split:
        print(
            "GPS Average time per batch {batch_time.avg:.5f}".format(
                batch_time=batch_time
            )
        )
        print(val_test + "GPS Prec@1 {top1.avg:.3f}%".format(top1=top1))
    elif len(mode_split) == 2 and 0 in mode_split and 1 in mode_split:
        print(
            "Lidar Image Average time per batch {batch_time.avg:.5f}".format(
                batch_time=batch_time
            )
        )
        print(val_test + "Lidar Image Prec@1 {top1.avg:.3f}%".format(top1=top1))
    elif len(mode_split) == 2 and 0 in mode_split and 2 in mode_split:
        print(
            "Lidar GPS Average time per batch {batch_time.avg:.5f}".format(
                batch_time=batch_time
            )
        )
        print(val_test + "Lidar GPS Prec@1 {top1.avg:.3f}%".format(top1=top1))
    elif len(mode_split) == 2 and 1 in mode_split and 2 in mode_split:
        print(
            "Image GPS Average time per batch {batch_time.avg:.5f}".format(
                batch_time=batch_time
            )
        )
        print(val_test + "Image GPS Prec@1 {top1.avg:.3f}%".format(top1=top1))
    elif len(mode_split) == 3:
        print(
            "Lidar Image GPS Average time per batch {batch_time.avg:.5f}".format(
                batch_time=batch_time
            )
        )
        print(val_test + "Lidar Image GPS Prec@1 {top1.avg:.3f}%".format(top1=top1))


def process_data(X, scale, gps=False):
    if gps:
        X = X.reshape(X.shape[0], X.shape[1], 1, 1)
        X = X / scale
        print("############GPS common scale##########", scale)
    else:
        X = X / scale
    return X


class Client_data_loader(object):
    def __init__(self, train_val_test, equipment, client_data_path):
        self.train_val_test = train_val_test
        self.y = np.asarray(
            np.argmax(
                np.load(os.path.join(client_data_path, f"{train_val_test}/y.npy")),
                axis=1,
            )
        )
        self.label = self.y.reshape(self.y.shape[0], 1)
        if "gps" in equipment:
            self.x_gps = np.asarray(
                np.load(os.path.join(client_data_path, f"{train_val_test}/X_gps.npy"))
            )
            self.feat_gps = process_data(self.x_gps, scale=1, gps=True)
        else:
            self.feat_gps = np.zeros((self.y.shape[0], 2, 1, 1))
        if "img" in equipment:
            self.x_img = np.asarray(
                np.load(os.path.join(client_data_path, f"{train_val_test}/X_img.npy"))
            )
            self.feat_img = process_data(self.x_img, scale=255)
        else:
            self.feat_img = np.zeros((self.y.shape[0], 45, 80, 3))
        if "lidar" in equipment:
            self.x_lidar = np.asarray(
                np.load(os.path.join(client_data_path, f"{train_val_test}/X_lidar.npy"))
            )
            self.feat_lidar = process_data(self.x_lidar, scale=1)
        else:
            self.feat_lidar = np.zeros((self.y.shape[0], 20, 20, 20))
        print(
            f"...............Loading {train_val_test} data from client path: {client_data_path}............",
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

        feat_common = np.stack((feat_lidar, feat_img, feat_gps), axis=0)
        label = self.label[index]  # change
        return torch.from_numpy(feat_common).type(torch.FloatTensor), torch.from_numpy(
            label
        ).type(torch.FloatTensor)


class Client_pipeline:
    def __init__(self, args, client_data_path, client_save_path, client_id, equipment):
        self.args = args
        self.client_id = client_id
        self.equipment = equipment
        self.mode_split = convert(equipment)
        self.client_data_path = client_data_path
        self.client_save_path = client_save_path
        self.__transfer_layer = ["out.weight", "out.bias"]
        self.__previous_accuracy = 0
        self.__delta_acc = 0

    def update_previous_accuracy(self, acc):
        self.__previous_accuracy = acc

    def update_delta_acc(self, delta_acc):
        self.__delta_acc = delta_acc
        # self.__previous_accuracy = acc / 100

    def get_delta_acc(self):
        # print("Delta Accuracy", self.__delta_acc)
        return self.__delta_acc

    def load_data(self):
        self.client_train_data = Client_data_loader(
            "train", self.equipment, self.client_data_path
        )
        self.client_val_data = Client_data_loader(
            "val", self.equipment, self.client_data_path
        )
        self.client_test_data = Client_data_loader(
            "test", self.equipment, self.client_data_path
        )
        self.train_size = len(self.client_train_data)
        self.__train_loader = DataLoader(
            self.client_train_data,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=8,
            worker_init_fn=seed_worker,
        )
        self.__val_loader = DataLoader(
            self.client_val_data,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=8,
            worker_init_fn=seed_worker,
        )
        self.__test_loader = DataLoader(
            self.client_test_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=8,
            worker_init_fn=seed_worker,
        )
        print(self.train_size)
        return self.train_size

    def get_train_size(self):
        return self.train_size

    def set_transfer_learning(self):
        for name, W in self.model_common.named_parameters():
            if name not in self.__transfer_layer:
                W.requires_grad = False
        if "lidar" in self.equipment:
            for name, W in self.lidar_model.named_parameters():
                W.requires_grad = False
        if "img" in self.equipment:
            for name, W in self.img_model.named_parameters():
                W.requires_grad = False
        if "gps" in self.equipment:
            for name, W in self.gps_model.named_parameters():
                W.requires_grad = False
        self.__transfer = True

    def set_relearning(self):
        for name, W in self.model_common.named_parameters():
            W.requires_grad = True
        if "lidar" in self.equipment:
            for name, W in self.lidar_model.named_parameters():
                W.requires_grad = True
        if "img" in self.equipment:
            for name, W in self.img_model.named_parameters():
                W.requires_grad = True
        if "gps" in self.equipment:
            for name, W in self.gps_model.named_parameters():
                W.requires_grad = True
        self.__transfer = False

    def load_model(
        self,
        model_common,
        lidar_model=None,
        lidar_mask=None,
        img_model=None,
        img_mask=None,
        gps_model=None,
        gps_mask=None,
    ):
        if "lidar" in self.equipment:
            self.lidar_model = lidar_model
            self.__lidar_mask = lidar_mask
            self.__previous_lidar_mask = lidar_mask
            self.__transfer_lidar_mask = None
        else:
            self.lidar_model = None
            self.__lidar_mask = None
            self.__transfer_lidar_mask = None
        if "img" in self.equipment:
            self.img_model = img_model
            self.__img_mask = img_mask
            self.__previous_img_mask = img_mask
            self.__transfer_img_mask = None
        else:
            self.img_model = None
            self.__img_mask = None
            self.__transfer_img_mask = None
        if "gps" in self.equipment:
            self.gps_model = gps_model
            self.__gps_mask = gps_mask
            self.__previous_gps_mask = gps_mask
            self.__transfer_gps_mask = None
        else:
            self.gps_model = None
            self.__gps_mask = None
            self.__transfer_gps_mask = None
        self.model_common = model_common
        self.__model_common_mask = dict(
            [
                (name, torch.ones_like(param))
                for name, param in model_common.named_parameters()
            ]
        )
        self.__previous_common_mask = self.__model_common_mask
        self.__transfer_model_common_mask = dict(
            [
                (
                    (name, torch.zeros_like(param))
                    if name not in self.__transfer_layer
                    else (name, torch.ones_like(param))
                )
                for name, param in model_common.named_parameters()
            ]
        )

    def save_model(self):
        temp_path = os.path.join(
            "experiments/federated_learning",
            "Client_" + str(self.client_id),
        )
        check_and_create(temp_path)
        if "lidar" in self.equipment:
            torch.save(
                self.lidar_model.state_dict(),
                os.path.join(temp_path, "lidar_model.pth"),
            )
        if "img" in self.equipment:
            torch.save(
                self.img_model.state_dict(),
                os.path.join(temp_path, "img_model.pth"),
            )
        if "gps" in self.equipment:
            torch.save(
                self.gps_model.state_dict(),
                os.path.join(temp_path, "gps_model.pth"),
            )
        torch.save(
            self.model_common.state_dict(),
            os.path.join(temp_path, "model_common.pth"),
        )

    def get_mask(self, mode):
        if mode == "lidar":
            if self.__transfer:
                self.__previous_lidar_mask = self.__transfer_lidar_mask
                return self.__transfer_lidar_mask
            self.__previous_lidar_mask = self.__lidar_mask
            return self.__lidar_mask
        elif mode == "img":
            if self.__transfer:
                self.__previous_img_mask = self.__transfer_img_mask
                return self.__transfer_img_mask
            self.__previous_img_mask = self.__img_mask
            return self.__img_mask
        elif mode == "gps":
            if self.__transfer:
                self.__previous_gps_mask = self.__transfer_gps_mask
                return self.__transfer_gps_mask
            self.__previous_gps_mask = self.__gps_mask
            return self.__gps_mask
        elif mode == "common":
            if self.__transfer:
                self.__previous_common_mask = self.__transfer_model_common_mask
                return self.__transfer_model_common_mask
            self.__previous_common_mask = self.__model_common_mask
            return self.__model_common_mask

    def update_model(self, model_common_params, lidar_params, img_params, gps_params):
        temp_path = os.path.join(
            "experiments/federated_learning",
            "Client_" + str(self.client_id),
        )
        try:
            self.model_common.load_state_dict(
                torch.load(
                    os.path.join(temp_path, "model_common.pth"), weights_only=True
                )
            )
            for name, W in self.model_common.named_parameters():
                with torch.no_grad():
                    W[self.__previous_common_mask[name].type(torch.bool)] = (
                        model_common_params[name][
                            self.__previous_common_mask[name].type(torch.bool)
                        ]
                    )
        except:
            print("Model common not found")
            self.model_common.load_state_dict(model_common_params)

        if "lidar" in self.equipment:
            try:
                self.lidar_model.load_state_dict(
                    torch.load(
                        os.path.join(temp_path, "lidar_model.pth"), weights_only=True
                    )
                )
                if self.__previous_lidar_mask is not None:
                    for name, W in self.lidar_model.named_parameters():
                        with torch.no_grad():
                            W[self.__previous_lidar_mask[name].type(torch.bool)] = (
                                lidar_params[name][
                                    self.__previous_lidar_mask[name].type(torch.bool)
                                ]
                            )
            except:
                print("Lidar model not found")
                self.lidar_model.load_state_dict(lidar_params)
        if "img" in self.equipment:
            try:
                self.img_model.load_state_dict(
                    torch.load(
                        os.path.join(temp_path, "img_model.pth"), weights_only=True
                    )
                )
                if self.__previous_img_mask is not None:
                    for name, W in self.img_model.named_parameters():
                        with torch.no_grad():
                            W[self.__previous_img_mask[name].type(torch.bool)] = (
                                img_params[name][
                                    self.__previous_img_mask[name].type(torch.bool)
                                ]
                            )
            except:
                print("Img model not found")
                self.img_model.load_state_dict(img_params)
        if "gps" in self.equipment:
            try:
                self.gps_model.load_state_dict(
                    torch.load(
                        os.path.join(temp_path, "gps_model.pth"), weights_only=True
                    )
                )
                if self.__previous_gps_mask is not None:
                    for name, W in self.gps_model.named_parameters():
                        with torch.no_grad():
                            W[self.__previous_gps_mask[name].type(torch.bool)] = (
                                gps_params[name][
                                    self.__previous_gps_mask[name].type(torch.bool)
                                ]
                            )
            except:
                print("GPS model not found")
                self.gps_model.load_state_dict(gps_params)

    def configure_optimizer(self):
        compined_params = list(self.model_common.parameters())
        if "lidar" in self.equipment:
            compined_params += list(self.lidar_model.parameters())
        if "img" in self.equipment:
            compined_params += list(self.img_model.parameters())
        if "gps" in self.equipment:
            compined_params += list(self.gps_model.parameters())
        self.__optimizer = torch.optim.Adam(
            compined_params, self.args.lr, weight_decay=0.0003
        )
        self.__criterion = torch.nn.CrossEntropyLoss()
        self.__distribution_loss = torch.nn.L1Loss()
        epoch_milestones = [50, 75, 95, 115, 140, 165, 190, 220, 250, 280]
        self.__scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.__optimizer,
            milestones=[i * len(self.__train_loader) for i in epoch_milestones],
            gamma=0.5,
        )
        """
        Set learning rate
        """
        self.__scheduler = None
        if self.args.lr_scheduler == "cosine":
            self.__scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.__optimizer,
                T_max=self.args.epochs * len(self.__train_loader),
                eta_min=4e-08,
            )
        elif self.args.lr_scheduler == "default":
            # my learning rate self.__scheduler for cifar, following https://github.com/kuangliu/pytorch-cifar
            # epoch_milestones = [65, 100, 130, 190, 220, 250, 280]
            epoch_milestones = [50, 75, 95, 115, 140, 165, 190, 220, 250, 280]

            """
            Set the learning rate of each parameter task to the initial lr decayed
            by gamma once the number of epoch reaches one of the milestones
            """
            self.__scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.__optimizer,
                milestones=[i * len(self.__train_loader) for i in epoch_milestones],
                gamma=0.5,
            )
        else:
            # adjust learning rate
            lr = self.args.lr * (0.1 ** (self.args.epochs // 25))
            for param_group in self.__optimizer.param_groups:
                param_group["lr"] = lr

    def client_local_training(self, epochs, WRITER=None):
        self.model_common.train()
        if "lidar" in self.equipment:
            self.lidar_model.train()
        if "img" in self.equipment:
            self.img_model.train()
        if "gps" in self.equipment:
            self.gps_model.train()

        for epoch in range(epochs):
            print(
                f".................Client {self.client_id} training epoch {epoch}.................."
            )
            # print('train func params',self.args, model, masks)
            losses = AverageMeter()
            top1 = AverageMeter()
            idx_loss_dict = {}
            for i, (data, label) in enumerate(self.__train_loader):
                self.__scheduler.step()
                specific_features = torch.zeros((data.shape[0], 3, 64)).float().cuda()
                _data = data.clone().float().cuda()
                for m in [0, 1, 2]:
                    if m not in self.mode_split:
                        _data[:, m, :, :, :] = 0
                        specific_features[:, m, :] = 0
                    elif m == 0:
                        lidar = _data[:, 0, :, :, :]
                        specific_features[:, m, :] = self.lidar_model(lidar)
                    elif m == 1:
                        img = _data[:, 1, :, :, :]
                        specific_features[:, m, :] = self.img_model(img)
                    elif m == 2:
                        gps = _data[:, 2, :, :, :]
                        specific_features[:, m, :] = self.gps_model(gps)

                input = _data
                target = label.long().cuda()
                input = input.reshape(
                    -1, input.shape[2], input.shape[3], input.shape[4]
                )
                specific_features = specific_features.reshape(
                    -1,
                    specific_features.shape[2],
                )
                self.model_common.set_mode_split(self.mode_split)
                shared_features, final_output = self.model_common(
                    input, specific_features
                )

                # TODO: Compute more loss functions, including cross entropy loss, similarity loss, and sparsity loss
                shared_loss = (
                    self.__distribution_loss(
                        shared_features[0 : len(shared_features) + 1 : 3],
                        shared_features[1 : len(shared_features) + 1 : 3],
                    )
                    + self.__distribution_loss(
                        shared_features[0 : len(shared_features) + 1 : 3],
                        shared_features[2 : len(shared_features) + 1 : 3],
                    )
                    + self.__distribution_loss(
                        shared_features[1 : len(shared_features) + 1 : 3],
                        shared_features[2 : len(shared_features) + 1 : 3],
                    )
                )
                ce_loss = (
                    self.__criterion(final_output, target[:, 0]) + 0.2 * shared_loss
                )

                # measure accuracy and record loss
                prec1, _ = accuracy(final_output, target, topk=(1, 5))
                losses.update(ce_loss.item(), input.size(0))
                top1.update(prec1[0], input.size(0))
                self.__optimizer.zero_grad()
                ce_loss.backward()
                if (self.__lidar_mask is not None) and (not self.__transfer):
                    freeze_weights(self.args, self.lidar_model, self.__lidar_mask)
                if (self.__img_mask is not None) and (not self.__transfer):
                    freeze_weights(self.args, self.img_model, self.__img_mask)
                if (self.__gps_mask is not None) and (not self.__transfer):
                    freeze_weights(self.args, self.gps_model, self.__gps_mask)
                self.__optimizer.step()
                if i % 100 == 0:
                    for param_group in self.__optimizer.param_groups:
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
                            len(self.__train_loader),
                            loss=losses,
                            top1=top1,
                        )
                    )
                if i % 100 == 0:
                    idx_loss_dict[i] = losses.avg

            # Validation
            val_batch_time = AverageMeter()
            val_top1 = AverageMeter()

            # switch to evaluate mode
            self.model_common.eval()
            if "lidar" in self.equipment:
                self.lidar_model.eval()
            if "img" in self.equipment:
                self.img_model.eval()
            if "gps" in self.equipment:
                self.gps_model.eval()
            end = time.time()
            for i, (data, label) in enumerate(self.__val_loader):
                specific_features = torch.zeros((data.shape[0], 3, 64)).float().cuda()
                _data = data.clone().float().cuda()
                for m in [0, 1, 2]:
                    if m not in self.mode_split:
                        _data[:, m, :, :, :] = 0
                        specific_features[:, m, :] = 0
                    elif m == 0:
                        lidar = _data[:, 0, :, :, :]
                        specific_features[:, m, :] = self.lidar_model(lidar)
                    elif m == 1:
                        img = _data[:, 1, :, :, :]
                        specific_features[:, m, :] = self.img_model(img)
                    elif m == 2:
                        gps = _data[:, 2, :, :, :]
                        specific_features[:, m, :] = self.gps_model(gps)

                input = _data
                target = label.long().cuda()
                input = input.reshape(
                    -1, input.shape[2], input.shape[3], input.shape[4]
                )
                specific_features = specific_features.reshape(
                    -1,
                    specific_features.shape[2],
                )
                self.model_common.set_mode_split(self.mode_split)
                shared_features, final_output = self.model_common(
                    input, specific_features
                )
                # measure accuracy and record loss
                prec1, _ = accuracy(final_output, target, topk=(1, 5))
                val_top1.update(prec1[0], input.size(0))
                # measure elapsed time
                val_batch_time.update(time.time() - end)
                end = time.time()
            show_results(self.mode_split, val_top1, val_batch_time, "Validation ")
        self.save_model()

    def model_testing_on_local_data(self):
        # switch to evaluate mode
        self.model_common.eval()
        if "lidar" in self.equipment:
            self.lidar_model.eval()
        if "img" in self.equipment:
            self.img_model.eval()
        if "gps" in self.equipment:
            self.gps_model.eval()
        batch_time = AverageMeter()
        top1 = AverageMeter()
        end = time.time()
        for i, (data, label) in enumerate(self.__test_loader):
            specific_features = torch.zeros((data.shape[0], 3, 64)).float().cuda()
            _data = data.clone().float().cuda()
            for m in [0, 1, 2]:
                if m not in self.mode_split:
                    _data[:, m, :, :, :] = 0
                    specific_features[:, m, :] = 0
                elif m == 0:
                    lidar = _data[:, 0, :, :, :]
                    specific_features[:, m, :] = self.lidar_model(lidar)
                elif m == 1:
                    img = _data[:, 1, :, :, :]
                    specific_features[:, m, :] = self.img_model(img)
                elif m == 2:
                    gps = _data[:, 2, :, :, :]
                    specific_features[:, m, :] = self.gps_model(gps)

            input = _data
            target = label.long().cuda()
            input = input.reshape(-1, input.shape[2], input.shape[3], input.shape[4])
            specific_features = specific_features.reshape(
                -1,
                specific_features.shape[2],
            )
            self.model_common.set_mode_split(self.mode_split)
            shared_features, final_output = self.model_common(input, specific_features)
            # measure accuracy and record loss
            prec1, _ = accuracy(final_output, target, topk=(1, 5))
            top1.update(prec1[0], input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        show_results(self.mode_split, top1, batch_time, "Test ")
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
