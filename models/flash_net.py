from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

torch.manual_seed(1)
# torch.set_deterministic(True)
torch.backends.cudnn.deterministic = True


###orginal lidar model from infocom
class FlashNet(nn.Module):
    print("**************using orginal lidar model from infocom***********")

    def __init__(self, modality, num_classes, shrink=1):
        super(FlashNet, self).__init__()
        self.shrink = shrink

        dropProb1 = 0.3
        dropProb2 = 0.2
        channel = 32
        # self.conv1 = nn.Conv2d(90, channel, kernel_size=(3, 3), bias=False)
        self.conv1 = nn.Conv2d(45, channel, kernel_size=(3, 3), bias=False)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)
        self.conv3 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)
        self.conv4 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)
        self.conv5 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)
        self.conv6 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)
        self.conv7 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)
        self.conv8 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)
        self.conv9 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)

        self.pool1 = nn.MaxPool2d((2, 2))
        self.pool2 = nn.MaxPool2d((1, 2))

        # self.hidden1 = nn.Linear(320, 1024)  #orginal
        # self.hidden1 = nn.Linear(2560, 1024)    #with zero padding
        self.hidden1 = nn.Linear(1280, 1024)  # with zero padding image/2
        # self.hidden2 = nn.Linear(1024, 512)  # we are not using this
        self.hidden3 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, 64)  # 128
        #######################
        self.drop1 = nn.Dropout(dropProb1)
        self.drop2 = nn.Dropout(dropProb2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, x):
        # FOR CNN BASED IMPLEMENTATION
        x = F.pad(x, (1, 1, 1, 1))
        a = x = self.relu(self.conv1(x))
        x = F.pad(x, (1, 1, 1, 1))
        x = self.relu(self.conv2(x))
        x = F.pad(x, (1, 1, 1, 1))
        x = self.relu(self.conv3(x))
        x = torch.add(x, a)
        x = self.pool1(x)
        b = x = self.drop1(x)

        x = F.pad(x, (1, 1, 1, 1))
        x = self.relu(self.conv4(x))
        x = F.pad(x, (1, 1, 1, 1))
        x = self.relu(self.conv5(x))
        x = torch.add(x, b)
        x = self.pool1(x)
        c = x = self.drop1(x)

        x = F.pad(x, (1, 1, 1, 1))
        x = self.relu(self.conv6(x))
        x = F.pad(x, (1, 1, 1, 1))
        x = self.relu(self.conv7(x))
        x = torch.add(x, c)
        x = self.pool2(x)
        d = x = self.drop1(x)

        x = F.pad(x, (1, 1, 1, 1))
        x = self.relu(self.conv8(x))
        x = F.pad(x, (1, 1, 1, 1))
        x = self.relu(self.conv9(x))
        x = torch.add(x, d)
        # print('xshape',x.shape)
        #########
        x = x.view(x.size(0), -1)

        # print("shape", x.shape)
        x = self.relu(self.hidden1(x))
        x = self.drop2(x)

        x = self.relu(self.hidden3(x))
        x = self.drop2(x)
        x = self.out(x)  # no softmax: CrossEntropyLoss()
        return x


class FlashNet_common(nn.Module):
    print("**************using orginal lidar model from infocom***********")

    def __init__(self, task, num_classes, shrink=1):
        super(FlashNet_common, self).__init__()
        self.flashnet = FlashNet(task, num_classes, shrink)
        channel = 128
        dropProb1 = 0.3
        dropProb2 = 0.4
        self.proj = nn.Linear(128, 64)
        self.fc1 = nn.Linear(192, 128)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(dropProb1)
        self.drop2 = nn.Dropout(dropProb2)
        self.out = nn.Linear(128, num_classes)

    def set_mode_split(self, mode_split):
        self.mode_split = mode_split

    def forward(self, x1, spec_features, val=False):
        # FOR CNN BASED IMPLEMENTATION
        shared_ft = self.flashnet(x1)
        general_shared_ft = torch.zeros_like(shared_ft[0 : len(shared_ft) + 1 : 3])
        for m in self.mode_split:
            general_shared_ft += shared_ft[m : len(shared_ft) + 1 : 3]
        general_shared_ft /= len(self.mode_split)
        r0 = shared_ft[0 : len(shared_ft) + 1 : 3]
        r1 = shared_ft[1 : len(shared_ft) + 1 : 3]
        r2 = shared_ft[2 : len(shared_ft) + 1 : 3]
        s0 = spec_features[0 : len(spec_features) + 1 : 3]
        s1 = spec_features[1 : len(spec_features) + 1 : 3]
        s2 = spec_features[2 : len(spec_features) + 1 : 3]
        if 0 in self.mode_split:
            f0 = torch.cat((r0, s0), 1)
            f0 = self.relu(self.proj(f0))
            f0 = f0 + r0
        else:
            f0 = general_shared_ft
            # shared_ft[0 : len(shared_ft) + 1 : 3] = general_shared_ft
        if 1 in self.mode_split:
            f1 = torch.cat((r1, s1), 1)
            f1 = self.relu(self.proj(f1))
            f1 = f1 + r1
        else:
            f1 = general_shared_ft
            # shared_ft[1 : len(shared_ft) + 1 : 3] = general_shared_ft
        if 2 in self.mode_split:
            f2 = torch.cat((r2, s2), 1)
            f2 = self.relu(self.proj(f2))
            f2 = f2 + r2
        else:
            f2 = general_shared_ft
            # shared_ft[2 : len(shared_ft) + 1 : 3] = general_shared_ft

        x = torch.cat((f0, f1, f2), 1)
        x = self.drop1(x)
        x = self.relu(self.fc1(x))
        x = self.drop2(x)
        out = self.out(x)  # + spec_features
        return shared_ft, out
