import torch
from torch import nn
from torch import optim

layer_size = 1024

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(input_layer, layer_size)#5917, layer_size)
        self.hidden2 = nn.Linear(layer_size, layer_size*2)
        self.hidden3 = nn.Linear(layer_size*2, layer_size*3)
        self.hidden4 = nn.Linear(layer_size*3, layer_size*4)
        self.hidden5 = nn.Linear(layer_size*4, layer_size*3)
        self.hidden6 = nn.Linear(layer_size*3, layer_size*2)
        self.hidden7 = nn.Linear(layer_size*2, layer_size)
        self.hidden8 = nn.Linear(layer_size, output_layer)
        #self.act1 = nn.ReLU()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = self.hidden6(x)
        x = self.hidden7(x)
        x = self.hidden8(x)
        return x


kernel_size = (5,5)

class NetworkImg(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Conv2d(1, 1, kernel_size, stride=1)
        self.hidden2 = nn.Conv2d(1, 2, kernel_size, stride=1)
        self.hidden3 = nn.Conv2d(2, 2, kernel_size, stride=1)
        self.hidden4 = nn.Conv2d(2, 2, kernel_size, stride=1)
        self.hidden5 = nn.Conv2d(2, 3, kernel_size, stride=1)
        self.hidden6 = nn.Conv2d(3, 3, kernel_size, stride=1)
        self.hidden7 = nn.Conv2d(3, 3, kernel_size, stride=1)
        self.hidden8 = nn.Conv2d(3, 4, kernel_size, stride=1)
        self.hidden9 = nn.Conv2d(4, 4, kernel_size, stride=1)
        self.hidden10 = nn.Conv2d(4, 4, kernel_size, stride=1)
        #self.pool1 = nn.MaxPool2d((3,2))
        

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = self.hidden6(x)
        x = self.hidden7(x)
        x = self.hidden8(x)
        x = self.hidden9(x)
        x = self.hidden10(x)
        return x


class NetworkVolume(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad1 = nn.ConstantPad3d((0, 0, 10, 6, 7, 4), 0)
        self.conv1 = nn.Conv2d(35, 32, 1)
        self.conv2 = nn.Conv2d(32, 30, 1)
        self.conv3 = nn.Conv2d(30, 28, 1)
        self.conv4 = nn.Conv2d(28, 24, 1)
        self.conv5 = nn.Conv2d(24, 22, 1)
        self.conv6 = nn.Conv2d(22, 20, 1)
        self.conv7 = nn.Conv2d(20, 18, 1)
        self.conv8 = nn.Conv2d(18, 15, 1)
        self.relu1 = nn.ReLU()
        self.conv9 = nn.Conv2d(15, 12, 1)
        self.conv10 = nn.Conv2d(12, 10, 1)
        self.conv11 = nn.Conv2d(10, 9, 1)
        self.conv12 = nn.Conv2d(9, 8, 1)
        self.conv13 = nn.Conv2d(8, 7, 1)
        self.conv14 = nn.Conv2d(7, 6, 1)
        self.conv15 = nn.Conv2d(6, 5, 1)
        self.conv16 = nn.Conv2d(5, 4, 1)
        

    def forward(self, x):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.relu1(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        return x
