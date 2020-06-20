# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 13:46:46 2020

@author: WKL
"""

import torch
from tools_for_model import oneHotEncoding

#定义神经网络
#定义神经网络
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv_filter = 64
        
        #输入4*4*11，输出1*4*self.conv_filter
        self.conv14 = torch.nn.Sequential(
                        torch.nn.Conv2d(11, self.conv_filter, kernel_size=(1,2), bias=True),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(self.conv_filter, self.conv_filter, kernel_size=(1,2), bias=True),
                        torch.nn.ReLU(),
                        torch.nn.MaxPool2d(stride=1, kernel_size=(1,2)))
        
        #输入4*4*11，输出4*1*self.conv_filter
        #这里注意细节，一开始搞错了，池化的kernel_size应该是2，1，一开始写成了1，2
        #导致输出为3*2*self.conv_filter
        self.conv41 = torch.nn.Sequential(
                        torch.nn.Conv2d(11, self.conv_filter, kernel_size=(2,1), bias=True),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(self.conv_filter, self.conv_filter, kernel_size=(2,1), bias=True),
                        torch.nn.ReLU(),
                        torch.nn.MaxPool2d(stride=1, kernel_size=(2,1)))
    
        #输入4*4*11，输出1*1*self.conv_filter
        self.conv41_14 = torch.nn.Sequential(
                        torch.nn.Conv2d(11, self.conv_filter, kernel_size=(2,1), bias=True),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(self.conv_filter, self.conv_filter, kernel_size=(1,2), bias=True),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(self.conv_filter, self.conv_filter, kernel_size=(2,2), bias=True),
                        torch.nn.ReLU(),
                        torch.nn.MaxPool2d(stride=1, kernel_size=(2,2)))
        
        #输入4*4*11，输出2*2*self.conv_filter
        self.conv22_2 = torch.nn.Sequential(
                        torch.nn.Conv2d(11, self.conv_filter, kernel_size=(2,2), stride=2, bias=True),
                        torch.nn.ReLU())
        
        #输入4*4*11，输出2*2*self.conv_filter
        self.conv22_1 = torch.nn.Sequential(
                        torch.nn.Conv2d(11, self.conv_filter, kernel_size=(2,2), stride=1, bias=True),
                        torch.nn.ReLU(),
                        torch.nn.MaxPool2d(stride=1, kernel_size=(2,2)))
        
        #输入4*4*11，输出2*2*self.conv_filter
        self.conv33 = torch.nn.Sequential(
                        torch.nn.Conv2d(11, self.conv_filter, kernel_size=(3,3), stride=1, bias=True),
                        torch.nn.ReLU())
        
        #输入4*4*11，输出1*1*self.conv_filter
        self.conv44 = torch.nn.Sequential(
                        torch.nn.Conv2d(11, self.conv_filter, kernel_size=(4,4), stride=1, bias=True),
                        torch.nn.ReLU())
        
        #全连接神经网络
        #dropout 防止过拟合
        self.dense = torch.nn.Sequential(
                        torch.nn.Linear(self.conv_filter * 22, 1024),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(p=0.3),
            
                        torch.nn.Linear(1024, 256),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(p=0.4),
            
                        torch.nn.Linear(256, 64),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(p=0.5),
                        torch.nn.Linear(64, 4))
        
        self.norm = torch.nn.BatchNorm1d(self.conv_filter * 22)

    def forward(self, x):
        #因为是多批次传入的，所以必须指定第一维是x.size()[0]
        #dim=1是按列合并，即横向合并
        batch_size = x.size()[0]
        #卷积
        x = torch.cat([self.conv14(x).view(batch_size, -1),
                       self.conv41(x).view(batch_size, -1),
                       self.conv41_14(x).view(batch_size, -1),
                       self.conv22_1(x).view(batch_size, -1),
                       self.conv22_2(x).view(batch_size, -1),
                       self.conv33(x).view(batch_size, -1),
                       self.conv44(x).view(batch_size, -1)], dim=1)
        
        #normalization
        x = self.norm(x)
        #全连接
        x = self.dense(x)
        
        return x

    def predictDirection(self, board):
        d = oneHotEncoding(board)
        d = d.view([1, 11, 4, 4])
        result = self(d)
        _, predict = torch.max(result.data, -1)
        return int(predict)
    
    

#初始化函数
def weights_init(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        #推荐在ReLU网络中使用
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')
        torch.nn.init.constant_(m.bias, 0.0)