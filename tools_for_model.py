# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 13:48:05 2020

@author: WKL
"""

import torch

# 用one-hot编码表示board
# 位置示意
#  1--2--3--4
#  5--6--7--8
#  9-10-11-12
# 13-14-15-16
# 编码示意
# 0-10  <-----> 1024-512-...-2-0
# 改成返回大小为11*4*4
def oneHotEncoding(board):
    
    from math import log
    data = torch.zeros(11, 4, 4, requires_grad=True)
    for i in range(4):
        for j in range(4):
            if board[i, j] == 0:
                data[10, i, j] = 1
            else:
                data[10-int(log(board[i, j], 2)), i, j] = 1
    #data -= 0.1 * torch.ones(16, 11)
    return data


# 获得初始数据
'''
def generateTrainSet(batch_size = 256):
    from expectimax import board_to_move
    data = torch.zeros(batch_size, 11, 4, 4, requires_grad=True)
    labels = torch.zeros(batch_size, dtype = torch.long)
    pos = 0
    #TrainSetSize
    game = Game(4, random=False)
    while pos<batch_size:
        #如果结束，则重新开始
        if game.end==1:
            game = Game(4, random=False)
        
        raw_data = oneHotEncoding(game.board)
        direction = board_to_move(game.board)
        data[pos, :, :, :] = raw_data
        labels[pos] = direction
        pos = pos + 1
        game.move(direction)
    return data, labels
'''


# 获得初始数据
# 改成返回值为epoch*batch_size*11*4*4
def generateTrainSet(epoch = 50, batch_size = 256):
    
    from expectimax import board_to_move
    from game import Game
    
    data = torch.zeros(epoch, batch_size, 11, 4, 4)
    labels = torch.zeros(epoch, batch_size, dtype = torch.long)
    for idx in range(epoch):
        pos = 0
        game = Game(4, random=False)
        while pos<batch_size:
            #如果结束，则重新开始
            if game.end==1:
                game = Game(4, random=False)
            raw_data = oneHotEncoding(game.board)
            direction = board_to_move(game.board)
            data[idx, pos, :, :, :] = raw_data
            labels[idx, pos] = direction
            pos = pos + 1
            game.move(direction)
    return data, labels