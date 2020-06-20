# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 17:00:56 2020

@author: WKL
"""
from game import Game
from displays import Display
from agent import MyAgent
from my_model import Net
import torch
import time


def single_run(size, score_to_win, AgentClass, model, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game, model, display=Display(), **kwargs)
    agent.play(verbose=True)
    return game.score

if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    N_TESTS = 10
    
    '''====================
    Use your own agent here.'''
    PATH = './2048.pth'  
    #PATH = './2048_CPU.pth'  
    model = Net()  
    model.load_state_dict(torch.load(PATH))  
    #这行语句在测试的时候一定要加，不然参数可能发生变化
    model.eval()
    TestAgent = MyAgent
    '''===================='''
    
    scores = []
    for _ in range(N_TESTS):
        score = single_run(GAME_SIZE, SCORE_TO_WIN,
                           AgentClass=TestAgent, model = model.cpu())
        #score = single_run(GAME_SIZE, SCORE_TO_WIN,
        #                   AgentClass=TestAgent, model = model)
        scores.append(score)
        time.sleep(1)
    
    print("Average scores: @%s times" % N_TESTS, sum(scores) / len(scores))