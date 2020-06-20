# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 21:55:23 2020

@author: WKL
"""
import json
import numpy as np
from agent import MyAgent
from game import Game
from my_model import Net
import torch

def generate_fingerprint(AgentClass, model, **kwargs):
    with open("board_cases.json") as f:
        board_json = json.load(f)

    game = Game(size=4, enable_rewrite_board=True)
    agent = AgentClass(game=game, myNet=model, **kwargs)

    trace = []
    for board in board_json:
        game.board = np.array(board)
        direction = agent.step()
        trace.append(direction)
    fingerprint = "".join(str(i) for i in trace)
    return fingerprint


from collections import Counter

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

fingerprint = generate_fingerprint(AgentClass=TestAgent, model=model.cpu())
#fingerprint = generate_fingerprint(AgentClass=TestAgent, model=model)

with open("EE369_fingerprint.json", 'w') as f:        
    pack = dict()
    pack['fingerprint'] = fingerprint
    pack['statstics'] = dict(Counter(fingerprint))
    f.write(json.dumps(pack, indent=4))