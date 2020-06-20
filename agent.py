# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 13:45:13 2020

@author: WKL
"""

import numpy as np
#好像并不需要这行语句
#from my_model import Net

class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        
        #从这里导入用cpp写好的expectimax agent
        from expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction


#定义自己的agent
class MyAgent(Agent):

    def __init__(self, game, myNet, display=None):
        super().__init__(game, display)
        self.model = myNet
        
    def step(self):
        '''
            direction:
            0: left
            1: down
            2: right
            3: up
        '''
        direction = self.model.predictDirection(self.game.board)
        return direction