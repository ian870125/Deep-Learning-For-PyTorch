#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
import os


# In[2]:


BOARD_ROWS = 3
BOARD_COLS = 3


# In[4]:


class Environment:
    def __init__(self, p1, p2):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.p1 = p1 
        self.p2 = p2 
        self.isEnd = False 
        self.boardHash = None
        self.playerSymbol = 1
    def gethash(self):
        self.boardHash = str(self.board.reshape(BOARD_COLS * BOARD_ROWS))
        return self.boardHash
    def is_done(self):
        for i in range(BOARD_ROWS):
            if sum(self.board[i, :]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[i, :]) == -3:
                self.isEnd = True
                return -1
        for i in range(BOARD_COLS):
            if sum(self.board[:, i]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[:, i]) == -3:
                self.isEnd = True
                return -1
        
        diag_sum1 = sum([self.board[i, i] for i in range(BOARD_COLS)])
        diag_sum2 = sum([self.board[i, BOARD_COLS - i - 1] for i in 
                                                     range(BOARD_COLS)])
        if diag_sum == 3:
            self.isEnd = True
            if diag_sum1 == 3 or diag_sum2 == 3:
                return 1
            else:
                return -1
        
        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0
        self.isEnd = False
        return None
    
    def availablePositions(self):
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i, j] == 0:
                    positions.append((i, j))
        return positions
    
    def updateState(self, position):
        self.board[position] = self.playerSymbol
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    def giveReward(self):
        result = self.is_done()
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(0.1)
            self.p2.feedReward(0.5)
    
    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1
    def play(self, rounds = 100):
        for i in range(rounds):
            if i % 1000 == 0:
                print(f'Rounds {i}')
            while not self.isEnd:
                positions = self.availablePositions()
                p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
                
                self.updateState(p1_action)
                board_hash = self.gethash()
                self.p1.addstate(board_hash)
                
                win = self.is_done()
                
                if win is not None:
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break
                else:
                    positions = self.availablePositions()
                    p2_action = self.p2.chooseAction(positions, self.board, self.playerSymbol)
                    self.updateState(p2_action)
                    board_hash = self.gethash()
                    self.p2.addstate(board_hash)
                    win = self.is_done()
                    if win is not None:
                        self.giveReward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break
    def play2(self, start_player = 1):
        is_first = True
        while not self.isEnd:
            positions = self.availablePositions()
            if not(is_first and start_player == 2):
                p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
                self.updateState(p1_action)
            is_first = False
            self.showBoard()
            win = self.is_done()
            if win is not None:
                if win == -1 or win == 1:
                    print(self.p1.name, 'win')
                else:
                    print('tie')
                self.reset()
                break
            else:
                positions = self.availablePositions()
                p2_action = self.p2.chooseAction(positions)
                self.updateState(p2_action)
                self.showBoard()
                win = self.is_done()
                if win is not None:
                    if win == -1 or win == 1:
                        print(self.p2.name, 'win')
                    else:
                        print('tie')
                    self.reset()
                    break
    def showBoard(self):
        for i in range(BOARD_ROWS):
            print('---------------------')
            out = '| '
            for j in range(BOARD_COLS):
                if self.board[i, j] == 1:
                    token = 'x'
                elif self.board[i, j] == -1:
                    token = 'o'
                else:
                    token = ' '
                out += token + ' | '
            print(out)
        print('---------------------')
    
    


# In[5]:


class player:
    def __init__(self, name, exp_rate = 0.3):
        self.name = name
        self.states = []
        self.lr = 0.2
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}
        
    def gethash(self, board):
        boardhash = str(board.reshape(BOARD_COLS * BOARD_ROWS))
        return boardHash
    
    def chooseAction(self, position, current_board, symbol):
        if np.random.uniform(0, 1) <= self.exp_rate:
            idx = np.random.choice(len(positions))
            action = positions[idx]
        else:
            value_max = -999
            for p in positions:
                next_board = current_board.copy()
                next_board[p] = symbol
                next_boardhash = self.gethash(next_board)
                value = 0 if self.states_value.get(next_board) is None else self.states_value.get(next_boardhash)
                
                if value >= value_max:
                    value_max = value
                    action = p
        return action
    
    def addState(self, state):
        self.states.append(state)
        
    def reset(self):
        self.states = []
        
    def feedReward(self, reward):
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
            
            reward = self.states_value[st]
    def savePolicy(self):
        fw = open(f'policy_{self.name}', 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()
    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()


# In[7]:


class HumanPlayer:
    def __init__(self, name):
        self.name = name
    def chooseAction(self, positions):
        while True:
            position = int(input('輸入位置(1~9):'))
            row = position // 3 
            col = position % 3 - 1
            if col < 0:
                row -= 1
                col = 2
            action = (row, col)
            if action in positions:
                return action
    def addState(self, state):
        pass
    
    def feedReward(self, reward):
        pass
    
    def reset(self):
        pass
    
    def first_draw():
        rv = '\n'
        n0 = 0
        for y in range(3):
            for x in range(3):
                idx = y * 3 + x
                no += 1
                rv += str(no)
                if x < 2 :
                    rv += '|'
            rv += '\n'
            if y < 2:
                rv += '-----\n'
        return rv
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        start_player = int(sys.argv[1])
    else:
        start_player = 1
        
    p1 = Player("p1")
    p2 = Player("p2")
    env = Environment(p1, p2)
    if not os.path.exists(f'policy_p{start_player}'):
        print("開始訓練...")
        env.play(50000)
        p1.savePolicy()
        p2.savePolicy()
    print(first_draw())
    p1 = Player("computer", exp_rate=0)
    p1.loadPolicy(f'policy_p{start_player}')
    p2 = HumanPlayer("human")
    env = Environment(p1, p2)
    
    # 開始比賽
    env.play2(start_player)

