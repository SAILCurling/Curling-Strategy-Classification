import numpy as np 
from copy import copy

from config import Config

class Memory:
    def __init__(self, size):
        self.size = size

        # input of network
        self.cur_bodies = np.empty((self.size, 16, 2), dtype = np.float32)
        self.cur_ShotNums = np.empty((self.size), dtype = np.int16)
        self.cur_WhiteToMoves = np.empty((self.size), dtype = np.bool)

        self.next_bodies = np.empty((self.size, 16, 2), dtype = np.float32)
        self.next_ShotNums = np.empty((self.size), dtype = np.int16)
        self.next_WhiteToMoves = np.empty((self.size), dtype = np.bool)

        # labels
        self.strategies = np.empty((self.size), dtype = np.int16)

        self.count = 0
        self.current = 0

    def add(self, cur_game_record, next_game_record):
        self.cur_bodies[self.current, ...] = cur_game_record['body']
        self.cur_ShotNums[self.current] =  cur_game_record['ShotNum']
        self.cur_WhiteToMoves[self.current] = cur_game_record['WhiteToMove']

        self.next_bodies[self.current, ...] = next_game_record['body']
        self.next_ShotNums[self.current] =  next_game_record['ShotNum']
        self.next_WhiteToMoves[self.current] = next_game_record['WhiteToMove']

        self.strategies[self.current, ...] = cur_game_record['ShotName']

        self.count = max(self.count, self.current + 1) 
        self.current = (self.current + 1) % self.size 

    def sample_batch(self, type='all'):
        if type == 'train':
            shuffle_idx = np.arange(0, int(self.count * 0.95))
        elif type == 'test':
            shuffle_idx = np.arange(int(self.count * 0.95), self.count)
        elif type == 'all':
            shuffle_idx = np.arange(0, self.count)
        else:
            assert(0)
        np.random.shuffle(shuffle_idx)
        shuffle_idx = shuffle_idx[:Config.train.BATCH_SIZE]
        
        cur_bodies = self.cur_bodies[shuffle_idx]
        cur_ShotNums = self.cur_ShotNums[shuffle_idx]
        cur_WhiteToMoves = self.cur_WhiteToMoves[shuffle_idx]

        next_bodies = self.next_bodies[shuffle_idx]
        next_ShotNums = self.next_ShotNums[shuffle_idx]
        next_WhiteToMoves = self.next_WhiteToMoves[shuffle_idx]

        strategies = self.strategies[shuffle_idx]
        return copy(cur_bodies), cur_ShotNums, cur_WhiteToMoves, copy(next_bodies), next_ShotNums, next_WhiteToMoves, strategies