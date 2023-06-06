# from collections import deque
import numpy as np
import random
import torch
import pickle as pickle

class rpm(object):
    # replay memory
    def __init__(self, buffer_size, item_count=6):
        self.buffer_size = buffer_size
        self.buffer = []
        self.index = 0
        self.item_count = item_count

    def append(self, obj):
        if self.size() > self.buffer_size:
            print('buffer size larger than set value, trimming...')
            self.buffer = self.buffer[(self.size() - self.buffer_size):]
        elif self.size() == self.buffer_size:
            self.buffer[self.index] = obj
            self.index += 1
            self.index %= self.buffer_size
        else:
            self.buffer.append(obj)

    def size(self):
        return len(self.buffer)

    def sample_batch(self, batch_size, device, only_state=False):
        if self.size() < batch_size:
            batch = random.sample(self.buffer, self.size())
        else:
            batch = random.sample(self.buffer, batch_size)
        if only_state:
            res = torch.stack(tuple(item[3] for item in batch), dim=0)            
            return res.to(device)
        else:
            #TODO
            #item_count = 6 #7
            res = []
            for i in range(self.item_count): #6):
                if batch[0][i] is None:
                    res.append(None) # No mask
                    continue
                k = torch.stack(tuple(item[i] for item in batch), dim=0)
                res.append(k.to(device))

            if self.item_count == 7:
                return res[0], res[1], res[2], res[3], res[4], res[5], res[6]    
            else:
                return res[0], res[1], res[2], res[3], res[4], res[5]#, res[6]
