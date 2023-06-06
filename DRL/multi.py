import cv2
import torch
import numpy as np
from utils.util import *

FRIDA = False
FRIDA_Renderer = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class fastenv():
    def __init__(self, opt, writer=None):
        self.max_episode_length = opt.max_step
        self.env_batch = opt.env_batch

        if FRIDA:
            from env2 import PaintFRIDA
            self.env = PaintFRIDA(opt)
        else:
            from env import Paint
            from DRL.ddpg import decode
            self.env = Paint(opt)
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space

        if opt.dataset == 'celeba':
            self.env.load_data_celeba()
        elif opt.dataset == 'pascal':
            self.env.load_data_pascal()
        elif opt.dataset == 'sketchy':
            self.env.load_data_sketchy()
        elif opt.dataset == 'cats':
            self.env.load_data_cat()
        elif opt.dataset == 'all':
            self.env.load_data_all()
        elif opt.dataset == 'mnist':
            self.env.load_data_mnist()

        self.writer = writer
        self.test = False
        self.log = 0
        self.dataset = opt.dataset
        self.opt = opt

    def save_image(self, log, step):
        #from paint_utils3 import show_img

        for i in range(self.env_batch):
            if self.env.imgid[i] <= 10:
                #show_img(env.env.canvas[i])
                canvas = cv2.cvtColor((to_numpy(self.env.canvas[i].permute(1, 2, 0))), cv2.COLOR_BGR2RGB)
                #print("Saved Image!")
                self.writer.add_image('{}/canvas_{}.png'.format(str(self.env.imgid[i]), str(step)), canvas, log)
        if step == self.max_episode_length:
            for i in range(self.env_batch):
                if self.env.imgid[i] < 50:
                    gt = cv2.cvtColor((to_numpy(self.env.gt[i].permute(1, 2, 0))), cv2.COLOR_BGR2RGB)
                    canvas = cv2.cvtColor((to_numpy(self.env.canvas[i].permute(1, 2, 0))), cv2.COLOR_BGR2RGB)
                    
                    if self.env.mask is not None:
                        mask = self.env.mask[i]
                        mask = cv2.cvtColor((to_numpy(mask.permute(1, 2, 0))), cv2.COLOR_BGR2RGB)
                        self.writer.add_image(str(self.env.imgid[i]) + '/_mask.png', mask, log)
                    self.writer.add_image(str(self.env.imgid[i]) + '/_target.png', gt, log)
                    self.writer.add_image(str(self.env.imgid[i]) + '/_canvas.png', canvas, log)
                    #print("Saved Image!")

    def step(self, action, episode_num):
        with torch.no_grad():
            ob, r, d, alpha, mask = self.env.step(torch.tensor(action).to(device), episode_num)
        if d[0]:
            if not self.test:
                self.dist = self.get_dist()
                for i in range(self.env_batch):
                    self.writer.add_scalar('train/dist', self.dist[i], self.log)
                    self.log += 1
        return ob, r, d, alpha, mask

    def get_dist(self):
        return to_numpy((((self.env.gt.float() - self.env.canvas.float()) / 255) ** 2).mean(1).mean(1).mean(1))
        
    def reset(self, test=False, episode=0):
        self.test = test
        ob = self.env.reset(self.test, episode * self.env_batch)
        return ob
