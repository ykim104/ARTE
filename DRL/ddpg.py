import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from Renderer.model import *
from DRL.rpm import rpm
from DRL.actor import *
from DRL.critic import *
from DRL.wgan import *
from utils.util import *

import copy

from DRL.content_loss import *
#from DRL.clip_loss import *
from DRL.clip_loss2 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

width = 128
coord = torch.zeros([1, 2, width, width])
for i in range(width):
    for j in range(width):
        coord[0, 0, i, j] = i / (width-1.)
        coord[0, 1, i, j] = j / (width-1.)
coord = coord.to(device)

criterion = nn.MSELoss()

#decoder_fns = [
#                'pretrained/renderer/renderer0.02.pkl'
               #'renderer_models/renderer0.05.pkl',
               #'renderer_models/renderer0.01.pkl'
               #'renderer_models/renderer0.1999999999999999.pkl', 
               #'renderer_models/renderer0.09999999999999985.pkl', 
               #'renderer_models/renderer0.049999999999999864.pkl', 
               #'renderer_models/renderer0.01.pkl'
#              ]
#decoder_cutoff = []#[50]#[10, 30, 65]
#decoders = []
#for decoder_fn in decoder_fns:
#    dec = FCN()
#    dec.load_state_dict(torch.load(decoder_fn))
#    decoders.append(copy.deepcopy(dec).to(device))


# TODO make options pass the renderer model
Decoder = FCN(input_size=7).to(device)
Decoder.load_state_dict(torch.load('models/renderers/FRIDA_lite6_renderer_0318.pkl')) 
n_strokes = 5
print("LOADED FRIDA renderer.")

def decode(x, canvas, brush_color="color"): # b * (10 + 3)
    n_strokes = int(x.shape[1]/10)
    #x = x.view(-1, 10 + 3)
    x = x.view(-1, 7 + 3)
    #print(x[0,:10])

    #stroke = 1 - Decoder(x[:, :10])
    stroke = 1 - Decoder(x[:, :7])

    stroke = stroke.view(-1, width, width, 1)
    #colored
    if brush_color=="color":
        color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
    elif brush_color=="white":
        color_stroke = stroke * torch.tensor([1, 1, 1]).view(-1,1,1,3).to(device) #white strokes
    elif brush_color=="black":
        color_stroke = stroke * torch.tensor([0, 0, 0]).view(-1,1,1,3).to(device) #white strokes
    
    stroke = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    stroke = stroke.view(-1, n_strokes, 1, width, width)
    color_stroke = color_stroke.view(-1, n_strokes, 3, width, width)
    
    for i in range(n_strokes):
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]

    return canvas

'''
def decode_multiple_renderers(x, canvas, episode_num): # b * (10 + 3)
    x = x.view(-1, 10 + 3)
    
    dec_ind = 0
    d = decoders[-1]
    for cutoff in decoder_cutoff:
        if episode_num < cutoff:
            d = decoders[dec_ind]
            break
        dec_ind += 1

    x = x.view(-1, 10 + 3)
    stroke = 1 - d(x[:, :10])
    stroke = stroke.view(-1, width, width, 1)
    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
    stroke = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    stroke = stroke.view(-1, n_strokes, 1, width, width)
    color_stroke = color_stroke.view(-1, n_strokes, 3, width, width)
    for i in range(n_strokes):
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
    return canvas
'''


def cal_trans(s, t):
    return (s.transpose(0, 3) * t).transpose(0, 3)
    
class DDPG(object):
    def __init__(self, opt, writer=None):
        self.opt = opt
        self.max_step = opt.max_step
        self.env_batch = opt.env_batch
        self.batch_size = opt.batch_size
        self.loss_fcn = opt.loss_fcn
        self.use_multiple_renderers = False # opt.use_multiple_renderers

        state_size = 9
        if (opt.loss_fcn == 'cm' or opt.loss_fcn == 'cml1')  and self.opt.built_in_cm:
            state_size = 10

        #self.actor = ResNet(state_size, 18, 13*n_strokes) # target, canvas, stepnum, coordconv 3 + 3 + 1 + 2
        #self.actor_target = ResNet(state_size, 18, 13*n_strokes)
        self.actor = ResNet(state_size, 18, 10*n_strokes) # target, canvas, stepnum, coordconv 3 + 3 + 1 + 2
        self.actor_target = ResNet(state_size, 18, 10*n_strokes)
        self.critic = ResNet_wobn(3 + state_size, 18, 1) # add the last canvas for better prediction
        self.critic_target = ResNet_wobn(3 + state_size, 18, 1) 

        #if not opt.use_multiple_renderers:
        #    Decoder.load_state_dict(torch.load('FRIDA_renderer.pkl')) #opt.renderer))

        self.actor_optim  = Adam(self.actor.parameters(), lr=1e-3) #learing rate: 1e-2
        self.critic_optim  = Adam(self.critic.parameters(), lr=1e-3)

        if (opt.resume != None):
            self.load_weights(opt.resume)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)
        
        # Create replay buffer
        self.memory = rpm(opt.rmsize * opt.max_step)

        # Hyper-parameters
        self.tau = opt.tau
        self.discount = opt.discount

        # Tensorboard
        self.writer = writer
        self.log = 0
        
        self.state = [None] * self.env_batch # Most recent state
        self.action = [None] * self.env_batch # Most recent action
        self.choose_device() 

    def play(self, state, target=False):
        if (self.loss_fcn == 'cm' or self.loss_fcn == 'cml1') and self.opt.built_in_cm:
            state = torch.cat((state[:, :6].float() / 255,  #canvas and target \
                               state[:, 6:7].float() / 255, # mask \
                               state[:, 6+1:7+1].float() / self.max_step, # step num \
                               coord.expand(state.shape[0], 2, width, width)), 1)
        else:
            state = torch.cat((state[:, :6].float() / 255,  #canvas and target \
                               state[:, 6:7].float() / self.max_step, # step num \
                               coord.expand(state.shape[0], 2, width, width)), 1)
        if target:
            return self.actor_target(state)
        else:
            return self.actor(state)

    def update_gan(self, state):
        canvas = state[:, :3]
        gt = state[:, 3 : 6]
        fake, real, penal = update(canvas.float() / 255, gt.float() / 255)
        if self.log % 20 == 0:
            self.writer.add_scalar('train/gan_fake', fake, self.log)
            self.writer.add_scalar('train/gan_real', real, self.log)
            self.writer.add_scalar('train/gan_penal', penal, self.log)       
        
    def evaluate(self, state, action, episode_num, mask=None, target=False):
        T = state[:, 6 : 7]
        gt = state[:, 3 : 6].float() / 255
        canvas0 = state[:, :3].float() / 255
        if self.use_multiple_renderers:
            canvas1 = decode_multiple_renderers(action, canvas0, episode_num)
        else:
            canvas1 = decode(action, canvas0, brush_color=self.opt.brush_color)

        if (self.loss_fcn == 'cm' or self.loss_fcn == 'cml1')  and self.opt.built_in_cm:
            T = state[:, 6+1 : 7+1]
            mask = state[:, 6:7].float() / 255

        reward = 0
        clip = 0.2

        
        if self.loss_fcn == 'gan':
            reward = cal_reward(canvas1, gt) - cal_reward(canvas0, gt)
        
        elif self.loss_fcn == 'l2':
            reward = ((canvas0 - gt) ** 2).mean(1).mean(1).mean(1) - ((canvas1 - gt) ** 2).mean(1).mean(1).mean(1)
        
        elif self.loss_fcn == 'l1':
            l1_0 = torch.abs(canvas0-gt)
            l1_0[l1_0 > clip] = clip

            l1_1 = torch.abs(canvas1-gt)
            l1_1[l1_1 > clip] = clip
            reward = (l1_0).mean(1).mean(1).mean(1) - (l1_1).mean(1).mean(1).mean(1)
        elif self.loss_fcn == 'cm':
            #elif self.loss_fcn == 'l2':
            reward += ((canvas0 - gt) ** 2).mean(1).mean(1).mean(1) - ((canvas1 - gt) ** 2).mean(1).mean(1).mean(1)
        
            try:
                #TODO: need to update mask if curriculum learning
                #mask = get_l2_mask(gt)
                #reward = ((canvas0 - gt) ** 2 * mask).mean(1).mean(1).mean(1) - ((canvas1 - gt) ** 2  * mask).mean(1).mean(1).mean(1)
                reward += ((canvas0 - gt) ** 2 * (mask + 0.1)).mean(1).mean(1).mean(1) - ((canvas1 - gt) ** 2  * (mask + 0.1)).mean(1).mean(1).mean(1)
            except Exception as e:
                print(e)
        
        elif self.loss_fcn == 'cml1':
            mask = get_l2_mask(gt)

            l1_0 = torch.abs(canvas0-gt)
            l1_0[l1_0 > clip] = clip

            l1_1 = torch.abs(canvas1-gt)
            l1_1[l1_1 > clip] = clip

            reward = (l1_0 * mask).mean(1).mean(1).mean(1) - (l1_1 * mask).mean(1).mean(1).mean(1)
        elif self.loss_fcn == 'l1_penalized':
            l1_0 = torch.abs(canvas0-gt)
            l1_0[l1_0 > clip] = clip

            l1_1 = torch.abs(canvas1-gt)
            l1_1[l1_1 > clip] = clip

            lam = 2.0

            diff = l1_0 - l1_1
            diff = diff * lam * (diff < 0) + diff * (diff > 0)

            reward = diff.mean(1).mean(1).mean(1)


        # CLIP loss
        elif self.loss_fcn == "clip":
            #reward0 = get_clip_conv_loss(canvas0, gt)
            #reward1 = get_clip_conv_loss(canvas1, gt)

            reward0 = clip_conv_loss(canvas0, gt)
            reward1 = clip_conv_loss(canvas1, gt)
            reward = 160.0*(reward0 - reward1)
            self.writer.add_scalar('train/clip_loss_0', reward0.mean(), self.log)
            self.writer.add_scalar('train/clip_loss_1', reward1.mean(), self.log)


        coord_ = coord.expand(state.shape[0], 2, width, width)
        if (self.loss_fcn == 'cm' or self.loss_fcn == 'cml1')  and self.opt.built_in_cm:
            merged_state = torch.cat([canvas0, canvas1, gt, mask, (T + 1).float() / self.max_step, coord_], 1)
        else:
            merged_state = torch.cat([canvas0, canvas1, gt, (T + 1).float() / self.max_step, coord_], 1)
        # canvas0 is not necessarily added
        if target:
            Q = self.critic_target(merged_state)
            return (Q + reward), reward#, [reward0, reward1]
        else:
            Q = self.critic(merged_state)
            if self.log % 20 == 0:
                self.writer.add_scalar('train/expect_reward', Q.mean(), self.log)
                self.writer.add_scalar('train/reward', reward.mean(), self.log)
            return (Q + reward), reward#, [reward0, reward1]
                
    def update_policy(self, lr, episode_num):
        self.log += 1
        
        for param_group in self.critic_optim.param_groups:
            param_group['lr'] = lr[0]
        for param_group in self.actor_optim.param_groups:
            param_group['lr'] = lr[1]
            
        # Sample batch
        state, action, reward, \
            next_state, terminal, mask = self.memory.sample_batch(self.batch_size, device)

        if self.loss_fcn == 'gan':
            self.update_gan(next_state)
        
        with torch.no_grad():
            next_action = self.play(next_state, True)
            target_q, _ = self.evaluate(next_state, next_action, episode_num, target=True, mask=mask)
            target_q = self.discount * ((1 - terminal.float()).view(-1, 1)) * target_q
                
        cur_q, step_reward = self.evaluate(state, action, episode_num, mask=mask)
        target_q += step_reward.detach()
        
        value_loss = criterion(cur_q, target_q)
        self.critic.zero_grad()
        value_loss.backward(retain_graph=True)
        self.critic_optim.step()

        action = self.play(state)
        pre_q, _ = self.evaluate(state.detach(), action, episode_num, mask=mask)
        policy_loss = -pre_q.mean()
        self.actor.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.actor_optim.step()
        
        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return -policy_loss, value_loss#, clip_loss

    def observe(self, reward, state, done, step, mask):
        # s0 = torch.tensor(self.state, device='cpu')
        s0 = self.state.cpu().clone().detach()
        a = to_tensor(self.action, "cpu")
        r = to_tensor(reward, "cpu")
        # s1 = torch.tensor(state, device='cpu')
        s1 = state.cpu().clone().detach()
        d = to_tensor(done.astype('float32'), "cpu")
        m = mask.cpu().clone().detach() if mask is not None else None
        for i in range(self.env_batch):
            self.memory.append([s0[i], a[i], r[i], s1[i], d[i], m[i] if mask is not None else None])
        self.state = state

    def noise_action(self, noise_factor, state, action):
        noise = np.zeros(action.shape)
        for i in range(self.env_batch):
            action[i] = action[i] + np.random.normal(0, self.noise_level[i], action.shape[1:]).astype('float32')
        return np.clip(action.astype('float32'), 0, 1)
    
    def set_action(self, state, action):
        self.action = action 
        
        if True: #debug:
            canvas = state[:, :3].float() / 255
            new_observation = decode(action.to("cuda:0"), canvas.to("cuda:0"), brush_color=self.opt.brush_color) 
            cv2.imshow("updated image",new_observation[0].cpu().detach().numpy().transpose(1, 2, 0))
            
            k=cv2.waitKey(1)
            if k==27:
                cv2.destroyWindow("updated image")
                

    def select_action(self, state, return_fix=False, noise_factor=0):
        self.eval()
        with torch.no_grad():
            action = self.play(state)
            action = to_numpy(action)

        if noise_factor > 0:        
            action = self.noise_action(noise_factor, state, action)

        self.train()
        self.action = action
        if return_fix:
            return action
        return self.action

    def reset(self, obs, factor):
        self.state = obs
        self.noise_level = np.random.uniform(0, factor, self.env_batch)

    def load_weights(self, path):
        if path is None: return
        self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(path)))
        self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(path)))
        if self.loss_fcn == 'gan':
            load_gan(path)
        
    def save_model(self, path):
        self.actor.cpu()
        self.critic.cpu()
        torch.save(self.actor.state_dict(),'{}/actor.pkl'.format(path))
        torch.save(self.critic.state_dict(),'{}/critic.pkl'.format(path))
        if self.loss_fcn == 'gan':
            save_gan(path)
        self.choose_device()

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()
    
    def train(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()
    
    def choose_device(self):
        if not self.use_multiple_renderers:
            Decoder.to(device)
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)
