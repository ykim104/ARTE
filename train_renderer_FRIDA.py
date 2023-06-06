''' Adapted from https://github.com/hzwer/ICCV2019-LearningToPaint '''

import cv2
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T

from utils.tensorboard import TensorBoard
from Renderer.model import FCN, FCNFRIDA
from Renderer.stroke_gen import *
import os
import argparse

CONSTRAINT_BRUSH_WIDTH = 0.01
CONSTRAINT_OPACITY = 1.0
CONSTRAINT_MAX_STROKE_LENGTH = 0.3

parser = argparse.ArgumentParser(description='Train Neural Renderer')
parser.add_argument('--name', default='FRIDA_lite6_renderer_0318', type=str, help='Name the output renderer file. Leave off ".pkl"')
parser.add_argument('--data_path', default='../Frida/src/dataset', type=str, help='Path to the dataset')
parser.add_argument('--resume', action='store_true', help='Resume training from file name')
parser.add_argument('--batch_size', default=64, type=int, help='Batch Size')
    
args = parser.parse_args()
args.resume = True 

renderer_fn = args.name + '.pkl'

if not os.path.exists('train_log_renderer'): os.mkdir('train_log_renderer')
log_dir = os.path.join('train_log_renderer', args.name)
if not os.path.exists(log_dir): os.mkdir(log_dir)

writer = TensorBoard(log_dir)

criterion = nn.MSELoss()
#net = FCNFRIDA()
net = FCN(input_size=7)
optimizer = optim.Adam(net.parameters(), lr=3e-6)
batch_size = args.batch_size

use_cuda = torch.cuda.is_available()
step = 0


def save_model():
    if use_cuda:
        net.cpu()
    torch.save(net.state_dict(), renderer_fn)
    if use_cuda:
        net.cuda()


def load_weights():
    pretrained_dict = torch.load(renderer_fn)
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

RANDOM_GENERATION = True 

if RANDOM_GENERATION is False:
    dataset_params = np.load(os.path.join(args.data_path,"stroke_data_numpy.npy"))
    print("Loaded " + str(len(dataset_params)) + " images.")
    #print(dataset_params[0])

    from PIL import Image
    dataset_images = []
    for i in range(len(dataset_params)):
        img = cv2.imread(os.path.join(args.data_path,'strokes','stroke_'+str(i)+'.png')) #, cv2.IMREAD_UNCHANGED)
        #img = Image.open(os.path.join(args.data_path,'strokes','stroke_'+str(i)+'.png'))
        img = np.asarray(img)/255.
        img = np.transpose(img, (2,0,1))
        dataset_images.append(img)
else:
    import sys
    sys.path.append('/home/yejinkim/Repos/roboart/Frida/src')
    from testground import draw_random_stroke

if args.resume:
    load_weights()


# For the constrained model. Can't make the brush width tiny all at once. Need to decrease slowly.
dilation_rate = 25
dec_brush_width_int = 1000 # Every this number of steps, decrease the brush width until target
while step < 600000:    
    if(step%dec_brush_width_int==0):
        if dilation_rate>1:
            dilation_rate -= 1
    
    # load data
    train_batch = []
    ground_truth = torch.zeros([batch_size, 128, 128], dtype=torch.uint8)
    
    for i in range(batch_size):
        if RANDOM_GENERATION:
            param, image = draw_random_stroke() #0~1    
            train_batch.append(param[:7])
            print(param[:7])
            
            image = image[0].cpu().detach().data.numpy()
            image = np.transpose(image,(1,2,0))

            kernel = np.ones((dilation_rate, dilation_rate), np.uint8)
            img_dilation = cv2.dilate(image, kernel, iterations=1)
            cv2.imshow("image", cv2.hconcat([image, img_dilation]))
            key = cv2.waitKey(0) 


            ground_truth[i] = torch.tensor((1-img_dilation)*255)
        else:
            f = np.random.randint(len(dataset_images))
            train_batch.append(dataset_params[f])
            ground_truth.append(dataset_images[f])
        
    if RANDOM_GENERATION:
        train_batch = torch.tensor(train_batch).float()
        ground_truth = ground_truth.float()/255.
    else:    
        train_batch = torch.tensor(train_batch).float()
        ground_truth = torch.tensor(ground_truth).float()
    
    # train
    net.train()
    if use_cuda:
        net = net.cuda()
        train_batch = train_batch.cuda()
        ground_truth = ground_truth.cuda()
    gen = net(train_batch)
    optimizer.zero_grad()
    loss = criterion(gen, ground_truth)
    loss.backward()
    optimizer.step()
    print("Training...", step, loss.item())
    if step < 200000:
        lr = 1e-4
    elif step < 400000:
        lr = 1e-5
    else:
        lr = 1e-6

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    writer.add_scalar("train/loss", loss.item(), step)
    if step % 500 == 0:
        net.eval()
        gen = net(train_batch)
        loss = criterion(gen, ground_truth)
        writer.add_scalar("val/loss", loss.item(), step)
        for i in range(32):
            G = T.ToPILImage()(gen[i])
            GT = T.ToPILImage()(ground_truth[i]) #ground_truth[i].cpu().data.numpy() * 255               
            writer.add_image("train/img{}.png".format(i), G, step)
            writer.add_image("train/img{}_truth.png".format(i), GT, step)
    if step % 1000 == 0:
        save_model()
    step += 1
