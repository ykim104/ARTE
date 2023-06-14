import torch 
import numpy as np
import gzip 

from continuous_brush_model import StrokeParametersToImage
from paint_utils3 import show_img

import sys
sys.path.append('../../')
from Renderer.model import *

device = torch.device('cpu')

# open files
with gzip.GzipFile('/home/yejinkim/Repos/roboart/Frida/lite6_cache/0306/extended_stroke_library_intensities.npy','r') as f:
    strokes = np.load(f).astype(np.float32)/255.
trajectories = np.load('/home/yejinkim/Repos/roboart/Frida/lite6_cache/0306/extended_stroke_library_trajectories.npy', allow_pickle=True, encoding='bytes') 

# Randomize
rand_ind = torch.randperm(strokes.shape[0])

for i in rand_ind:
    stroke = torch.tensor(strokes[i]).unsqueeze(0)
    trajectory = torch.tensor(trajectories[i]).unsqueeze(0)

    h, w = stroke[0].shape[0], stroke[0].shape[1] 
    hs, he = int(0.4*h), int(0.75*h)  #changes the height        
    ws, we = int(0.45*w), int(0.6*w) #changes the width of show_img plot

    stroke = stroke[:, hs:he, ws:we]

    # from Frida - without transformation 
    h, w = stroke[0].shape[0], stroke[0].shape[1] 
    model_file = '/home/yejinkim/Repos/roboart/Frida/src/cache/param2img1.pt'
    param2img = StrokeParametersToImage(h,w)
    param2img.load_state_dict(torch.load(model_file))
    param2img.eval()
    param2img.to(device)

    print(trajectory)
    pred_stroke_frida = param2img(trajectory.float().to(device))


    # ddpg model - with transformation
    trajectory = trajectory[0]
    stroke_length = trajectory[12]
    stroke_bend = trajectory[5]
    stroke_z = trajectory[6]
    stroke_alpha = 0
    params = torch.zeros((1,7))
    params[0][0] = (stroke_length-10)/(50.0-10.0)# stroke lengt  
    params[0][1] = (stroke_bend+20) / 40.0 # stroke bend
    params[0][2] = stroke_z  # stroke z
    params[0][3] = stroke_alpha # stroke alpha
    params[0][4] = 0 # a
    params[0][5] = 0.5 #(0.5+1)/2.0 # x
    params[0][6] = 0.5 #(0.5+1)/2.0 # y
    Decoder = FCN(input_size=7).to(device)
    Decoder.load_state_dict(torch.load('../../models/renderers/FRIDA_lite6_renderer_0318.pkl')) 
    pred_stroke_fcn = 1 - Decoder(params)


    import torchvision.transforms as T
    rs = T.Resize((1024,1024))
    pred_stroke_fcn = rs(pred_stroke_fcn)

    show_img(255-stroke)
    show_img(255-pred_stroke_frida.cpu().detach())
    show_img(255-pred_stroke_fcn.cpu().detach())
