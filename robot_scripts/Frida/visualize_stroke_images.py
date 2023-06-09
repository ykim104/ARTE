import pickle
import numpy as np
import gzip 
import torch 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from paint_utils3 import show_img
from continuous_brush_model import train_param2stroke
from options import Options

#strokes = np.load('/tmp/extended_stroke_library_intensities.npy', allow_pickle=True)
#trajs = np.load('/tmp/extended_stroke_library_trajectories.npy', allow_pickle=True)

with gzip.GzipFile('../lite6_cache/0306/extended_stroke_library_intensities.npy','r') as f:
        strokes = np.load(f).astype(np.float32)/255.
trajectories = np.load('../lite6_cache/0306/extended_stroke_library_trajectories.npy', allow_pickle=True, encoding='bytes') 

opt = Options()
opt.gather_options()

from paint_utils3 import create_tensorboard
opt.writer = create_tensorboard()

train_param2stroke(opt,strokes=strokes,trajectories=trajectories)