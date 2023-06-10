# 
#
import CLIP_.clip as clip
from utils.sketch_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np
from skimage.color import rgb2gray
from scipy.ndimage.filters import gaussian_filter
from skimage.filters import threshold_otsu
import cv2


model, preprocess = clip.load("ViT-B/32", "cuda", jit=False)
model.eval().to("cuda")
        

class SaliencyMap:
    def __init__(self, target_im, n_strokes, opt=None, plot=False):
        self.opt = opt
        self.inds = None
        self.n_strokes = n_strokes
        self.inputs = target_im  #(B,C,H,W)
        self.canvas_width = target_im.shape[2]
        self.canvas_height = target_im.shape[3]

        self.define_attention_input(self.inputs)
        self.set_attention_map()
        self.set_attention_threshold_map()
        if plot:
            self.plot_attention()


    def define_attention_input(self, target_im):
        data_transforms = transforms.Compose([
#                    T.ToPILImage(),
                    preprocess.transforms[-1],
                    T.Resize((224,224))
                ])
        self.image_input_attn_clip = data_transforms(target_im).to("cuda")
        #print("attn input: ", self.image_input_attn_clip.shape)


    def plot_attention(self):
        #self.opt.use_wandb = False 
        #self.opt.output_dir = ""
        plot_atten(self.attention_map, self.thresh_map, self.image_input_attn_clip, self.inds,
                            False, "{}.jpg".format(
                            "attention_map"))


    def softmax(self, x, tau=0.2):
        e_x = np.exp(x / tau)
        return e_x / e_x.sum() 


    def set_attention_threshold_map(self):
        attn_map = (self.attention_map - self.attention_map.min()) / (self.attention_map.max() - self.attention_map.min())
        
        if True: #$self.xdog_intersec:
            xdog = XDoG_()
            im_xdog = xdog(self.image_input_attn_clip[0].permute(1,2,0).cpu().numpy(), k=10)
            intersec_map = (1 - im_xdog) * attn_map
            attn_map = intersec_map

        attn_map_soft = np.copy(attn_map)
        attn_map_soft[attn_map > 0] = self.softmax(attn_map[attn_map > 0], tau=0.5)
        self.thresh_map = attn_map_soft

        # sample random starting point
        k = self.n_strokes #1 * 4 # self.num_stages * self.num_paths
        

        self.inds = np.random.choice(range(attn_map.flatten().shape[0]), size=k, replace=True, p=attn_map_soft.flatten())
        self.inds = np.array(np.unravel_index(self.inds, attn_map.shape)).T
        
        self.inds_normalized = np.zeros(self.inds.shape)
        self.inds_normalized[:, 0] =  self.inds[:, 1] * (1/224)
        self.inds_normalized[:, 1] =  self.inds[:, 0] * (1/224)
        self.inds_normalized = self.inds_normalized.tolist()



    def set_attention_map(self):
        text_input = clip.tokenize(["none"]).to("cuda")
        self.attention_map = interpret(self.image_input_attn_clip, text_input, model, device="cuda")
        



#https://github.com/yael-vinker/CLIPasso/blob/main/models/painter_params.py

class XDoG_(object):
    def __init__(self):
        super(XDoG_, self).__init__()
        self.gamma=0.98
        self.phi=200
        self.eps=-0.1
        self.sigma=0.8
        self.binarize=True
        
    def __call__(self, im, k=10):
        if im.shape[2] == 3:
            im = rgb2gray(im)
        imf1 = gaussian_filter(im, self.sigma)
        imf2 = gaussian_filter(im, self.sigma * k)
        imdiff = imf1 - self.gamma * imf2
        imdiff = (imdiff < self.eps) * 1.0  + (imdiff >= self.eps) * (1.0 + np.tanh(self.phi * imdiff))
        imdiff -= imdiff.min()
        imdiff /= imdiff.max()
        if self.binarize:
            th = threshold_otsu(imdiff)
            imdiff = imdiff >= th
        imdiff = imdiff.astype('float32')
        return imdiff


def interpret(image, texts, model, device):
    images = image.repeat(1, 1, 1, 1)
    res = model.encode_image(images)
    model.zero_grad()
    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(1, num_tokens, num_tokens)
    cams = [] # there are 12 attention blocks
    for i, blk in enumerate(image_attn_blocks):
        cam = blk.attn_probs.detach() #attn_probs shape is 12, 50, 50
        # each patch is 7x7 so we have 49 pixels + 1 for positional encoding
        cam = cam.reshape(1, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0)
        cam = cam.clamp(min=0).mean(dim=1) # mean of the 12 something
        cams.append(cam)  
        R = R + torch.bmm(cam, R)
              
    cams_avg = torch.cat(cams) # 12, 50, 50
    cams_avg = cams_avg[:, 0, 1:] # 12, 1, 49
    image_relevance = cams_avg.mean(dim=0).unsqueeze(0)
    image_relevance = image_relevance.reshape(1, 1, 7, 7)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bicubic')
    image_relevance = image_relevance.reshape(224, 224).data.cpu().numpy().astype(np.float32)
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    return image_relevance