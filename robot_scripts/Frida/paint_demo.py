import datetime
import sys
import os
import cv2
import torch
import argparse 

from painter import Painter
from my_tensorboard import TensorBoard
from strokes import *
from paint_utils import canvas_to_global_coordinates, show_img #*

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

width = 128
T = torch.ones([1, 1, width, width], dtype=torch.float32).to(device)
coord = torch.zeros([1, 2, width, width])
for i in range(width):
    for j in range(width):
        coord[0, 0, i, j] = i / (width-1.)
        coord[0, 1, i, j] = j / (width-1.)
coord = coord.to(device)

use_sim_only = False
parser = argparse.ArgumentParser(description='Learning to Paint')
parser.add_argument('--use_cache', action='store_true')
parser.add_argument('--debug', default=False, type=bool)
parser.add_argument('--max_step', default=40, type=int, help='max length for episode')
parser.add_argument('--actor', default='./../../models/actor.pkl', type=str, help='Actor model')
parser.add_argument('--renderer', default='./../../models/renderers/FRIDA_lite6_renderer_0318.pkl', type=str, help='renderer model')
parser.add_argument('--img', default='./../../images/flower.jpg', type=str, help='test image')
parser.add_argument('--imgid', default=0, type=int, help='set begin number for generated image')
parser.add_argument('--divide', default=1, type=int, help='divide the target image to get better resolution')
args = parser.parse_args()

canvas_cnt = args.divide * args.divide



date_and_time = datetime.datetime.now()
run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
writer = TensorBoard('{}/{}'.format(opt.tensorboard_dir, run_name))

from options import Options
opt = Options()
opt.debug = args.debug
opt.use_cache = args.use_cache
opt.dont_retrain_stroke_model = True 
opt.gather_options()

"""
Get Target Imge (or target text for the future)
"""
debug = True
if debug:
    import torchvision
    import torchvision.datasets as datasets
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
    img = mnist_testset[4][0]
    img = img.convert('RGB')
    img = 255-np.asarray(img)
            
    #print(img.shape)
else:
    img = cv2.imread(args.img, cv2.IMREAD_COLOR)


# pad the borders
img_shape = (img.shape[1], img.shape[0]) # w, h
pad_w = int(img_shape[0]*0.1)
pad_h = int(img_shape[1]*0.1)
img = cv2.copyMakeBorder(
                 img, 
                 pad_h,
                 pad_h,
                 pad_w,
                 pad_w,
                 cv2.BORDER_CONSTANT, 
                 value=[255,255,255]
              )


img = cv2.resize(img, (width, width))
target_img = img.reshape(1, width, width, 3) 
target_img = np.transpose(target_img, (0, 3, 1, 2))
target_img = torch.tensor(target_img).to(device).float() / 255.

objective_data = target_img # target_text


"""
Learned Agents
"""
#sys.path.append("./../..")
sys.path.append(os.path.abspath("./../.."))
from DRL.actor import *
from Renderer.model import FCN

n_strokes = 5
actor = ResNet(9,18,10*n_strokes) # state_size, 18, 10*n_strokes
actor.load_state_dict(torch.load(args.actor))
actor.to(device).eval()

sim_renderer = FCN(input_size=7)
sim_renderer.load_state_dict(torch.load(args.renderer))
sim_renderer.to(device).eval()


def decode(x, canvas, brush_color="color"): # b * (10 + 3)
    x = x.view(-1, 7 + 3)
    stroke = 1 - sim_renderer(x[:, :7])
    stroke = stroke.view(-1, width, width, 1)
    
    if brush_color=="color":
        color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
    elif brush_color=="white":
        color_stroke = stroke * torch.tensor([1, 1, 1]).view(-1,1,1,3).to(device) #white strokes
    elif brush_color=="black":
        color_stroke = stroke * torch.tensor([0, 0, 0]).view(-1,1,1,3).to(device) #white strokes
    elif brush_color=="green":
        color_stroke = stroke * torch.tensor([0, 1, 0]).view(-1,1,1,3).to(device) #white strokes
            
    stroke = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    stroke = stroke.view(-1, n_strokes, 1, width, width)
    color_stroke = color_stroke.view(-1, n_strokes, 3, width, width)
    
    res = []
    for i in range(n_strokes):
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
        res.append(canvas)
    return canvas, res



"""
Painter object that holds fnxs for
- camera
- robot
- writer
- canvas to global coordinates
- camera robot calibration
- brush heights
- paint/rug/water positions
- cache 
- param2img.pt (model that I won't be using. For FRIDA paper)
"""
painter = Painter(opt, robot="lite6", use_cache=True, writer=writer) 
print("Painter loaded.")


# get canvas
#canvas = torch.zeros([1, 3, width, width]).to(device)
#patch_img = cv2.resize(img, (width * args.divide, width * args.divide))
#patch_img = large2small(patch_img)
#patch_img = np.transpose(patch_img, (0, 3, 1, 2))
#patch_img = torch.tensor(patch_img).to(device).float() / 255.


canvas_before = painter.camera.get_canvas()
show_img(canvas_before/255.0, title="Ready to start painting.")
#writer.add_

#while True:
#    canvas = painter.camera.get_canvas()
#    cv2.imshow("Canvas", canvas)
#    cv2.waitKey(1)


os.system('mkdir output')
strokes_without_getting_new_paint = 4
strokes_without_cleaning = 0
painter.to_neutral()

# visualize masked version
ksize = (4, 4)
canvas = painter.camera.get_canvas()
gray = cv2.cvtColor(canvas,cv2.COLOR_RGB2GRAY)
_,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
thresh = ~thresh
canvas = cv2.bitwise_and(canvas, canvas, mask =thresh)
canvas[np.where((canvas==[0,0,0]).all(axis=2))] = [255,255,255]
canvas = cv2.resize(canvas,(width, width))




sim_canvas = None
with torch.no_grad():
    if args.divide == 1:
        for i in range(args.max_step):
                # get simulated canvas
                stepnum = T * i / args.max_step 

                canvas = cv2.blur(canvas, ksize)
                canvas = torch.tensor(canvas)
                canvas = torch.permute(canvas,(2,0,1))
                canvas = canvas.unsqueeze(0).to(device).float()/255.
                               
                if sim_canvas is not None and use_sim_only:
                    print("using simulated canvas")
                    state = torch.cat([sim_canvas, target_img, stepnum, coord],1) # 3,3,1,2
                    actions = actor(state)                
                    sim_canvas, res = decode(actions, sim_canvas, brush_color="black")
                
                else:
                    state = torch.cat([canvas, target_img, stepnum, coord],1) # 3,3,1,2
                    actions = actor(state)                
                    sim_canvas, res = decode(actions, canvas, brush_color="green")
                
                
                # paint strokes
                actions = actions.cpu().detach().numpy()
                print(actions)

                actions = actions[0]
                for i in range(n_strokes):
                    
                    stroke_length = actions[i*10+0]*40+10
                    bend = actions[i*10+1]*20-10 #40-20 # -20 ~ 20
                    z = actions[i*10+2]
                    alpha = actions[i*10+3]
                    stroke = simple_parameterization_to_real(stroke_length, bend, z, alpha=0)
                    print("stroke parmas ", stroke_length, bend, z)

                    print(actions[i*10+4], actions[i*10+5], actions[i*10+6])
                    rotation = (actions[i*10+4]*2-1)*3.14 # radians                    
                    y = actions[i*10+5]+0.05#*2-1 # -1 ~ 1 ->> 
                    x = actions[i*10+6]-0.05#*2-1

                    if x>=1 or y>=1 or x<=0 or y<=0:
                        continue
                    #rigid
                    # canvas coord are proportions from bottom left
                    x, y, _ = canvas_to_global_coordinates(x,y,None,painter.opt,robot="lite6")
                    y *= -1
                    print("transformation ", rotation, x, y)

                    color_r = actions[i*10+7]
                    color_g = actions[i*10+8]
                    color_b = actions[i*10+9]
                    
                    if use_sim_only is False:
                        #get color
                        # TODO: black:0, nothing else is set
                        paint_index = 0 
                        painter.to_neutral()

                        if strokes_without_cleaning >= 12:
                            painter.clean_paint_brush()
                            painter.get_paint(0)
                            strokes_without_cleaning, strokes_without_getting_new_paint = 0, 0
                        if strokes_without_getting_new_paint >= 4:
                            painter.get_paint(0)
                            strokes_without_getting_new_paint = 0
                        strokes_without_getting_new_paint += 1
                        strokes_without_cleaning += 1
                        painter.robot.reset()

                        #paint
                        stroke.paint(painter, x, y, rotation, wait=True)
                
                if use_sim_only is False:
                    painter.robot.zero_joints()
                    while '1' == input("Get Canvas Photo. Press 1 to reset the robot."):
                        painter.robot.zero_joints()

                canvas = painter.camera.get_canvas()

                #visuialize masked version
                gray = cv2.cvtColor(canvas,cv2.COLOR_RGB2GRAY)
                _,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
                thresh = ~thresh
                canvas = cv2.bitwise_and(canvas, canvas, mask =thresh)
                canvas[np.where((canvas==[0,0,0]).all(axis=2))] = [255,255,255]

                
                canvas = cv2.resize(canvas,(width, width))
                
                sim_canvas_copy = sim_canvas.cpu().detach().numpy()[0].transpose((1,2,0))
                sim_canvas_copy *= 255 # or any coefficient
                sim_canvas_copy = sim_canvas_copy.astype(np.uint8)
                
                all_canvases = cv2.hconcat((sim_canvas_copy, canvas, img))
                show_img(all_canvases, title="In Progress...")


canvas_after = painter.camera.get_canvas()
show_img(canvas_after/255.0, title="Painting complete.")
