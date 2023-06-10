import datetime
import sys
import os
import cv2
import torch
import argparse 

from painter import Painter
from options import Options
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


parser = argparse.ArgumentParser(description='Learning to Paint')
parser.add_argument('--use_cache', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--max_step', default=10, type=int, help='max length for episode')
parser.add_argument('--actor', default='./../../models/actor_mnist.pkl', type=str, help='Actor model')
parser.add_argument('--renderer', default='./../../models/renderers/FRIDA_lite6_renderer_0318.pkl', type=str, help='renderer model')
parser.add_argument('--img', default='./../../images/flower.jpg', type=str, help='test image')
parser.add_argument('--imgid', default=0, type=int, help='set begin number for generated image')
parser.add_argument('--divide', default=1, type=int, help='divide the target image to get better resolution')
args = parser.parse_args()

canvas_cnt = args.divide * args.divide



date_and_time = datetime.datetime.now()
run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
writer = TensorBoard('{}/{}'.format(opt.tensorboard_dir, run_name))

opt = Options()
opt.use_cache = args.use_cache
opt.dont_retrain_stroke_model = True 
opt.gather_options()

"""
Get Target Imge (or target text for the future)
"""
img = cv2.imread(args.img, cv2.IMREAD_COLOR)
img_shape = (img.shape[1], img.shape[0])

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


os.system('mkdir output')
strokes_without_getting_new_paint = 999 
strokes_without_cleaning = 9999
painter.to_neutral()

canvas = painter.camera.get_canvas()
canvas = cv2.resize(canvas,(width, width))

sim_canvas = None
with torch.no_grad():
    if args.divide == 1:
        for i in range(args.max_step):
                # get simulated canvas
                stepnum = T * i / args.max_step 

                canvas = torch.tensor(canvas)
                canvas = torch.permute(canvas,(2,0,1))
                canvas = canvas.unsqueeze(0).to(device).float()/255.
                               
                if sim_canvas is not None & opt.debug:
                    print("using simulated canvas")
                    state = torch.cat([sim_canvas, target_img, stepnum, coord],1) # 3,3,1,2
                    actions = actor(state)                
                    sim_canvas, res = decode(actions, sim_canvas, brush_color="black")
                
                else:
                    state = torch.cat([canvas, target_img, stepnum, coord],1) # 3,3,1,2
                    actions = actor(state)                
                    sim_canvas, res = decode(actions, canvas, brush_color="black")
                
                '''
                # paint strokes
                actions = actions.cpu().detach().numpy()
                for i in range(n_strokes):
    
                    stroke_length = actions[i][0]
                    bend = actions[i][1]
                    z = actions[i][2]
                    alpha = actions[i][3]
                    stroke = simple_parameterization_to_real(stroke_length, bend, z, alpha=0)

                    rotation = actions[i][4] # radians                    
                    x = actions[i][5]
                    y = actions[i][6]
                    x, y, _ = canvas_to_global_coordinates(x,y,None,painter.opt)
                    
                    color_r = actions[i][7]
                    color_g = actions[i][8]
                    color_b = actions[i][9]
                    
                    #get color
                    # TODO: black:0, nothing else is set
                    paint_index = 0 
                    painter.to_neutral()

                    if strokes_without_cleaning >= 12:
                        #self.clean_paint_brush()
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
                '''

                canvas = painter.camera.get_canvas()
                canvas = cv2.resize(canvas,(width, width))

                sim_canvas_copy = sim_canvas.cpu().detach().numpy()[0].transpose((1,2,0))
                sim_canvas_copy *= 255 # or any coefficient
                sim_canvas_copy = sim_canvas_copy.astype(np.uint8)
                
                all_canvases = cv2.hconcat((sim_canvas_copy, canvas, img))
                show_img(all_canvases, title="In Progress...")


canvas_after = painter.camera.get_canvas()
show_img(canvas_after/255.0, title="Painting complete.")
