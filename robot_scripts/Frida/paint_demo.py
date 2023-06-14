import datetime
import sys
import os
import cv2
import torch
import argparse 
import threading
import time

from painter import Painter
from my_tensorboard import TensorBoard
from strokes import simple_parameterization_to_real #*
from paint_utils import canvas_to_global_coordinates, show_img #*
from options import Options

sys.path.append(os.path.abspath("./../.."))
from DRL.actor import *
from Renderer.model import FCN



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


width, height = 128, 128
T = torch.ones([1, 1, width, width], dtype=torch.float32).to(device)
ksize = (4,4)

class RobotPainter():
    def __init__(self, target, opt):
        self.canvas_img = None 
        self.canvas_orig_img = None
        self.sim_canvas = None 
        self._sim_canvas_img = None
        self.get_target(target)

        self.n_strokes = 5
        self.actor = ResNet(9,18,10*self.n_strokes)
        self.actor.load_state_dict(torch.load(opt.actor))
        self.actor.to(device).eval()
        
        self.renderer = FCN(input_size=7)
        self.renderer.load_state_dict(torch.load(opt.renderer))
        self.renderer.to(device).eval()

        self.strokes_without_cleaning = 0
        self.strokes_without_getting_new_paint = 4
        self.painter = Painter(opt, robot="lite6", use_cache=opt.use_cache, writer=opt.writer) 
        time.sleep(2.5)
        
        # launch a listener thread to stream camera 
        self.thread1 = threading.Thread(target=self.run)
        self.thread1.daemon = True 
        self.thread1.start()

        # if not simulate
        self.thread2 = threading.Thread(target=self.paint)
        self.thread2.daemon = True 
        self.thread2.start()

    def get_target(self, target):
        debug = True
        if debug:
            import torchvision
            import torchvision.datasets as datasets
            mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
            img = mnist_testset[4][0]
            img = img.convert('RGB')
            img = 255-np.asarray(img)
            self._img = self.preprocess_target_img(img)
            self.target_img = self.target_tensor_img(self._img)
            return        

        for k, v in target.items():
            if k == "image_path":
                self.img = cv2.imread(v, cv2.IMREAD_COLOR)
                self.target_img = self.preprocess_target_img(self.img)
            elif k == "text_path":
                self.text = ""
                #self.target_text = # token
            else:
                print("inappropriate target input.")

    def preprocess_target_img(self, img):
        # pad
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

        # reshize & tensorize
        img = cv2.resize(img, (width, width))
        return img 
    
    def target_tensor_img(self, img):
        target_img = img.reshape(1, width, width, 3) 
        target_img = np.transpose(target_img, (0, 3, 1, 2))
        target_img = torch.tensor(target_img).to(device).float() / 255.
        return target_img


    def decode(self, x, canvas, brush_color="color"): # b * (10 + 3)
        x = x.view(-1, 7 + 3)
        stroke = 1 - self.renderer(x[:, :7])
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
        stroke = stroke.view(-1, self.n_strokes, 1, width, width)
        color_stroke = color_stroke.view(-1, self.n_strokes, 3, width, width)
        
        res = []
        for i in range(self.n_strokes):
            canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
            res.append(canvas)
        return canvas, res


    #@staticmethod
    #def display_controls():
    def paint(self):
        for i in range(40):
            stepnum = T * i / args.max_steps 
            self.paint_stroke(stepnum.to(device), coord, simulate=False)
            print("Paint Step ", i)


    def paint_stroke(self, stepnum, coord, simulate=True):
        #stepnum = T * self.counter / args.max_step 
    
        if simulate:
            #if self.sim_canvas is None and self.canvas is None:
            #    self.sim_canvas = torch.ones((3,width,width))   
            if self.sim_canvas is None:
                self.canvas_img = self.process_canvas(self.canvas_orig)
                canvas = cv2.blur(self.canvas_img, ksize)
                canvas = torch.tensor(canvas)
                canvas = torch.permute(canvas,(2,0,1))
                canvas = canvas.unsqueeze(0).to(device).float()/255.
                self.sim_canvas = canvas
                   
            state = torch.cat([self.sim_canvas, self.target_img, stepnum, coord],1) # 3,3,1,2
        else:
            # move robot to zero joints first
            self.painter.robot.zero_joints()
            time.sleep(2)

            # down resolute 
            self.canvas_img = self.process_canvas(self.canvas_orig)
            canvas = cv2.blur(self.canvas_img, ksize)
            
            # tensor
            canvas = torch.tensor(canvas)
            canvas = torch.permute(canvas,(2,0,1))
            canvas = canvas.unsqueeze(0).to(device).float()/255.
            state = torch.cat([canvas, self.target_img, stepnum, coord],1) # 3,3,1,2
        actions = self.actor(state)

        if simulate:
            self.sim_canvas, res = self.decode(actions, self.sim_canvas, brush_color="black")    
        else:
            self.sim_canvas, res = self.decode(actions, canvas, brush_color="green")
        
        sim_canvas = self.sim_canvas.cpu().detach().numpy()[0].transpose((1,2,0))
        sim_canvas *= 255 # or any coefficient
        sim_canvas = sim_canvas.astype(np.uint8)
        self._sim_canvas_img = sim_canvas
        
        if not simulate:
            actions = actions.cpu().detach().numpy()[0]
            self.paint_actions(actions)


    def paint_actions(self, actions):
        for i in range(self.n_strokes):
            stroke_length = actions[i*10+0]*40+10
            bend = actions[i*10+1]*20-10 #40-20 # -20 ~ 20
            z = actions[i*10+2]
            alpha = actions[i*10+3]
            stroke = simple_parameterization_to_real(stroke_length, bend, z, alpha=0)
            
            rotation = (actions[i*10+4]*2-1)*3.14 # radians                    
            y = actions[i*10+5]+0.05#*2-1 # -1 ~ 1 ->> 
            x = actions[i*10+6]-0.05#*2-1
            if x>=1 or y>=1 or x<=0 or y<=0:
                continue
            
            # canvas coord are proportions from bottom left
            x, y, _ = canvas_to_global_coordinates(x,y,None,self.painter.opt,robot="lite6")
            y *= -1

            color_r = actions[i*10+7]
            color_g = actions[i*10+8]
            color_b = actions[i*10+9]
            
            # paint
            self.painter.to_neutral()
            if self.strokes_without_cleaning >= 12:
                self.painter.clean_paint_brush()
                self.painter.get_paint(0)
                self.strokes_without_cleaning, self.strokes_without_getting_new_paint = 0, 0
            if self.strokes_without_getting_new_paint >= 4:
                self.painter.get_paint(0)
                self.strokes_without_getting_new_paint = 0
            self.strokes_without_getting_new_paint += 1
            self.strokes_without_cleaning += 1
            self.painter.robot.reset()

            #paint
            stroke.paint(self.painter, x, y, rotation, wait=True)
    

    def run(self):
        while True:
            #print("Get Canvas")
            canvas_orig_img = self.painter.camera.get_canvas()
            canvas_orig_img = cv2.resize(canvas_orig_img, (width, width))
            self.canvas_orig_img = cv2.cvtColor(canvas_orig_img, cv2.COLOR_BGR2RGB)

    @property
    def canvas_orig(self):
        return self.canvas_orig_img

    def process_canvas(self, canvas):
        gray = cv2.cvtColor(canvas,cv2.COLOR_RGB2GRAY)
        _,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
        thresh = ~thresh
        
        canvas = cv2.bitwise_and(canvas, canvas, mask =thresh)
        canvas[np.where((canvas==[0,0,0]).all(axis=2))] = [255,255,255]
        canvas = cv2.resize(canvas,(width, width))
        return canvas
    
    @property
    def canvas(self):
        #while '1' == input("Get Canvas Photo. Press 1 to reset the robot."):
        return self.canvas_img
        
    #def all_canvases(self):
    #    all_canvases = cv2.hconcat((sim_canvas_copy, self.img))
        #


date_and_time = datetime.datetime.now()
run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")

if __name__ == "__main__":
    coord = torch.zeros([1, 2, width, width])
    for i in range(width):
        for j in range(width):
            coord[0, 0, i, j] = i / (width-1.)
            coord[0, 1, i, j] = j / (width-1.)
    coord = coord.to(device)

    parser = argparse.ArgumentParser(description='Paint Demo')
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--debug', default=False, type=bool)
    parser.add_argument('--max_steps', default=40, type=int, help='max length for episode')
    parser.add_argument('--actor', default='./../../models/actor.pkl', type=str, help='Actor model')
    parser.add_argument('--renderer', default='./../../models/renderers/FRIDA_lite6_renderer_0318.pkl', type=str, help='renderer model')
    parser.add_argument('--img', default='test_img.jpg', type=str, help='test image')
    parser.add_argument('--imgid', default=0, type=int, help='set begin number for generated image')
    parser.add_argument('--divide', default=1, type=int, help='divide the target image to get better resolution')
    args = parser.parse_args()

    opt = Options()    
    opt.actor = args.actor
    opt.renderer = args.renderer
    opt.debug = args.debug
    opt.use_cache = args.use_cache
    opt.dont_retrain_stroke_model = True 
    opt.gather_options()
    opt.writer = TensorBoard('{}/{}'.format(opt.tensorboard_dir, run_name))
    

    target = dict()
    target["image_path"] = args.img
    

    robot_painter = RobotPainter(target, opt)
    
    while True:
        co = robot_painter.canvas_orig
        c = robot_painter.canvas
        sim = robot_painter._sim_canvas_img
        tim = robot_painter._img

        if sim is None:
            sim = c 

        if c is not None and co is not None and sim is not None:
            cv2.imshow("stream", cv2.hconcat((sim, c, co, tim)))
            if cv2.waitKey(1) & 0xFF == ord('q'): # wait for 1 millisecond
                break

    cv2.destroyAllWindows()
