import datetime
import sys
import os
import cv2
import torch
import argparse 

from painter import Painter
from my_tensorboard import TensorBoard
from strokes import simple_parameterization_to_real #*
from paint_utils import canvas_to_global_coordinates, show_img #*
from paint_inference import decode 



date_and_time = datetime.datetime.now()
run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
writer = TensorBoard('{}/{}'.format(opt.tensorboard_dir, run_name))

width, height = 128, 128

class RobotPainter():
    def __init__(self, target, opt):
        print()
        self.canvas = None 
        self.sim_canvas = None 
        self.n_strokes = 5

        self.painter = Painter(opt, robot="lite6", use_cache=opt.use_cache, writer=writer) 

        self.actor = ResNet(9,18,10*n_strokes)
        self.actor.load_state_dict(torch.load(opt.actor))
        self.actor.to(device).eval()
        
        self.renderer = FCN(input_size=7)
        self.renderer.load_state_dict(torch.load(opt.renderer))
        self.renderer.to(device).eval()

        # launch a listener thread to stream camera 
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True 
        self.thread.start()


    #@staticmethod
    #def display_controls():
    def paint(self, stepnum, coord, simulate=True):
        #stepnum = T * self.counter / args.max_step 
    
        if simulate:
            if self.sim_canvas is None and self.canvas is None:
                self.sim_canvas = torch.ones((3,width,width))   
            elif self.sim_canvas is None:
                self.sim_canvas = self.canvas
            
            state = torch.cat([self.sim_canvas, self.target_img, stepnum, coord],1) # 3,3,1,2
        else:
            canvas = cv2.blur(self.canvas, ksize)
            canvas_orig = self.canvas_orig

            canvas = torch.tensor(canvas)
            canvas = torch.permute(canvas,(2,0,1))
            canvas = canvas.unsqueeze(0).to(device).float()/255.
            state = torch.cat([canvas, self.target_img, stepnum, coord],1) # 3,3,1,2
        actions = actor(state)

        if simulate:
            self.sim_canvas, res = decode(actions, self.sim_canvas, brush_color="black")    
        else:
            self.sim_canvas, res = decode(actions, canvas, brush_color="green")
            actions = actions.cpu().detach().numpy()[0]
            self.paint_actions(actions)

        sim_canvas_copy = self.sim_canvas.cpu().detach().numpy()[0].transpose((1,2,0))
        sim_canvas_copy *= 255 # or any coefficient
        sim_canvas_copy = sim_canvas_copy.astype(np.uint8)
        all_canvases = cv2.hconcat((sim_canvas_copy, self.target_img))
        #show_img(all_canvases, title="In Progress...")

        return _canvases
    
    def paint_actions(self, actions)
        for i in range(self.n_strokes):
            stroke_length = actions[i*10+0]*40+10
            bend = actions[i*10+1]*20-10 #40-20 # -20 ~ 20
            z = actions[i*10+2]
            alpha = actions[i*10+3]
            stroke = self.simple_parameterization_to_real(stroke_length, bend, z, alpha=0)
            
            rotation = (actions[i*10+4]*2-1)*3.14 # radians                    
            y = actions[i*10+5]+0.05#*2-1 # -1 ~ 1 ->> 
            x = actions[i*10+6]-0.05#*2-1
            if x>=1 or y>=1 or x<=0 or y<=0:
                continue
            
            # canvas coord are proportions from bottom left
            x, y, _ = canvas_to_global_coordinates(x,y,None,painter.opt,robot="lite6")
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
        cv2.namedWindow("stream", cv2.WINDOW_NORMAL)
        while True:
            self.canvas_orig = painter.camera.get_canvas()
            cv2.imshow('stream', self.canvas_orig)
            cv2.waitKey(1)
        cv2.destroyAllWindows()

    @property
    def canvas(self):
        while '1' == input("Get Canvas Photo. Press 1 to reset the robot."):
            painter.robot.zero_joints()

        canvas_orig = self.canvas_orig
        gray = cv2.cvtColor(canvas_orig,cv2.COLOR_RGB2GRAY)
        _,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
        thresh = ~thresh
        
        canvas = cv2.bitwise_and(canvas_orig, canvas_orig, mask =thresh)
        canvas[np.where((canvas==[0,0,0]).all(axis=2))] = [255,255,255]
        canvas = cv2.resize(canvas,(width, width))
        return self.canvas = canvas
    

if __name__ == "__main__":
    robot_painter = RobotPainter()
    for i in range(args.max_steps):
        _canvases = robot_painter.paint(i, coord)
        cv2.imshow("Actions", _canvases)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
