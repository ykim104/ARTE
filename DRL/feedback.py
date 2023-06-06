import cv2
import numpy as np
import math

from DRL.ddpg import decode
import torch

# Mouse callback function
global click_list
positions, click_list = [], []
def callback(event, x, y, flags, param):
    if event == 1: 
        click_list.append((x,y))
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image', callback)


class HumanFeedback:
    def __init___(self, opt):
        self.type = opt.feedback_type

    #@staticmethod
    def select_new_action(prev_action, prev_observation, next_observation=None):
        #print(prev_action.shape, prev_action[0])
        batch_size = prev_action.shape[0]
        n_brush = int(prev_action.shape[1]/10)
        print("BatchSize: ", batch_size, ", N BrushStrokes", n_brush)

        print("Press 'esc' to move on. Press any other button to re-make a stroke.")
        corrected_action = torch.zeros((prev_action.shape))
        for b in range(batch_size):            
            #show image between prev_observation, next_observation, gt
            gt = prev_observation[b, 3 : 6].float().cpu().detach().numpy().transpose(1, 2, 0) / 255
            prev_canvas = prev_observation[b, :3].float().cpu().detach().numpy().transpose(1, 2, 0) / 255
            #next_canvas = next_observation[b, :3].float().cpu().detach().numpy().transpose(1, 2, 0) / 255
            prev_canvas_torch = prev_observation[b, :3].float() / 255
            
            #cv2.imshow("gt/prev/next", cv2.hconcat([gt, prev_canvas, next_canvas]))
            #k = cv2.waitKey(0)
            #if k == 27:
            #    cv2.destroyWindow("gt/prev/next") # if user presses 'esc'

            #print(click_list)
            #user_action = torch.zeros(corrected_action[b].unsqueeze(0).shape)
            #print("user action shape: ", user_action.shape)
            img = prev_canvas_torch.unsqueeze(0)
            
            for i in range(n_brush): #position in click_list:
                try_again = True
    
                while try_again:
                    print(b, i)

                    # init
                    _action = torch.zeros(10) #np.zeros(10)

                    # get images
                    #_img = decode(_action.unsqueeze(0).to("cuda:0"), img.to("cuda:0"))
                    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
                    cv2.setMouseCallback('image', callback)
                    cv2.imshow("image", cv2.hconcat([img[0].cpu().detach().numpy().transpose(1, 2, 0),gt]))

                    k = cv2.waitKey(0)
                    if k == 27:
                        cv2.destroyWindow("image")
                        #print(click_list)

                    #x1, y1 = click_list[i*2][0],click_list[i*2][1]
                    #x2, y2 = click_list[i*2+1][0],click_list[i*2+1][1]
                    #x1, y1 = click_list[0][0]-128,click_list[0][1]
                    #x2, y2 = click_list[1][0]-128,click_list[1][1]
                    #click_list = []
                    x1, y1 = click_list[-2][0]-128,click_list[-2][1]
                    x2, y2 = click_list[-1][0]-128,click_list[-1][1]
                    #print(x1,y1,x2,y2)


                    x_coord = x1/128.0
                    y_coord = y1/128.0
                    rotation = np.arctan2((y2-y1),(x2-x1)) #- 3.14/2 #radiance
                    
                    #Adjust rotation so that: 
                    #0, 0.5, 1 = horizontal, 0.25, 0.75 = vertical, 0.125 & 0.625 = /, 0.325 * 0.875 = \, 
                    rotation = np.pi - rotation # mirror
                    if rotation < 0 : # (convert to 0 ~ 2pi)
                        rotation += 2*np.pi 
                    rotation = rotation/(2*3.14) # normalize 0~1
                    #print(x_coord, y_coord, rotation)

                    # stroke length, bend, alpha, z(thickness), x coord, y coord, rotation , color (r/g/b)
                    stroke_length = 1.0 #float(input("stroke length: "))
                    stroke_bend = 0.5 #input("stroke bend: ")
                    stroke_z = 0.25 #input("stroke thickness: ")
                    stroke_alpha = 0.0 #input("stroke length")

                    # data
                    _action[0] = float(stroke_length)
                    _action[1] = float(stroke_bend)
                    _action[2] = float(stroke_z)
                    _action[3] = float(stroke_alpha) 
                    
                    _action[4] = rotation #float(input("rotation 0-1: ")) #rotation 
                    _action[5] = x_coord
                    _action[6] = y_coord
                    
                    _action[7] = 0.04#color_r 
                    _action[8] = 0.02#color_g 
                    _action[9] = 0.03#color_b 
                    
                    #print("actions: ", _action)
                    _img = decode(_action.unsqueeze(0).to("cuda:0"), img.to("cuda:0"))
                    cv2.imshow("new image. to redraw, press 'd'", cv2.hconcat([_img[0].cpu().detach().numpy().transpose(1, 2, 0),gt]))
                    k = cv2.waitKey(0)
                    if k == 27:
                        print("Moving on to the next brush stroke")
                        try_again = False  
                        corrected_action[b][i*10:i*10+10] = _action #.append(_action)
                        img = _img
                        cv2.destroyWindow("new image. to redraw, press 'd'")
                        break
                    else:
                        #user_input = input("to exit, press 'd': ") 
                        cv2.destroyWindow("new image. to redraw, press 'd'")
                            #break # if user presses 'esc'
                    
                    if try_again is False:
                        break

            # view updated observation
            new_observation = decode(corrected_action[b].unsqueeze(0).to("cuda:0"), prev_canvas_torch.to("cuda:0")) 
            cv2.imshow("updated image",new_observation[0].cpu().detach().numpy().transpose(1, 2, 0))
            k = cv2.waitKey(0)
            if k==27: 
                cv2.destroyWindow("updated image")
                continue

        print("Corrected Action:", corrected_action.shape)
        return corrected_action

    