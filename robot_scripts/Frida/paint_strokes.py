#! /usr/bin/env python
##########################################################
#################### Copyright 2022 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

import os
import time
import sys
import cv2
import numpy as np
import argparse
import datetime

from painter import Painter
from options import Options

from my_tensorboard import TensorBoard


if __name__ == '__main__':
    opt = Options()
    opt.gather_options()

    date_and_time = datetime.datetime.now()
    run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
    writer = TensorBoard('{}/{}'.format(opt.tensorboard_dir, run_name))
    writer.add_text('args', str(sys.argv), 0)

    painter = Painter(opt, robot="lite6", 
        use_cache=opt.use_cache, writer=writer)

    

