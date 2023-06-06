# ARTE 

Yejin Kim and Dohyek Lee
<br />Sponsored by ZER01NE<br />

### TASK LIST
-[ ]
-[ ]


### 1. Introduction
#### 1.1  ARTE Skecher: Reinforcement Learning
#### 1.2. ARTE SKetcher: Objectives
- l2
- clip_conv_loss
- clip_fc_loss
- text



### 2. Run in Simulation
#### 2.1. Installation
`code`
    virtualenv venv -p python3.8
    pip install torch==1.13.1+cu117
`code`

#### 2.2 Train Brush Strokes Renderer
#### 2.3 Train ARTE Sketcher
`code`
    python train.py --loss_fcn clip
`code`
#### 2.4 Paint by Inferences




### 3. Run With Robot
#### 3.1 Installation
#### 3.2 Robot-Camera Calibration
#### 3.3 Collecting Brush Strokes Data
#### 3.4 Paint by Inferences



### References
1. Learning to Paint [https://github.com/megvii-research/ICCV2019-LearningToPaint]
2. *Content Masked Loss [https://github.com/pschaldenbrand/ContentMaskedLoss]
3. *CLIPPasso [https://github.com/yael-vinker/CLIPasso]
4. **FRIDA [https://github.com/cmubig/Frida/tree/master]<br />
**Heavily influenced Sim* <br />
***Heavily influenced the robot code*

### Acknowledgements
ZER01NE


