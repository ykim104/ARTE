# UFactory LITE6 

1. Hand-eye calibration (draw dots) 

2. generate brush stroke libraries. 2+ papers(how many strokes did I collect last time?) 

3. train brush strokes (FRIDA param2img) 

4. train renderer for DRL env (FRIDA param2img -> Decoder FCN)  

5. train policy/actor (DDPG) 

6. inference; paint with a robot



#### Starting

URL: <ip.add.re.ss>:18333


#### Robot-Camera Calibration

This script below will calculate robot-camera calibration (extrinsic calibration) and generate **brush stroke libraries** if not already stored in cache.

```
python3 robot_scripts/Frida/paint_strokes.py
```


To change the range of parameters of strokes to generate, update those values in *options.py*

```
        #TODO: spacing of strokes: right now they will overlap. And if small strokes are small, rest of the column will be left blank.
        self.MIN_STROKE_LENGTH = 1# 0.001
        self.MAX_STROKE_LENGTH = 60 #0.06
        self.MIN_STROKE_Z = 5 #0.05
        self.MAX_BEND = 20 #0.02 #2cm
```


#### Training Robot Strokes (generate Renderer)



#### Painting with a Robot

```
python3 paint_inference.py --use_cache
```

paramas to use
- --use_cache
- --max_step
- --actor
- --renderer
- --img
- --imgid
- --divide
- --debug

### References
1. https://github.com/cmubig/Frida

