# UFactory LITE6 

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


### References
1. https://github.com/cmubig/Frida

