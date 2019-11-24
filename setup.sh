#!/bin/bash

v4l2-ctl --set-fmt-video=width=320,height=240,pixelformat=5
v4l2-ctl --set-ctrl=compression_quality=8 #8 is my default
v4l2-ctl --set-ctrl=scene_mode=11 #11=sports 8=night 0=none
v4l2-ctl --set-ctrl=auto_exposure=1 #0=auto #1=manual
v4l2-ctl --set-ctrl=exposure_time_absolute=1000 #1=fastest default=1000 
v4l2-ctl --set-ctrl=iso_sensitivity=0
v4l2-ctl --set-ctrl=iso_sensitivity_auto=0
v4l2-ctl --set-ctrl=rotate=0
v4l2-ctl -p 20
v4l2-ctl --set-ctrl=exposure_time_absolute=100 #1=fastest default=1000 

DISPLAY=:0 python magicwand.py --setup

