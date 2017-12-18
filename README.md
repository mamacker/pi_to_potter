# pi_to_potter
Works! 

## The simplest way:
- First create a fresh raspbian disk with the Desktop OS
- Follow install instructions to get OpenCV 3.1
  - Run through the instructions found in: steps_taken.txt
- Download the git
- run:
  - python trained.py
  
## Using the v4l2 driver for camera control:
The v4l2 driver allows you to dial in the exposure, brightness and more.
 - Do the above.  
 - Follow the instructions in:
   - enable_v4l2.txt
 - run:
   - python trained_v4l2.py
  
Blog article about this code base:
https://bloggerbrothers.com/2017/12/09/turn-on-a-lamp-with-a-gesture-ir-cam-image-processing/
