# pi_to_potter
Want to recreate the Universal Studios Magical experience?  Try this out, in a couple of hours you can have your own magic wand, running on a Raspberry PI 3.

This uses machine learning, and background subtraction with OpenCV all on a little $35 raspberry pi.  Several people have now recreated the whole exerience using this tech.  It might even be better than the original, since *this* wand experience can be taught new tricks. ;)

See the whole build, tips, tricks and comments at: https://bloggerbrothers.com/2017/12/09/turn-on-a-lamp-with-a-gesture-ir-cam-image-processing/

## The simplest way to get started:
- Create a fresh raspbian disk with the Desktop OS
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
