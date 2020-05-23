# pi_to_potter
Want to recreate the Universal Studios Magical experience?  Try this out, in a couple of hours you can have your own magic wand, running on a Raspberry PI 3/4.

This uses machine learning, and background subtraction with OpenCV all on a little $35 raspberry pi.  Several people have now recreated the whole exerience using this tech.  It might even be better than the original, since *this* wand experience can be taught new tricks. ;)

See the whole build, tips, tricks and comments at: https://bloggerbrothers.com/2017/12/09/turn-on-a-lamp-with-a-gesture-ir-cam-image-processing/

## The simplest way to get started:
- Create a fresh raspbian disk with the Desktop OS
- Run through the instructions found in: steps_taken.txt
- Download the git repository
- run:
  - python magic_wand.py

## To start the support webpage:
- cd nodeservice
- npm install
- node index.js 

## To get LOTS more detail:
Blog article about this code base:
https://bloggerbrothers.com/2017/12/09/turn-on-a-lamp-with-a-gesture-ir-cam-image-processing/


