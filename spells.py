import os
from gpiozero import LED
import ble
import music

# You might be running on a device that doesn't have GPIOs
try:
    digitalLogger = LED(17)
    otherpin = LED(27)
except:
    pass


bubblesSwitch = False;
class Spells:
    def __init__(self, args):
        self.args = args
        if args.use_ble:
            ble.runScanAndSet(False)

    def cast(self, spell):
        global bubblesSwitch;
        #Invoke IoT (or any other) actions here
        if (spell=="center"):
            music.play_mp3(f'{home_address}/pi_to_potter/music/reys.mp3')
        elif (spell=="circle"):
            music.play_mp3(f'{home_address}/pi_to_potter/music/audio.mp3')
        elif (spell=="eight"):
            print("Togging digital logger.")
            music.play_mp3(f'{home_address}/pi_to_potter/music/tinkle.mp3')
            digitalLogger.toggle();
            None
        elif (spell=="left"):
            print("Toggling magic crystal.")
            if self.args.use_ble:
                ble.toggleBLE();
        elif (spell=="square"):
            None
        elif (spell=="swish"):
            None
        elif (spell=="tee"):
            print("Togging bubbles.")
            bubblesSwitch = not bubblesSwitch;
            music.play_mp3(f'{home_address}/pi_to_potter/music/spellshot.mp3')
            if (bubblesSwitch):
                os.system(f'{home_address}/pi_to_potter/bubbleson.sh');
            else:
                os.system(f'{home_address}/pi_to_potter/bubblesoff.sh');
            None
        elif (spell=="triangle"):
            print("Toggling outlet.")
            print("Playing audio file...")
            music.play_mp3(f'{home_address}/pi_to_potter/music/wonder.mp3')
        elif (spell=="zee"):
            print("Toggling 'other' pin.")
            print("Playing audio file...")
            music.play_mp3(f'{home_adress}/pi_to_potter/music/zoo.mp3')
            None
        print("CAST: %s" %spell)
