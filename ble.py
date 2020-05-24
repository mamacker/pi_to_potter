from __future__ import absolute_import
from __future__ import print_function
from bluepy import btle
from bluepy.btle import Scanner, DefaultDelegate
import music

found = False
class ScanDelegate(DefaultDelegate):
    def __init__(self):
        DefaultDelegate.__init__(self)

    def handleDiscovery(self, dev, isNewDev, isNewData):
        global found;
        if isNewDev:
            print("Discovered device", dev.addr)
            if (dev.addr == 'cb:22:99:ce:97:8f'):
                found = True
        elif isNewData:
            print("Received new data from", dev.addr)

scanner = Scanner().withDelegate(ScanDelegate())

failures = 0;
def runScanAndSet(state):
    global found;
    found = False;
    devices = scanner.scan(3)
    try:
        peripheral = btle.Peripheral('cb:22:99:ce:97:8f', btle.ADDR_TYPE_RANDOM)
        failures = 0;
    except:
        failures += 1;
        if (failures < 10):
            runScanAndSet(state);
        else:
            failures = 0;
        return;

    finally:
        guid = '713d0003503e4c75ba943148f18d941e'
        characteristic = peripheral.getCharacteristics(uuid=guid)[0];
        if (state):
            turnOn(characteristic);
            turnOn(characteristic);
        if (not state):
            turnOff(characteristic);
            turnOff(characteristic);

def turnOn(characteristic):
    # Set Output
    command = bytearray(3);
    command[0] = 0x53; #S
    command[1] = 0x04;
    command[2] = 0x01;

    print(str(command))
    characteristic.write(command);

    # Turn on
    command = bytearray(3);
    command[0] = 0x54; #T
    command[1] = 0x04;
    command[2] = 0x01;

    print(str(command))
    characteristic.write(command);

def turnOff(characteristic):
    # Set Output
    command = bytearray(3);
    command[0] = 0x53; #S
    command[1] = 0x04;
    command[2] = 0x01;

    print(str(command))
    characteristic.write(command);

    # Turn on
    command = bytearray(3);
    command[0] = 0x54; #T
    command[1] = 0x04;
    command[2] = 0x00;

    print(str(command))
    characteristic.write(command);

bleState = False;
def toggleBLE():
    global bleState;
    global bellProcess
    music.play_wav(f'{home_address}/pi_to_potter/music/bell.wav');

    bleState = not bleState;
    runScanAndSet(bleState);
    time.sleep(10);
    music.stop_wav()

