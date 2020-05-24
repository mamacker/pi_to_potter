import subprocess
import sys, traceback
import os

wav_process = None
def play_wav(file_path):
    global wav_process
    try:
        if wav_process is not None:
            wav_process.kill();
        wav_process = subprocess.Popen(["/usr/bin/aplay", file_path]);
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("*** print_exception:")
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)

def stop_wav():
    global wav_process
    try:
        if wav_process is not None:
            wav_process.kill();
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("*** print_exception:")
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)

def play_mp3(file_path):
    os.system('killall mpg321');
    os.system(f'mpg321 {file_path} &')
