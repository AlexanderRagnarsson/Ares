import keyboard
import time
import pyautogui


def play_hearhtstone():
    print(pyautogui.position())

while True:
    if keyboard.read_key() == 'g':
        play_hearhtstone()
        time.sleep(0.5)
    elif keyboard.read_key() == 'q':
        break