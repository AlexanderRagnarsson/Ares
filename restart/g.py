import pyautogui


for i in range(10000):
    img = pyautogui.screenshot()
    img.save(f'lolol/{i}.png')