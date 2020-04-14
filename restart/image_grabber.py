import pyautogui
import keyboard


def save_image(pos, path, name):
    img = pyautogui.screenshot(region=(pos))
    img.save(f'{path}{name}.png')

def get_images():
    card_x_pos = 1200
    card_y_pos = 147

    y_delta = (789 - 150) / 19 

    image_x_pos_0 = 853
    image_y_pos_0 = 98

    image_x_pos_1 = 1121
    image_y_pos_1 = 465

    width = image_x_pos_1 - image_x_pos_0
    height = image_y_pos_1 - image_y_pos_0

    i = 1
    while i < 22:
        pyautogui.moveTo(card_x_pos,card_y_pos)
        save_image((image_x_pos_0,image_y_pos_0,width,height),'card_images/',str(i))
        card_y_pos += y_delta
        i += 1
        if 5 < i < 18:
            image_y_pos_0 += y_delta
        # time.sleep(0.2)


if __name__ == "__main__":
    # while True:
    #     if keyboard.read_key() == 'x':
    #         print(pyautogui.position())
    import time

    time.sleep(2)
    get_images()

# Point(x=1200, y=147)
# Point(x=1200, y=147)
# Point(x=1199, y=181)
# Point(x=1199, y=181)
# Point(x=1200, y=215)
# Point(x=1200, y=215)
# Point(x=1199, y=249)
# Point(x=1199, y=249)
# Point(x=1204, y=280)
# Point(x=1204, y=280)
# Point(x=853, y=112)
# Point(x=853, y=112)
# Point(x=1121, y=458)
# Point(x=1121, y=458)