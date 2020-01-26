import pyautogui
import keyboard
import time
import random


# print(dir(pyautogui))

SCREEN_RES = 1600,900

CORNER = 0, 39

CONFIRM = 814, 750

COIN_FIRST = 487, 462
COINT_X_DELTA = 210

FIRST = 512, 458
X_DELTA = 285

ADD_TO_X = 20       # What we add the x value of card in hand to land in the middle of the card
Y_HAND = 910        # 

END_TURN = 1306,453
HERO_POWER = 940,727

ENEMY_HERO = 800,201


Y_ENEMY_BOARD = 373
Y_BOARD = 532

BOARD_X_DELTA = (859-740) / 2

BOARD_X_MIDDLE = 797

EVEN_ODD_BOARD_DELTA = 0

# email = alexanderragnarssonag@gmail.com
# PASSWORD = Areserbestur1
# first_car = Ares#26419

# HAND = [712]
# HAND = [655,766]                                # 111
# HAND = [604,713,824]                            # 110 
# HAND = [544,657,763,875]                        # 110.33?
# HAND = [534,626,711,896,892]                    # 89.5?
# HAND = [524,608,678,748,819,890]                # 73.2?
# HAND = [521,589,651,715,774,838,902]            # 63.5?
# HAND = [507,572,632,686,739,790,846,903]        # 56.5?
# HAND = [508,564,616,665,711,761,806,855,914]    # 50.75
# HAND = [494,912]                                # 46.6
# FIRST_CARD_HAND = 548 - 8 * len(HAND) 


# MINION = []


def act_human(coordinate):
    # small = coordinate/20
    # change = random.randrange(-small,small)
    # return coordinate + change
    return coordinate

def first_card_x(card_count):
    if card_count >= 8:
        return 507 - card_count % 8 * 5 + ADD_TO_X
    elif card_count >= 3:
        return 507 + card_count * 8 + ADD_TO_X
    elif card_count == 2:
        return 655 + ADD_TO_X
    else:       # 1 card in hand
        return 712 + ADD_TO_X


def get_delta_x(card_count):
    # if card_count == 10:
        # return 40
    if card_count >= 5:
        return round(110 - 6.5*card_count) 
    elif card_count == 1:
        return 0
    else:
        return 110
print(get_delta_x(10))

def play_card(card_number=1,card_count=1):
    card_pos = act_human(first_card_x(card_count) + (card_number - 1)*get_delta_x(card_count)), act_human(Y_HAND)
    pyautogui.moveTo(card_pos)
    time.sleep(0.1)
    screen_x,screen_y = SCREEN_RES
    screen_x /= 2
    screen_y /= 2

    # pyautogui.mouseDown(button='left')
    # time.sleep(0.1)
    pyautogui.dragTo(act_human(screen_x),act_human(screen_y), 0.2, button = 'left')
    # pyautogui.mouseUp(button='left')
    time.sleep(0.5)

def end_turn():
    time.sleep(0.3)
    pyautogui.moveTo(END_TURN)
    pyautogui.mouseDown(duration=0.1)
    pyautogui.mouseUp(duration=0.1)
    time.sleep(15)

def hero_power():
    pyautogui.moveTo(HERO_POWER)
    pyautogui.dragTo(act_human(ENEMY_HERO[0]),act_human(ENEMY_HERO[1]),0.15,button='left')
    # pyautogui.mouseDown(duration=0.1)
    # pyautogui.mouseUp(duration=0.1)
    time.sleep(0.1)

def SMORC():
    for i in range(7):
        minion_pos = act_human(BOARD_X_MIDDLE + (i*BOARD_X_DELTA + EVEN_ODD_BOARD_DELTA)),act_human(Y_BOARD)
        pyautogui.moveTo(minion_pos)
        pyautogui.dragTo(act_human(ENEMY_HERO[0]),act_human(ENEMY_HERO[1]),0.15,button='left')
        attack_all_minions(minion_pos)

        time.sleep(0.02)

        minion_pos = act_human(BOARD_X_MIDDLE + - (i*BOARD_X_DELTA + EVEN_ODD_BOARD_DELTA)),act_human(Y_BOARD)
        pyautogui.moveTo(minion_pos)
        pyautogui.dragTo(act_human(ENEMY_HERO[0]),act_human(ENEMY_HERO[1]),0.15,button='left')
        attack_all_minions(minion_pos)

        time.sleep(0.02)        


def play_spell(card_number=1,card_count=1):
    screen_x,screen_y = SCREEN_RES
    screen_x /= 2
    screen_y /= 2

    card_pos = first_card_x(card_count) + (card_number - 1)*get_delta_x(card_count), Y_HAND
    pyautogui.moveTo(card_pos)
    time.sleep(0.1)

    pyautogui.dragTo(act_human(ENEMY_HERO[0]),act_human(ENEMY_HERO[1]), 0.15, button = 'left')
    attack_all_minions(card_pos)
    pyautogui.mouseUp(button='left')


def attack_all_minions(original_pos):
    for i in range(7):
        pyautogui.dragTo(act_human(BOARD_X_MIDDLE + (i*BOARD_X_DELTA + EVEN_ODD_BOARD_DELTA)),act_human(Y_ENEMY_BOARD),0.04,button='left')
        pyautogui.moveTo(original_pos)
        pyautogui.dragTo(act_human(BOARD_X_MIDDLE - (i*BOARD_X_DELTA + EVEN_ODD_BOARD_DELTA)),act_human(Y_ENEMY_BOARD),0.04,button='left')
        
        time.sleep(0.005)

# for i in range(10,0,-1):

    # hero_power

while True:
    # if keyboard.read_key() == 'q':
    #     break
    time.sleep(1)
    # for i in range(1,5):

    for j in range(1,11):
        play_spell(j,10)
        pyautogui.click(button='right')
            # play_card(i,j)

    SMORC()
    pyautogui.click(button='right')

    hero_power()

    end_turn()
    time.sleep(4)




def play_hearhtstone():
    print(pyautogui.position())

while True:
    if keyboard.read_key() == 'g':
        play_hearhtstone()
        time.sleep(0.5)
    elif keyboard.read_key() == 'q':
        break