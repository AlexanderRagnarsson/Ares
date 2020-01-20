# https://drive.google.com/drive/folders/1p8u2Cpl2-atj0kynW74yfswbp8TLEGWp?fbclid=IwAR1ov-P3kFA4IcHoVJdzVxpcD_TSxNu7DAM5H1OY0LzpB-M4x3X84uDTOkI
from operator import itemgetter
import pytesseract
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import numpy
import os
import platform

max_mana = 3
max_attack = 4
max_hero_health = 30
max_health = 2
satisfying = 0.99
standard_res = 1080
card_res = 1080

# Kannski betra að hafa frekar min delta miðað við 1
# (maxið er kannski verri niðustaða)

def find_mana(card):
    mana_list = []
    for num in range(1,max_mana+1):
        mana = f'Hearthstone myndir\\New folder\\{num} mana.png'
        mana_list.append((num, get_max_val(card,mana)))
        if mana_list[-1][1] > satisfying:
            return mana_list[-1][0]
    print(mana_list)
    return closest_match(mana_list)


def find_health(card, hero=False):
    if hero:
        max_hp = max_hero_health
    else:
        max_hp = max_health
    hp_list = []

    for num in range(1,max_hp+1):
        hp = f'Hearthstone myndir\\New folder\\{num} hp.png'
        hp_list.append((num, get_max_val(card,hp)))
        if hp_list[-1][1] > satisfying:
            return hp_list[-1][0]
    print(hp_list)
    return closest_match(hp_list)


def find_attack(card):
    attack_list = []
    for num in range(1,max_attack+1):
        attack = f'Hearthstone myndir\\New folder\\{num} attack.png'
        attack_list.append((num, get_max_val(card,attack)))
        if attack_list[-1][1] > satisfying:
            return attack_list[-1][0]

    print(attack_list)
    return closest_match(attack_list)


def get_max_val(card, to_find):
    img = cv.imread(card,0)
    img2 = img.copy()
    template = cv.imread(to_find,0)

    scale_percent = card_res/standard_res
    new_width = int(template.shape[1] * card_res/standard_res)
    new_height = int(template.shape[0] * card_res/standard_res)
    dim = (new_width, new_height)

    template = cv.resize(template, dim, interpolation = cv.INTER_AREA)


    # w, h = template.shape[::-1]
    methods = ['cv.TM_CCOEFF_NORMED','cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF_NORMED']

    # colormap = 'gray'
    # colormap = 'Greys_r'
    total = 0



    for meth in methods:
        img = img2.copy()
        method = eval(meth)
        # Apply template Matching
        res = cv.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        total += max_val
    
    # if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
    #     top_left = min_loc
    # else:
    #     top_left = max_loc
    # bottom_right = (top_left[0] + w, top_left[1] + h)
    # cv.rectangle(img,top_left, bottom_right, 255, 2)
    # plt.subplot(121),plt.imshow(res,cmap = colormap)
    # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(img,cmap = colormap)
    # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    # plt.suptitle(meth)
    # plt.show()


    return total/len(methods)

def closest_match(a_list):
    """ Accepts a list of tuples (stat, certainty) returns closest match """
    the_stat, min_delta = a_list[0][0], abs(a_list[0][1]-1)

    for stat, match in a_list:
        if abs(match-1) < min_delta:
            the_stat, min_delta = stat, abs(match - 1)
        
    return the_stat

import time

for i in range(-445, -410, 1):
    card_res += i
    the_card = 'Hearthstone myndir\\Apprentice muligan.png'
    print(i, 'Stats:', find_mana(the_card),find_attack(the_card),find_health(the_card))
    card_res -= i

# Miðað við 1080p þá virkaði -100 á argent í hand
# Miðað við 1080p þá virkaði -450 á argent í muligan   -440 nákvæmast
# Fyrir muligan: -440, virkaði fyrir argent, annoy og apprentice