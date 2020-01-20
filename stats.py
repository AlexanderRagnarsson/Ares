from operator import itemgetter

max_mana = 12
max_attack = 20
max_hero_health = 30
max_health = 15

# Kannski betra að hafa frekar min delta miðað við 1
# (maxið er kannski verri niðustaða)

def find_mana(card):
    mana_list = []
    for num in range(1,max_mana+1):
        mana = f'Hearthstone myndir\\{num} mana.png'
        mana_list.append((num, get_max_val(card,mana)))
    
    return closest_match(mana_list)

def find_health(card, hero=False):
    if hero:
        max_hp = max_hero_health
    else:
        max_hp = max_health
    hp_list = []

    for num in range(1,max_hp+1):
        hp = f'Hearthstone myndir\\{num} hp.png'
        hp_list.append((num, get_max_val(card,hp)))
    
    return closest_match(hp_list)

def find_attack(card):
    attack_list = []
    for num in range(1,max_attack+1):
        attack = f'Hearthstone myndir\\{num} mana.png'
        attack_list.append((num, get_max_val(card,attack)))
    return closest_match(attack_list)

def get_max_val(card, template):
    card_copy = card.copy()
    methods = ['cv.TM_CCOEFF_NORMED','cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF_NORMED']

    # colormap = 'gray'
    # colormap = 'Greys_r'

    total_list = []

    for meth in methods:
        img = card_copy.copy()
        method = eval(meth)
        # Apply template Matching
        res = cv.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        total += max_val
        total_list.append(max_val)

        # # Þessi kóði sýnir hvar hann spottaði hlutinn sem hann fann (voða kúl)
        # # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
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
    
    return total_list/len(methods)

def closest_match(a_list):
    """ Accepts a list of tuples (stat, certainty) returns closest match """
    the_stat, min_delta = a_list[0][0], abs(a_list[0][1]-1)

    for stat, match in a_list:
        if abs(match-1) < min_delta:
            the_stat, min_delta = stat, match
        
    return the_stat