from operator import itemgetter


# Kannski betra að hafa frekar min delta miðað við 1
# (maxið er kannski verri niðustaða)

def find_mana(card):
    mana_list = []
    for num in range(1,12):
        mana = f'Hearthstone myndir\\{num} mana.png'
        mana_list.append((num, get_max_val(card,mana)))
    return max(mana_list, key=itemgetter(1))[0]

def find_health(card):
    hp_list = []
    for num in range(1,12):
        hp = f'Hearthstone myndir\\{num} hp.png'
        hp_list.append((num, get_max_val(card,hp)))
    return max(hp_list, key=itemgetter(1))[0]

def find_attack(card):
    attack_list = []
    for num in range(1,12):
        attack = f'Hearthstone myndir\\{num} mana.png'
        attack_list.append((num, get_max_val(card,attack)))
    return max(attack_list, key=itemgetter(1))[0]

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