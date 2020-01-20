import pytesseract
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import numpy
from pathlib import Path
import platform
from fixed_path import fix

def subimg(img1,img2):
    img1=numpy.asarray(img1)
    img2=numpy.asarray(img2)

    #img1=numpy.array([[1,2,3],[4,5,6],[7,8,9]])
    #img2=numpy.array([[0,0,0,0,0],[0,1,2,3,0],[0,4,5,6,0],[0,7,8,9,0],[0,0,0,0,0]])

    img1y=img1.shape[0]
    img1x=img1.shape[1]

    img2y=img2.shape[0]
    img2x=img2.shape[1]

    stopy=img2y-img1y+1
    stopx=img2x-img1x+1

    for x1 in range(0,stopx):
        for y1 in range(0,stopy):
            x2=x1+img1x
            y2=y1+img1y

            pic=img2[y1:y2,x1:x2]
            test=pic==img1

            if test.all():
                return x1, y1

    return False

def read_card(an_img):
    '''Returns the stats of a given card from its image.'''

    templates = ["Hearthstone myndir/New folder/1 mana.png","Hearthstone myndir/New folder/1 attack.png",\
        "Hearthstone myndir/New folder/1 hp (abusive) bigger.png"]

    for data_img in templates:
        data_img = fix(data_img)
        total = 0
        template = cv.imread(data_img,0)
        for meth in methods:
            method = eval(meth)
        # Apply template Matching
            res = cv.matchTemplate(img,template,method)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            total += max_val
        if 0.9 < total/3 < 1.1:
            print(data_img[29:])


# big=Image.open('Hearthstone myndir\\Angry chicken full.png')            

# small = Image.open('Hearthstone myndir\\New folder\\1 hp (abusive).png')

# print(subimg(small, big))



img = cv.imread(fix("Hearthstone myndir/Angry chicken full.png"),0)
img2 = img.copy()
template = cv.imread(fix('Hearthstone myndir/New folder/1 hp (abusive) bigger.png'),0)

total = 0
total_list = []
other_list = []

w, h = template.shape[::-1]
# All the 6 methods for comparison in a list
# methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            # 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
# Hin þrjú skila mjög stóru max_val
methods = ['cv.TM_CCOEFF_NORMED','cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF_NORMED']
read_card(img2)

colormap = 'rainbow'
# colormap = 'binary'

for meth in methods:
    img = img2.copy()
    method = eval(meth)
    # Apply template Matching
    res = cv.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    total += max_val
    total_list.append(max_val)
    other_list.append(min_val)
    # Þessi kóði sýnir hvar hann spottaði hlutinn sem hann fann (voða kúl)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(img,top_left, bottom_right, 255, 2)
    plt.subplot(121),plt.imshow(res,cmap = colormap)
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = colormap)
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()

# Eftir að ákveða fastana
if 0.9 < total/3 < 1.1:
    print(f'Init total: {total/3}')
else:
    print(f'Notinit total: {total/3}') 

print(total_list)
print(other_list)