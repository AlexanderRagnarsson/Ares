import pytesseract
from PIL import Image
import numpy

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


big=Image.open('Hearthstone myndir\\Angry chicken full.png')            

small = Image.open('Hearthstone myndir\\New folder\\1 hp (abusive).png')

print(subimg(small, big))