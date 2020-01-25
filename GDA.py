# detection library for images
from PIL import Image
import numpy
import time
import random
import math
class GDA():
    def get_hs_number(image):
        pass

    def get_client(image):
        #rgb_values = numpy.array(image.convert('L'))
        i2 = GDA.__delta_image(image.convert('L'))
        i2.show()
        i2.convert("RGBA").save('test.PNG', 'PNG')

    def __get_loops_from_image(image):
        SEPARATION_CONSTANT = int(0.2 * 255)
        gray = image.convert('L')
        gray_array = numpy.array(gray)
        y = len(gray_array)//2
        dxy_matrix = [(dx,dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if dx != 0 or dy != 0]
        for x in range(1, len(gray_array[0]) - 1):
            delta_sum = 0
            current_value = int(gray_array[y][x])
            for dxy in dxy_matrix:
                delta_sum += abs(current_value - gray_array[y+dxy[0]][x+dxy[1]])
            if delta_sum//8 > 255 * SEPARATION_CONSTANT:
                pass







    def __delta_image(grascale_image):
        pixels_original = numpy.array(grascale_image)
        pixels_delta = []
        for y in range(len(pixels_original)):
            pixels_delta += [[0]*len(pixels_original[y])]

        biggest = 0
        for y in range(1, len(pixels_original)-1):
            for x in range(1, len(pixels_original[y])-1):
                delta_sum = 0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            delta_sum += abs(int(pixels_original[y][x])-int(pixels_original[y+dy][x+dx]))
                pixels_delta[y][x] = delta_sum//8
                if pixels_delta[y][x] > biggest:
                    biggest = pixels_delta[y][x]
        ratio = 255/biggest
        for y in range(len(pixels_delta)):
            for x in range(len(pixels_delta[y])):
                pixels_delta[y][x] = int(pixels_delta[y][x]*ratio)
                if pixels_delta[y][x] > 255*0.2:
                    pixels_delta[y][x] = 255
                else:
                    pixels_delta[y][x] = 0
        delta_image = Image.fromarray(numpy.array(pixels_delta))
        return delta_image



image = Image.open("Hearthstone_numbers/number_12_2.png")
t1 = time.time()
GDA.get_client(image)
t2 = time.time()
print(t2-t1)