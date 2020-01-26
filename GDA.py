# detection library for images
from PIL import Image
import numpy
import time
import random
import math
class GDA:
    SEPARATION_CONSTANT = int(0.1 * 255)

    def get_hs_number(image):
        pass

    def get_client(image):
        #rgb_values = numpy.array(image.convert('L'))
        gray_image = image.convert("L")
        loops = GDA.__get_loops_from_image(image)
        gray = GDA.__delta_image(gray_image)
        gray = gray.convert("RGBA")
        gray_array = numpy.array(gray)
        i2 = Image.fromarray(gray_array)
        i2.show()
        i2.convert("RGBA").save('test.PNG', 'PNG')

    def __get_loops_from_image(image):
        loop_images = []
        gray = image.convert('L')
        gray_array = numpy.array(gray)
        y = len(gray_array)//2
        x = 1
        while x != len(gray_array[0])-1:
            if GDA.__delta_pixel((x, y), gray_array) > GDA.SEPARATION_CONSTANT:
                position = (x, y)
                is_loop, loop_image, highest_x_value_in_loop = GDA.__find_delta_loop(gray_array, position)
                if is_loop:
                    GDA.__extract_loop((x, y), gray_array)
            x += 1

    def __extract_loop(position, gray_array):
        smallest_x = position[0]
        biggest_x = position[0]
        smallest_y = position[1]
        biggest_y = position[1]
        walk_mem = dict()
        walk_mem[True] = []

        # getting all cords in loop
        def walk(x, y):
            for dxy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (x+dxy[0], y+dxy[1]) not in walk_mem:
                    if GDA.__delta_pixel((x+dxy[0], y+dxy[1]), gray_array) > GDA.SEPARATION_CONSTANT:
                        walk_mem[(x+dxy[0], y+dxy[1])] = True
                        walk_mem[True] += [(x+dxy[0], y+dxy[1])]
                        walk(x+dxy[0], y+dxy[1])
        walk(position[0], position[1])
        # finding biggest and smallest x and y
        loop = walk_mem[True]
        for xy in loop:
            x, y = xy
            if x < smallest_x:
                smallest_x = x
            elif biggest_x < x:
                biggest_x = x

            if y < smallest_y:
                smallest_y = y
            elif biggest_y < y:
                biggest_y = y

        # creating image from info
        dy = biggest_y - smallest_y + 1
        dx = biggest_x - smallest_x + 1
        image_loop_array = []
        for i in range(dy):
            image_loop_array += [[0]*dx]

        for xy in loop:
            x, y = xy
            image_loop_array[y-smallest_y][x-smallest_x] = 255

        image_loop = Image.fromarray(numpy.array(image_loop_array))
        image_loop.show()











    def __find_delta_loop(gray_array, position):
        # if loop is not found return False, None, None
        # if loop is found return True, Image_of_loop, highest_x_value_in_loop relative too grid
        x, y = position
        x1, x2 = x, x
        # finding edge of border
        while True:
            # Edge case if number is on border this will check index -1 and do something stupid
            if GDA.__delta_pixel((x1 - 1, y), gray_array) > GDA.SEPARATION_CONSTANT:
                x1 -= 1
            else:
                break

        while True:
            # Edge case if number is on border this will check index i above max length and crash
            if GDA.__delta_pixel((x2 + 1, y), gray_array) > GDA.SEPARATION_CONSTANT:
                x2 += 1
            else:
                break

        walk_mem = []
        for i in gray_array:
            walk_mem += [[-1] * len(i)]

        for i in range(x1, x2+1):
            walk_mem[y][i] = 1
            walk_mem[y + 1][i] = 1
            walk_mem[y + 2][i] = 0
        print(position, "SEARCH---------------EDGE:",x1,x2)
        if GDA.__find_the_zero((x, y), walk_mem, gray_array):
            return True, None, None
        return False, None, None

    TEST = []
    def __find_the_zero(position, walk_mem, grey_array):
        GDA.TEST += [position]
        x, y = position
        walk_mem[y][x] = 1
        for dxy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if walk_mem[y+dxy[1]][x+dxy[0]] == 0:
                print("found 0!!!!!!!!!!!!", x, y)
                return True
            elif walk_mem[y+dxy[1]][x+dxy[0]] == -1 and GDA.__delta_pixel((x+dxy[0], y+dxy[1]), grey_array) > GDA.SEPARATION_CONSTANT:
                if GDA.__find_the_zero((x + dxy[0], y + dxy[1]), walk_mem, grey_array) == True:
                    return True
        return False





    def __delta_pixel(cord, gray_image_array):
        x, y = cord
        dxy_matrix = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if dx != 0 or dy != 0]
        delta_sum = 0
        current_value = int(gray_image_array[y][x])
        for dxy in dxy_matrix:
            delta_sum += abs(current_value - gray_image_array[y + dxy[0]][x + dxy[1]])
        return delta_sum//8






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
        ratio = 1#255/biggest
        for y in range(len(pixels_delta)):
            for x in range(len(pixels_delta[y])):
                pixels_delta[y][x] = int(pixels_delta[y][x]*ratio)
                if pixels_delta[y][x] > GDA.SEPARATION_CONSTANT:
                    pixels_delta[y][x] = 255
                else:
                    pixels_delta[y][x] = 0
        delta_image = Image.fromarray(numpy.array(pixels_delta))
        return delta_image

image = Image.open("Hearthstone_numbers/number_10_2.png")
t1 = time.time()
GDA.get_client(image)
t2 = time.time()
print(t2-t1)