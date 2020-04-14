from PIL import Image
import numpy as np

class Img:
    def make_square(self, img):
        cols,rows = img.size
        extra = abs(rows-cols)/2
        
        if rows>cols:
            r = (0,extra,cols,cols+extra)
        else:
            r = (extra,0,rows+extra,rows)

        return img.crop(r)

    def get_images(self, img_name_list, square = False, final_size=(0,0)):
        """ Returns a numpy matrix of matrixes representing the image
            Warps every tensor value on the range(-1,1) to square and resize images change the default values """
        return_list = []
        for img in img_name_list:
            i = Image.open(img)
            # i.load()
            if square:
                i = self.make_square(i)
                i = i.resize(final_size, Image.ANTIALIAS)
            img_arr = np.asarray(i)
            img_arr = img_arr.flatten()
            img_arr = img_arr.astype(np.float32)
            img_arr = (img_arr-128)/128
            return_list.append(img_arr)
        
        return np.array(return_list)

if __name__ == "__main__":
    def get_img_names():
        return_list = []
        for i in range(1,22):
            return_list.append(f'card_images/{i}.png')
        return return_list

    i = Img()
    images = i.get_images(get_img_names())
    print(images)