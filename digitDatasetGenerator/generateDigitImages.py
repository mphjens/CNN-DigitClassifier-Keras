import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy

width = 28
height = 28
path = "digits/" #Where to save the images

def speckle(img, value, amount):
    im = ndimage.imread(img)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if(np.random.randint(0,1/amount) == 0):
                im[i][j] = value

    return im

fnts = [ImageFont.truetype('OpenSans-Regular.ttf', 28 ),
ImageFont.truetype('OpenSans-Bold.ttf', 28),
ImageFont.truetype('OpenSans-Semibold.ttf', 28 ),
ImageFont.truetype('OpenSans-Light.ttf', 28 )
        ]

for cn in range(0,10): #For each digit
    for i in range(0, 200): #200 variations
        # make a blank image for the text
        img = Image.new('L', (int(width * (1 + np.random.uniform(-0.2, 0.2))), int(height * (1 + np.random.uniform(-0.2, 0.2))) ), (255))
        d = ImageDraw.Draw(img)
        fnt = fnts[np.random.randint(0,len(fnts))]
        #Write the digit to the image (fill=0 means black text)
        d.text((7 + np.random.randint(-5, 5),-5 + np.random.randint(-3, 3)), str(cn), font=fnt, fill=(0))
        #img.save("digits/"+str(cn) + "_" + str(i) + ".png")

        im2 = img.convert('RGBA')
        # rotated image
        rot = im2.rotate(np.random.uniform(-3, 3))
        # a white image same size as rotated image
        fff = Image.new('RGBA', rot.size, (255,) * 4)
        # create a composite image using the alpha layer of rot as a mask
        out = Image.composite(rot, fff, rot)

        out = out.convert("L")
        out = out.resize((width, height))

        #todo, this is inefficient. Convert to np array before saving and pass that to the speckle function
        out.save("digits/"+str(cn) + "_" + str(i) + ".png")
        specced = speckle(path+"/"+str(cn) + "_" + str(i) + ".png", 0, np.random.uniform(0, 0.1))
        scipy.misc.imsave(path+"/"+str(cn) + "_" + str(i) + ".png", specced)
        specced = speckle(path+"/" + str(cn) + "_" + str(i) + ".png", 255, np.random.uniform(0, 0.1))
        scipy.misc.imsave(path+"/" + str(cn) + "_" + str(i) + ".png", specced)

