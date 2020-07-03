import os
from os import listdir
from PIL import Image

dir_path = "images/"


for filename in listdir(dir_path):
    if filename.endswith('.jpg'):
        try:
            img = Image.open(dir_path+"\\"+filename) # open the image file
            img.verify() # verify that it is, in fact an image
        except (IOError, SyntaxError) as e:
            print('Bad file:', filename)
            #os.remove(base_dir+"\\"+filename) (Maybe)
    else:
        print("other Format: ", filename)