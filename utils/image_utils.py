# Opencv http://docs.opencv.org/3.0-beta/modules/refman.html
import cv2
# print("Opencv version: {}".format(cv2.__version__))

# Glob - Unix style pathname pattern expansion 
# https://docs.python.org/2/library/glob.html
import glob

import matplotlib.image as mpimg

def read_image(filename, image_color):
	from skimage.io import imread
	
	image = imread(filename)
	#image = cv2.imread(filename, image_color)
	return image    

def read_images(folder, extensions = ['.jpg', '.jpeg', '.png'], image_color = 1):
	# image_color == 1: cv2.IMREAD_COLOR : Loads a color image. Any transparency of image will be neglected. It is the default flag.
	# image_color == 0: cv2.IMREAD_GRAYSCALE : Loads image in grayscale mode
	# image_color == -1: cv2.IMREAD_UNCHANGED : Loads image as such including alpha channel	
	
	images = []
	for ext in extensions:
		name = folder + '/*' + ext
		# print(name)
		file_names = glob.glob(name)
		# print(file_names)
		for image_name in file_names:
			# print(image_name)
			image = read_image(image_name, image_color)
			images.append(image)
	return images
	
	
def save_image(image, filename):
		
	from scipy.misc import imsave
	imsave(filename, image)
	
	#cv2.imwrite(filename,image)
