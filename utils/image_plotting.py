#https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html#matplotlib.pyplot.imshow	
#https://matplotlib.org/api/_as_gen/matplotlib.colors.Colormap.html#matplotlib.colors.Colormap 	
import matplotlib.pyplot as plt
# Opencv http://docs.opencv.org/3.0-beta/modules/refman.html
import cv2
# print("Opencv version: {}".format(cv2.__version__))	
# Numpy http://www.numpy.org/
import numpy as np
# print("Numpy version: {}".format(np.__version__)) 

from mpl_toolkits.mplot3d import Axes3D

# Plotting images
def plot_images(images, color_map = None, columns = 5, scale = 1):	
	plt.figure(figsize=(scale*20,scale*10))
	for i, image in enumerate(images):
		plt.subplot(len(images) / columns + 1, columns, i + 1)
		plt.imshow(image, color_map)
	
	
def draw_boxes(img, bboxes, color = (0, 0, 255), thick = 6):
	# Make a copy of the image
	draw_img = np.copy(img)
	# Iterate through the bounding boxes
	for bbox in bboxes:
		# Draw a rectangle given bbox coordinates
		cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
	# Return the image copy with boxes drawn
	return draw_img
	
	
def plot3d(pixels, colors_rgb, axis_labels=list("RGB"), axis_limits=((0, 255), (0, 255), (0, 255))):
	"""Plot pixels in 3D."""

	# Create figure and 3D axes
	fig = plt.figure(figsize=(8, 8))
	ax = Axes3D(fig)

	# Set axis limits
	ax.set_xlim(*axis_limits[0])
	ax.set_ylim(*axis_limits[1])
	ax.set_zlim(*axis_limits[2])

	# Set axis labels and sizes
	ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
	ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
	ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
	ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

	# Plot pixel values with colors given in colors_rgb
	ax.scatter(
		pixels[:, :, 0].ravel(),
		pixels[:, :, 1].ravel(),
		pixels[:, :, 2].ravel(),
		c=colors_rgb.reshape((-1, 3)), edgecolors='none')

	return ax  # return Axes3D object for further manipulation


def drawBoudingBoxes(image, bboxes, color, thick):
	copy_image = np.copy(image)
	# https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html#rectangle
	for box in bboxes:
		cv2.rectangle(copy_image, box[0], box[1], color, thick)
	return copy_image
	
	
	
