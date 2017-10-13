# Plotting images
def plot_images(images, color_map = None, columns = 5, scale = 1):
	#https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html#matplotlib.pyplot.imshow	
	#https://matplotlib.org/api/_as_gen/matplotlib.colors.Colormap.html#matplotlib.colors.Colormap 	
	import matplotlib.pyplot as plt
	
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