# Opencv http://docs.opencv.org/3.0-beta/modules/refman.html
import cv2
# print("Opencv version: {}".format(cv2.__version__))	
# Numpy http://www.numpy.org/
import numpy as np
# print("Numpy version: {}".format(np.__version__))  

def convert_to_gray(images):
	images_gray = []
	for image in images:	
		# Convert BGR to HSV
		gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		images_gray.append(gray)
	
	return images_gray

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def sobel_operator(image, orient='x', thresh_min=0, thresh_max=255, sobel_kernel = 3):
	# Convert to grayscale
	gray = convert_to_gray([image])
	
	# Apply x or y gradient with the OpenCV Sobel() function
	if orient == 'x':
		abs_sobel = np.absolute(cv2.Sobel(gray[0], cv2.CV_64F, 1, 0, ksize=sobel_kernel))
	if orient == 'y':
		abs_sobel = np.absolute(cv2.Sobel(gray[0], cv2.CV_64F, 0, 1, ksize=sobel_kernel))
		
	# Rescale back to 8 bit integer
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
	# Create a copy and apply the threshold
	binary_output = np.zeros_like(scaled_sobel)
	# Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
	binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
	
	return binary_output

	
def mag_thresh(img, sobel_kernel=3, mag_thresh=(20, 100)):
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# Take both Sobel x and y gradients
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Calculate the gradient magnitude
	gradmag = np.sqrt(sobelx**2 + sobely**2)
	# Rescale to 8 bit
	scale_factor = np.max(gradmag)/255 
	gradmag = (gradmag/scale_factor).astype(np.uint8) 
	# Create a binary image of ones where threshold is met, zeros otherwise
	binary_output = np.zeros_like(gradmag)
	binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

	# Return the binary image
	return binary_output


# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0.6, 1.3)):
	# Grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# Calculate the x and y gradients
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Take the absolute value of the gradient direction, 
	# apply a threshold, and create a binary image result
	absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
	binary_output =  np.zeros_like(absgraddir)
	binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
	
	# Return the binary image
	return binary_output
	
	
def binary_operator(image, thresh_min=180, thresh_max=255):	
	gray = np.copy(image)
	if len(image.shape) == 3:	
		gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	binary = np.zeros_like(gray)
	binary[(gray > thresh_min) & (gray <= thresh_max)] = 1
	
	return binary
	
	
def convert_rgb_to_hls(image):
	
	hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
	H = hls[:,:,0]
	L = hls[:,:,1]
	S = hls[:,:,2]
	
	return H, L, S	


def convert_rgb_to_hsv(image):
	
	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	H = hsv[:,:,0]
	S = hsv[:,:,1]
	V = hsv[:,:,2]
	
	return H, S, V
	
	
def perspective_transform(image):
	
	img_size = (image.shape[1], image.shape[0])
	width, height = img_size
	offset = 200
	src = np.float32(
	[[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
	[((img_size[0] / 6) - 10), img_size[1]],
	[(img_size[0] * 5 / 6) + 60, img_size[1]],
	[(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
	dst = np.float32(
	[[(img_size[0] / 4), 0],
	[(img_size[0] / 4), img_size[1]],
	[(img_size[0] * 3 / 4), img_size[1]],
	[(img_size[0] * 3 / 4), 0]])	
	
	
	#src = np.float32([
	#[  588,   446 ],
	#[  691,   446 ],
	#[ 1126,   673 ],
	#[  153 ,   673 ]])
	
	#dst = np.float32([[offset, 0], [img_size[0] - offset, 0], [img_size[0] - offset, img_size[1]], [offset, img_size[1]]])
	M = cv2.getPerspectiveTransform(src, dst)
	Minv = cv2.getPerspectiveTransform(dst, src)
	binary_warped = cv2.warpPerspective(image, M, (width, height))
	
	return binary_warped, Minv
	
	
	
	
	