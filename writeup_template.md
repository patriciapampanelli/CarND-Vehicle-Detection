# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image_car_1]: ./report_images/vehicles_1.png
[image_car_2]: ./report_images/vehicles_2.png
[image_car_3]: ./report_images/vehicles_3.png
[image_not_car_1]: ./report_images/not_vehicles_1.png
[image_not_car_2]: ./report_images/not_vehicles_2.png
[image_not_car_3]: ./report_images/not_vehicles_3.png
[hog_image_car_1]: ./report_images/hog_car_normalized_yuv.png
[hog_image_not_car_1]: ./report_images/hog_not_car_normalized_yuv.png
[hog_image_car_1_ycrcb]: ./report_images/hog_car_normalized_ycrcb.png
[hog_image_not_car_1_ycrcb]: ./report_images/hog_not_car_normalized_ycrcb.png

[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

###### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf.

You're reading it!

---
###### Histogram of Oriented Gradients (HOG)

###### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook called Histogram of Oriented Gradients. I used the code below from the opencv library (reference: [HOGDescriptor](https://docs.opencv.org/2.4/modules/gpu/doc/object_detection.html)):

```python
def get_hog_features(img, color_space='YUV'):
	hog = cv2.HOGDescriptor((64,64), (16,16), (8,8), (8,8), 9)
	return np.ravel(hog.compute(img))
```

I started by reading in all the `vehicle` and `non-vehicle` images, as shown below. 

```python
# Cars
images_car = image_utils.read_images("./vehicles/*/")
# Not cars
images_not_car = image_utils.read_images("./non-vehicles/*/")
```

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:
![Car 1][image_car_1]
![Car 2][image_car_2]
![Car 3][image_car_3]
![Not a Car 1][image_not_car_1]
![Not a Car 2][image_not_car_2]
![Not a Car 3][image_not_car_3]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

![Histogram of Gradients - Car Sample][hog_image_car_1]
![Histogram of Gradients - Non Car Sample][hog_image_not_car_1]

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![Histogram of Gradients - Car Sample][hog_image_car_1_ycrcb]
![Histogram of Gradients - Non Car Sample][hog_image_not_car_1_ycrcb]

###### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and these parameters presented the best results. In other words, the detections were more precise and I got less false positives. 

###### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I tried three different classifiers from sklearn and I also used GridSearchCV for parameters optimizations:

- [SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
Seconds to train SVC: 1024.566371s
Train Accuracy of SVC =  1.0
Test Accuracy of SVC = 0.994369
Seconds to predict with SVC: 0.006285s
- [DecisionTreeClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)
Seconds to train Decision Tree: 276.273658s
Train Accuracy of Decision Tree =  0.979377815315
Test Accuracy of Decision Tree = 0.906250
Seconds to predict with Decision Tree: 0.000229s
- [Multi-Layer Perceptron classifier](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier)
Seconds to train MLP: 344.863771s
Train Accuracy of MLP =  1.0
Test Accuracy of MLP = 0.992117
Seconds to predict with MLP: 0.000392s

---
###### Sliding Window Search

###### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

A single function, process_frame in python notebook CarND-Vehicle-Detection, is used to extract features using hog sub-sampling and make predictions. The hog sub-sampling helps to reduce calculation time for finding HOG features and thus provided higher throughput rate.

###### 2. Show some examples of test images to demonstrate how your pipeline is working. What did you do to optimize the performance of your classifier?

I used a GridSeachCV from sklearn for optimizing the performance of my classifier trying differente activation functions and solvers, as shown below:

```python
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

# Create a Classifier
clt = MLPClassifier(verbose=True)

parameters = {'activation':('logistic', 'relu'), 'solver':('sgd', 'adam')}
clt = GridSearchCV(mlp, param_grid=parameters)

# Train the model
clt.fit(X_train, y_train)
# Check the score of the SVC
print('Train Accuracy of MLP = ', clt.score(X_train, y_train))
print('Test Accuracy of MLP = {:0.6f}'.format(clt.score(X_test, y_test)))
# Check the prediction time for a single sample
prediction = clt.predict(X_test[0].reshape(1, -1))
```

---
###### Video Implementation

###### 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](https://github.com/patriciapampanelli/CarND-Vehicle-Detection/blob/master/result.mp4)

###### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions:

```python
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap
```

I then used [`cv2.findContours`](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html) to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.  

---

###### Discussion

###### 1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail?  What could you do to make it more robust?

The main difficulty I faced was working initially with the smaller dataset. In this context, the classifiers were not able to generalize and correctly identify the cars. Therefore, I decided to switch to the larger dataset. With more images, the classifiers were able to correctly identify the cars with few false positives.

