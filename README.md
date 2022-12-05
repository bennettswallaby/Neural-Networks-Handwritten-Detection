# Neural-Networks-Handwritten-Detection
In this machine learning project, we will recognize handwritten characters, i.e, English alphabets from A-Z. This we are going to achieve by modeling a neural network that will have to be trained over a dataset containing images of alphabets.

##Project Prerequisites
Below are the prerequisites for this project:

Python (3.7.4 used)
IDE (Jupyter used)
Required frameworks are

Numpy (version 1.16.5)
cv2 (openCV) (version 3.4.2)
Keras (version 2.3.1)
Tensorflow (Keras uses TensorFlow in backend and for some image preprocessing) (version 2.0.0)
Matplotlib (version 3.1.1)
Pandas (version 0.25.1)

The dataset for this project contains 372450 images of alphabets of 28×2, all present in the form of a CSV file:
[Handwritten Character Detection](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format)



![page](https://user-images.githubusercontent.com/118565420/205702196-b4ef0908-34d3-479e-b30a-ddbb9c50df3a.jpg)

The convolution layers are generally followed by maxpool layers that are used to reduce the number of features extracted and ultimately the output of the maxpool and layers and convolution layers are flattened into a vector of single dimension and are given as an input to the Dense layer (The fully connected network).

![cnn](https://user-images.githubusercontent.com/118565420/205702330-c07bf60b-50e8-4b0d-897a-7137fcf793a0.jpg)


## Conclusion
Handwritten characters have been recognized with more than 97% test accuracy. This can be also further extended to identifying the handwritten characters of other languages too.
