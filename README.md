# FacialExpressionRecognition

This is my implementation of a convolutional neural network for facial expression recognition. 

I used the fer2013 dataset on Kaggle. The model has an accuracy of ~68% before using the averaging method and ~69% 
after applying averaging method. 

Due to limited time, I did not create an emsemble. An emsemble of multiple trained models using different initialization 
might improve the result a bit more.

To train the model using GPU simply run THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnnmodel.py. 

Different packages need to be installed. You can also used pre-installed Keras and Theano AMI on Amazon Web Services. Imutils, 
OpenCV can be installed by pip and conda. They are used for the averaging method.

An demonstration of the CNN:
http://52.53.186.250/myapp/

I used C++ library Dlib for face detection. 

The Source Code for Django app: 

