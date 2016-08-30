# FacialExpressionRecognition

This is my implementation of a Convolutional Neural Network for Facial Expression Recognition. 

I used the fer2013 dataset on Kaggle. The model has an accuracy of ~68% on the test set before using the averaging method and ~69% 
after applying the averaging method. 

Due to limited time, I did not create an emsemble. An emsemble of multiple trained models using different initialization 
might improve the result a bit more.

To train the model using GPU run THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnnmodel.py

To test the accuracy after averaging run  THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python averagingmethod.py

Different packages need to be installed. You can also used pre-installed Keras and Theano AMI on Amazon Web Services. Imutils, 
OpenCV can be installed by pip and conda. They are used for the averaging method.

An demonstration of the CNN:
http://lamuong.com/myapp/

I used C++ library Dlib for face detection. 

The Source Code for Django app: 
https://github.com/LamUong/DjangoWithCNN

I used several ideas in these papers to create my model:

https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/icmi2015_ChaZhang.pdf

http://www.cs.toronto.edu/~tang/papers/dlsvm.pdf


