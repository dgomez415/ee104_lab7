# Authors
Authors: David Gomez, Osman Saeday

Co-Authors: Christopher Pham

# Video demonstration
Link to demonstration video: https://youtu.be/ZKNuK-Nbx5w

# Prerequisites
Prior to running the following programs, make sure the following packages are installed in your current version of Python: numpy, tensorflow, keras, matplotlib, jupyter

# Assembling Convolutional Neural Network model
cnn.ipynb assembles a convolutional neural network (CNN) model that can be used to recognize images of various objects (airplanes, deer, etc.). 
Original code provided to us by Tensorflow with an inital accuracy rating of ~70%. Upgraded model implementing Dropout Regularization and
Batch Normalization to improve accuracy rating ~88%. Upgraded model uses code by Jason Brownlee. Link to code: https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/ 

# Testing Convolutional Neural Network model
CNNbaseline.py and test.py are used to test our CNN model on it's ability to accurately determine the image shown it based on images it has never seen before.
CNNbaseline.py runs our model and creates a local file of the train model (MyGroup_CIFARmodel_baseline.h5). test.py uses that created file and imports previously unrecognized images
and shows those images to the model. The model then does it's best job to determine what the image is of. 

# Game
balloon.py is a game where you control a hot air balloon that must stay afloat in the air, avoiding obstacles on the way. 
The game was modified in the following ways: 'More High Scores', 'Lives', 'Speed It', and 'Add in Multiple Obstacles'
