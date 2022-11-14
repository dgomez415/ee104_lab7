# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 16:00:11 2022
## https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
@author: Christopher
"""



import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

#https://stackoverflow.com/questions/69687794/unable-to-manually-load-cifar10-dataset
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0



## verify that the dataset looks correct
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# define cnn BaseLine model
model = models.Sequential()

### This section is from https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/cnn.ipynb#scrollTo=WRzW5xSDDbNF ###
### You will improve this section for higher accuracy 
# Create the convolutional base and Add Dense layers on top
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
# Here's the complete architecture of your model:
model.summary()

### End code from https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/cnn.ipynb#scrollTo=WRzW5xSDDbNF ####




## Compile and train the model
opt = 'adam'
model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=100    , 
                    validation_data=(test_images, test_labels))


## Evaluate the model
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('Test accuracy =')
print(test_acc)

## save trained model in file "MyGroup_CIFARmodel.h5"
# You will use this trained model to test the images 
model.save('MyGroup_CIFARmodel_baseline.h5')

## Save file to your local computer if you will test it locally
##  using the provided file test_image.py
#https://neptune.ai/blog/google-colab-dealing-with-files 
from google.colab import files
files.download('MyGroup_CIFARmodel.h5')

# ## If you run on GoogleColab, then add the following code on GoogleColab ##
# ###########################################################################

# # load the trained CIFAR10 model
# from keras.models import load_model
# model = load_model('MyGroup_CIFARmodel.h5')

# def load_image(filename):
# 	img = load_img(filename, target_size=(32, 32))
# 	img = img_to_array(img)
# 	img = img.reshape(1, 32, 32, 3)
# 	img = img / 255.0
# 	return img

# #https://stackoverflow.com/questions/72479044/cannot-import-name-load-img-from-keras-preprocessing-image
# from tensorflow.keras.utils import load_img
# from tensorflow.keras.utils import img_to_array
# from keras.models import load_model

# # get the image from the internet
# URL = "https://wagznwhiskerz.com/wp-content/uploads/2017/10/home-cat.jpg"
# picture_path  = tf.keras.utils.get_file(origin=URL)
# img = load_image(picture_path)
# result = model.predict(img)


# # show the picture
# image = plt.imread(picture_path)
# plt.imshow(image)

# # show prediction result.
# print('\nPrediction: This image most likely belongs to ' + class_names[int(result.argmax(axis=-1))])

# # get the image from the internet
# URL = "https://image.shutterstock.com/image-vector/airplane-600w-646772488.jpg"
# picture_path  = tf.keras.utils.get_file(origin=URL)
# img = load_image(picture_path)
# result = model.predict(img)


# # show the picture
# image = plt.imread(picture_path)
# plt.imshow(image)

# # show prediction result.
# print('\nPrediction: This image most likely belongs to ' + class_names[int(result.argmax(axis=-1))])


############################################################
## This website has everything to improve your accuracy  ###
############################################################

# https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/ 

