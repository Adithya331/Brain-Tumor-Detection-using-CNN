#imports
import cv2
import os
import keras
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.layers import Conv2D , MaxPooling2D
from keras.layers import Activation , Dropout , Flatten , Dense
from keras.utils.np_utils import to_categorical


images_directory = './datasets/' 

no_tumor_images = os.listdir(images_directory + 'no/') 
yes_tumor_images = os.listdir(images_directory + 'yes/') 
dataset = []
label = []

INPUT_SIZE = 64

print(no_tumor_images)
print(yes_tumor_images)


for i , image_name in enumerate(no_tumor_images):
  if(image_name.split('.')[1]=='jpg'):
    image = cv2.imread(images_directory +'no/' + image_name)
    image = Image.fromarray(image , 'RGB')
    image = image.resize((INPUT_SIZE , INPUT_SIZE)) 
    dataset.append(np.array(image))#Adding the images as a numpy arrays to dataset
    label.append(0)

print(len(dataset))

for i , image_name in enumerate(yes_tumor_images):
  if(image_name.split('.')[1]=='jpg'):
    image = cv2.imread(images_directory +'yes/' + image_name)
    image = Image.fromarray(image , 'RGB')
    image = image.resize((INPUT_SIZE , INPUT_SIZE)) 
    dataset.append(np.array(image))#Adding the images as a numpy arrays to dataset
    label.append(1)

print("length of dataset ",len(dataset))
print("length of label ",len(label))

dataset = np.array(dataset)
label = np.array(label)

# splitting the dataset into training and testing sets
x_train , x_test , y_train , y_test = train_test_split(dataset , label , test_size = 0.2 , train_size=0.8, random_state=0)
#this is actually x and y coordinate
print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

# divide data into 80% and 20%

x_train = normalize(x_train  , axis=1)
x_test = normalize(x_train  , axis=1)

y_train = to_categorical(y_train , num_classes=2)
y_test = to_categorical(y_test , num_classes=2) #numsclass is 2 because we have 2 classes(yes and no)


from keras.activations import sigmoid
from keras.backend import flatten
from keras.engine.sequential import Sequential


#Model builing 
model  = Sequential() # Initialising CNN 
#first convolutional layer
model.add(Conv2D(32 , (3,3) , input_shape = (INPUT_SIZE , INPUT_SIZE , 3))) 
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2 , 2))) 

#second convolutional layer
model.add(Conv2D(32 , (3,3) , kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2 , 2)))

#third convolutional layer
model.add(Conv2D(64 , (3,3) , kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2 , 2)))


model.add(Flatten()) 
model.add(Dense(64)) # 
model.add(Activation('relu')) 
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('sigmoid')) 

# Training the CNN (Compiler the cnn)
model.compile(loss='categorical_crossentropy' , optimizer = 'adam' ,  metrics=['accuracy'])


# training the cnn on training dataset
res = model.fit(np.array(x_train), np.array(y_train), verbose=1, epochs=20 ,shuffle = False)

model.save('BrainTumor.h5')

#now testing

import matplotlib.pyplot as plt
print(res.history.keys())
print()
# loss
plt.plot(res.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

print("Training accuracy")

plt.plot(res.history['accuracy'])
plt.title('Model Accuracy')
plt.legend(['Training Accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()


