from __future__ import print_function

encodedd = None;

#Merge = ;


#def encodedd():
#	global out
#	out = 'encodedd'



import keras
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Model
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.layers import Merge
from keras.layers import Reshape
from keras import regularizers
#from keras.utils.layer_utils import layer_from_config
#from .base_layer import Layer, Node, InputSpec
#from .input_layer import Input, InputLayer
#from .network import Network, get_source_inputs
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline



data_path = '/home/siva/Desktop/data/'
imgs = np.empty((256, 256), int)
#fs = (12, 12)

#print imgs.shape

filenames = sorted(os.listdir(data_path))
d = [] # vector of classification labels
p=0
for img_name in filenames:
    img = plt.imread(data_path + img_name)
    img  = np.resize(img, (256, 256))    
#    print img_name,":",img.shape, len(img), type(img)
    if p==0:
	imgs=(img)
	p=1
    else:
    	imgs = np.append(imgs, img, axis=0)
#    print(len(imgs))
#    imgs=np.append(imgs,img)
#    print len(imgs)
#    print imgs
#    print imgs.shape
    d.append(int(img_name[1]))
#    print "image: ",type(imgs), type(img) 
#    print "labels: ",d 
#    print imgs
#    d=np.asarray(d)
imgs = np.reshape(imgs, [ 213, 256, 256])
#print(imgs.shape)
#print(len(d))
#imgs = np.reshape(imgs,[214,256])
#print imgs.shape, len(imgs), type(imgs),len(d)
train_images, test_images, train_labels, test_labels = train_test_split(imgs, d, test_size=0.33, random_state=42)


from keras.utils import to_categorical


print('Training data shape : ', train_images.shape, len(train_labels))

print('Testing data shape : ', test_images.shape, len(test_labels))

# Find the unique numbers from the train labels
classes = np.unique(train_labels)

classes=np.append(classes,0)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)
plt.figure(figsize=[4,2])

# Display the first image in training data
plt.subplot(121)
plt.imshow(train_images[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(train_labels[0]))

# Display the first image in testing data
plt.subplot(122)
plt.imshow(test_images[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(test_labels[0]))


# Find the shape of input images and create the variable input_shape
print(train_images.shape[1:])
nRows,nCols = train_images.shape[1:]
nDims = nRows
print(nCols)
train_data = train_images.reshape(train_images.shape[0], nRows, nCols, 1)
test_data = test_images.reshape(test_images.shape[0], nRows, nCols, 1)
input_shape = (nRows, nCols, 1)

# Change to float datatype
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

# Scale the data to lie between 0 to 1
train_data /= 255
test_data /= 255

print(len(train_labels))
print(len(test_labels))
# Change the labels from integer to categorical data
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

#print(type(train_labels_one_hot))
#print(type(train_labels))



# Display the change for category label using one-hot encoding
print('Original label 0 : ', train_labels[0])
print('After conversion to categorical ( one-hot ) : ', train_labels_one_hot[0])

def autoencoder(input_img):
	global encodedd
#	autoencode = Sequential()
#	print(type(encoded))
#	encoded = Dense(256, activation='relu')(input_img)
	encoded = Dense(256, activation='relu')(input_img)
	encoded = Dense(128, activation='relu')(encoded)
	encoded = Dense(32, activation='relu')(encoded)
#	encoded = Dense()
	encodedd = np.resize(encoded,[2097152,1])
        

	decoded = Dense(128, activation='relu')(encoded)
	decoded = Dense(256, activation='relu')(decoded)
#	decoded = Dense(256, activation='relu')(decoded)
	decoded= Dense(1, activation='relu')(decoded)
	print(type(encodedd))
	return encoded

def createModel():
    global encodedd
    global Merge
    model = Sequential()
#    a = Sequential()
    input_img = Input(shape = [256,256,1])
   
    autoencode = Model(input_img, autoencoder(input_img))
#    a.add(autoencode)
#    a.add(Reshape((2097152,1)))
    autoencode.compile(loss='mean_squared_error', optimizer = RMSprop())
#    print(type(autoencode))
#    output = np.resize(autoencode)
#    model2 = Sequential()
#    shape= [None,256,256,32]
    model.add(autoencode)
#    print(type(model))
    # The first two layers with 32 filters of window size 3x3
#    model.add(Conv2D(10, (3, 3), padding='same', activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2,2)))
#    model.add(Conv2D(10, (3, 3), padding='same', activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.6))

#    model.add(Conv2D(10, (3, 3), padding='same', activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.6))

#    model.add(Conv2D(10, (3, 3), padding='same', activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.6))

#    model.add(Conv2D(10, (3, 3), padding='same', activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.6))
#    model.add(Reshape((640,1)))
#    merged_model = Sequential()
#    merged_model.add(Merge([model,a], mode='concat', concat_axis = 1))
#    print(type(model))
#    model.add(Reshape((640,1)))
#    output = np.resize(model,[640,1])
#    print(type(output))
#    print(np.shape(encodedd))
#    out = np.append(output,encodedd)
#    print(type(out))
#    model.add()
    model.add(Flatten())
    print(np.shape(model))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(1))
    model.add(Dense(nClasses, activation='softmax'))
   # model.add(Dropout(0.6))
    return model

#print(np.shape(encodedd))
#print(np.shape(output))
model1 = createModel()
batch_size = 50
epochs = 50
model1.compile(optimizer='rmsprop', loss='categorical_crossentropy')

#print(type(model1))
model1.summary()


#autoencoder.summary()

#writer.add_summary(model2).eval()
print(len(train_labels_one_hot))
history = model1.fit(train_data, train_labels_one_hot, batch_size=batch_size, epochs=epochs, verbose=1,validation_data=(test_data, test_labels_one_hot))
model1.evaluate(test_data, test_labels_one_hot)



plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)





model2 = createModel()

model2.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 256
epochs = 150
datagen = ImageDataGenerator(
#         zoom_range=0.2, # randomly zoom into images
#         rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


# datagen.fit(train_data)

# Fit the model on the batches generated by datagen.flow().
history2 = model2.fit_generator(datagen.flow(train_data, train_labels_one_hot, batch_size=batch_size),
                              steps_per_epoch=int(np.ceil(train_data.shape[0] / float(batch_size))),
                              epochs=epochs,
                              validation_data=(test_data, test_labels_one_hot),
                              workers=4)

model2.evaluate(test_data, test_labels_one_hot)



plt.figure(figsize=[8,6])
plt.plot(history2.history['loss'],'r',linewidth=3.0)
plt.plot(history2.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)



plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)



plt.figure(figsize=[8,6])
plt.plot(history2.history['acc'],'r',linewidth=3.0)
plt.plot(history2.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
