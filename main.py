import os, pathlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D, Dense, BatchNormalization


# Initializing paths for general dataset and fake & real faces
realFaces = pathlib.Path.cwd().parent /"FakeFace_NewModel_TesnorFlow" / "real_and_fake_face" / "training_real"
fakeFaces = pathlib.Path.cwd().parent /"FakeFace_NewModel_TesnorFlow" / "real_and_fake_face" / "training_fake"
realFacesPath, fakeFacesPath = os.listdir(realFaces), os.listdir(fakeFaces)
dataPath = pathlib.Path.cwd().parent / "FakeFace_NewModel_TesnorFlow" / "real_and_fake_face"

# For training purposes we transform, rescale images
transformImages = ImageDataGenerator(horizontal_flip=True, vertical_flip=False, rescale=1. / 255, validation_split=0.2)
train = transformImages.flow_from_directory(dataPath, class_mode="binary", target_size=(96, 96), batch_size=32, subset="training")
validate = transformImages.flow_from_directory(dataPath, class_mode="binary", target_size=(96, 96), batch_size=32, subset="validation")

# We use mobileNetV2 library to extract features from image, in previous model we used 64 X 64 shape for image
# here we use 94 X 94 for a better performance
mobilenetV2 = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')

# We stack layers in sequential architecture
model = Sequential([mobilenetV2, GlobalAveragePooling2D(), Dense(256, activation=tf.nn.relu), BatchNormalization(), Dropout(0.2), Dense(2, activation=tf.nn.softmax)])
model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# This is a "scheduler" which decreases a learning rate after 10 epochs and prints a learning rate
def trainingProcess(epoch):
    if epoch <= 2:
        return 0.001
    elif epoch > 2 and epoch <= 15:
        return 0.0001
    else:
        return 0.00001


# Learning Rate callbakcs
lrCallbacks = tf.keras.callbacks.LearningRateScheduler(trainingProcess)
# we can choose # of epochs to run
storedModel = model.fit_generator(train, epochs=20, callbacks=[lrCallbacks], validation_data=validate)

validationLoss, trainLoss = storedModel.history['val_loss'], storedModel.history['loss']
validationAccuracy, trainAccuracy = storedModel.history['val_accuracy'], storedModel.history['accuracy']


# Plot the figures
axisLength = range(20)
plt.figure()
plt.plot(axisLength, trainLoss)
plt.plot(axisLength, validationLoss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train loss and Validation loss')
plt.grid(True)
plt.legend(['Trained', 'Validation'])
plt.show()

plt.figure()
plt.plot(axisLength, trainAccuracy)
plt.plot(axisLength, validationAccuracy)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train accuracy and Validation accuracy')
plt.grid(True)
plt.legend(['Train', 'Validation'], loc=4)
plt.show()
