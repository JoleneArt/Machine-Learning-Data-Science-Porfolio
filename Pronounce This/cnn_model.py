import pandas as pd
import numpy as np 
import itertools
import keras
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from keras.models import Sequential 
from keras import optimizers
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense 
from keras import applications 
from keras.utils.np_utils import to_categorical 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
%matplotlib inline
import math 
import datetime
import time

img_width, img_height = 224, 224 
 
#Create a bottleneck file
top_model_weights_path = ‘bottleneck_fc_model.h5’
# loading up our datasets
train_data_dir = ‘data/train’ 
validation_data_dir = ‘data/validation’ 
test_data_dir = ‘data/test’
 
epochs = 10
batch_size = 64

vgg16 = applications.VGG16(include_top=False, weights=’imagenet’)
datagen = ImageDataGenerator(rescale=1. / 255) 

generator_top = datagen.flow_from_directory( 
   train_data_dir, 
   target_size=(img_width, img_height), 
   batch_size=batch_size, 
   class_mode=’categorical’, 
   shuffle=False) 
 
nb_train_samples = len(generator_top.filenames) 
num_classes = len(generator_top.class_indices) 
 
train_data = np.load(‘bottleneck_features_train.npy’) 
 
train_labels = generator_top.classes 
 
# convert the training labels to categorical vectors 
train_labels = to_categorical(train_labels, num_classes=num_classes)



#This is the best model we found. For additional models, check out I_notebook.ipynb
start = datetime.datetime.now()
model = Sequential() 
model.add(Flatten(input_shape=train_data.shape[1:])) 
model.add(Dense(100, activation=keras.layers.LeakyReLU(alpha=0.3))) 
model.add(Dropout(0.5)) 
model.add(Dense(50, activation=keras.layers.LeakyReLU(alpha=0.3))) 
model.add(Dropout(0.3)) 
model.add(Dense(num_classes, activation=’softmax’))
model.compile(loss=’categorical_crossentropy’,
   optimizer=optimizers.RMSprop(lr=1e-4),
   metrics=[‘acc’])
history = model.fit(train_data, train_labels, 
   epochs=7,
   batch_size=batch_size, 
   validation_data=(validation_data, validation_labels))
model.save_weights(top_model_weights_path)
(eval_loss, eval_accuracy) = model.evaluate( 
    validation_data, validation_labels, batch_size=batch_size,     verbose=1)
print(“[INFO] accuracy: {:.2f}%”.format(eval_accuracy * 100)) 
print(“[INFO] Loss: {}”.format(eval_loss)) 
end= datetime.datetime.now()
elapsed= end-start
print (‘Time: ‘, elapsed)


#Graphing our training and validation
acc = history.history[‘acc’]
val_acc = history.history[‘val_acc’]
loss = history.history[‘loss’]
val_loss = history.history[‘val_loss’]
epochs = range(len(acc))
plt.plot(epochs, acc, ‘r’, label=’Training acc’)
plt.plot(epochs, val_acc, ‘b’, label=’Validation acc’)
plt.title(‘Training and validation accuracy’)
plt.ylabel(‘accuracy’) 
plt.xlabel(‘epoch’)
plt.legend()
plt.figure()
plt.plot(epochs, loss, ‘r’, label=’Training loss’)
plt.plot(epochs, val_loss, ‘b’, label=’Validation loss’)
plt.title(‘Training and validation loss’)
plt.ylabel(‘loss’) 
plt.xlabel(‘epoch’)
plt.legend()
plt.show()
   

categorical_test_labels = pd.DataFrame(test_labels).idxmax(axis=1)
categorical_preds = pd.DataFrame(preds).idxmax(axis=1)
confusion_matrix= confusion_matrix(categorical_test_labels, categorical_preds)

#To get better visual of the confusion matrix:
def plot_confusion_matrix(cm, classes,
   normalize=False,
   title=’Confusion matrix’,
   cmap=plt.cm.Blues):
 
 ‘’’prints pretty confusion metric with normalization option ‘’’
   if normalize:
     cm = cm.astype(‘float’) / cm.sum(axis=1)[:, np.newaxis]
     print(“Normalized confusion matrix”)
   else:
     print(‘Confusion matrix, without normalization’)
  
   plt.imshow(cm, interpolation=’nearest’, cmap=cmap)
   plt.title(title)
   plt.colorbar()
   tick_marks = np.arange(len(classes))
   plt.xticks(tick_marks, classes, rotation=45)
   plt.yticks(tick_marks, classes)
 
   fmt = ‘.2f’ if normalize else ‘d’
   thresh = cm.max() / 2.
   for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt), horizontalalignment=”center”, color=”white” if cm[i, j] > thresh else “black”)
 
   plt.tight_layout()
   plt.ylabel(‘True label’)
   plt.xlabel(‘Predicted label’) 


def read_image(file_path):
   print(“[INFO] loading and preprocessing image…”) 
   image = load_img(file_path, target_size=(224, 224)) 
   image = img_to_array(image) 
   image = np.expand_dims(image, axis=0)
   image /= 255. 
   return image
        
def test_single_image(path):
  animals = [‘butterflies’, ‘chickens’, ‘elephants’, ‘horses’, ‘spiders’, ‘squirells’]
  images = read_image(path)
  time.sleep(.5)
  bt_prediction = vgg16.predict(images) 
  preds = model.predict_proba(bt_prediction)
  for idx, animal, x in zip(range(0,6), animals , preds[0]):
   print(“ID: {}, Label: {} {}%”.format(idx, animal, round(x*100,2) ))
  print(‘Final Decision:’)
  time.sleep(.5)
               
  for x in range(3):
   print(‘.’*(x+1))
   time.sleep(.2)
  class_predicted = model.predict_classes(bt_prediction)
  class_dictionary = generator_top.class_indices 
  inv_map = {v: k for k, v in class_dictionary.items()} 
  print(“ID: {}, Label: {}”.format(class_predicted[0],  inv_map[class_predicted[0]])) 
  return load_img(path)
path = ‘data/test/yourpicturename’
test_single_image(path)
