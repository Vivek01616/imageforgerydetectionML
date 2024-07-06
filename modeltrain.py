import numpy as np
np.random.seed(2)
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from PIL import Image, ImageChops, ImageEnhance
import os
def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.png'
    
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality = quality)
    temp_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, temp_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image
image_size = (128, 128)
def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0
X = [] # ELA converted images
Y = [] # 0 for fake, 1 for real
import random
path = 'datasets/real'
for dirname, _, filenames in os.walk(path):
     for filename in filenames:
         if filename.endswith('jpg') or filename.endswith('png')or filename.endswith('tif'):
             full_path = os.path.join(dirname, filename)
             X.append(prepare_image(full_path))
             Y.append(1)
             if len(Y) % 500 == 0:
                 print(f'Processing {len(Y)} images')
print(len(X), len(Y))
path = 'datasets/fake'
for dirname, _, filenames in os.walk(path):
     for filename in filenames:
         if filename.endswith('jpg') or filename.endswith('png')or filename.endswith('tif'):
             full_path = os.path.join(dirname, filename)
             X.append(prepare_image(full_path))
             Y.append(0)
             if len(Y) % 500 == 0:
                 print(f'Processing {len(Y)} images')

print(len(X), len(Y))
X = np.array(X)
Y = to_categorical(Y, 2)
X = X.reshape(-1, 128, 128, 3)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state=5)
X = X.reshape(-1,1,1,1)
print(len(X_train), len(Y_train))
print(len(X_val), len(Y_val))
def build_model():
     model = Sequential()
     model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'valid', activation = 'relu', input_shape = (128, 128, 3)))
     model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'valid', activation = 'relu', input_shape = (128, 128, 3)))
     model.add(MaxPool2D(pool_size = (2, 2)))
     model.add(Dropout(0.25))
     model.add(Flatten())
     model.add(Dense(256, activation = 'relu'))
     model.add(Dropout(0.5))
     model.add(Dense(2, activation = 'softmax'))
     return model
model = build_model()
model.summary()
epochs = 30
batch_size = 32
init_learning_rate = 1e-4

optimizer = Adam(learning_rate = init_learning_rate, decay = init_learning_rate/epochs)
model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
early_stopping = EarlyStopping(monitor = 'val_acc',
                               min_delta = 0,
                               patience = 2,
                               verbose = 0,
                               mode = 'auto')
hist = model.fit(X_train,
                  Y_train,
                  batch_size = batch_size,
                  epochs = epochs,
                 validation_data = (X_val, Y_val),
                 callbacks = [early_stopping])
model.save('run1.h5')