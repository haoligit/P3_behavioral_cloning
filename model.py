import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle

from keras.models import Sequential, Model
from keras.layers import Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, Input, Lambda, SpatialDropout2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import cv2

# define batch size and epochs
BATCH_SIZE = 64
EPOCHS = 20
# improt image data
DATA_FOLDER = "./data/data"
DATA_FILE = "{}/driving_log.csv".format(DATA_FOLDER)

data = []
with open(DATA_FILE, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        data.append(line)
# remove row 0
data.pop(0)

# split train and validation set randomly
train_data, valid_data = train_test_split(data, test_size=0.2)
# print(np.shape(train_data), np.shape(valid_data))

# for training set, use 2x center camera, 1x left camera and 1x right camera
train_image_file = []
train_steering = []
steering_correction = 0.25 # steering correction for left/right camera

for row in train_data:
    train_image_file.append(row[0]) # center 1x
    train_steering.append(float(row[3])) # 
    train_image_file.append(row[1]) # left 1x
    train_steering.append(float(row[3]) + steering_correction) # 
    train_image_file.append(row[2]) # right 1x
    train_steering.append(float(row[3]) - steering_correction) # 
    train_image_file.append(row[0]) # center 1x
    train_steering.append(float(row[3])) # 
train_image_file, train_steering = shuffle(train_image_file, train_steering)

# for validation set, only use center camera
valid_image_file = []
valid_steering = []
for row in valid_data:
    valid_image_file.append(row[0]) # center only
    valid_steering.append(float(row[3])) #  
valid_image_file, valid_steering = shuffle(valid_image_file, valid_steering)

# image augment by shifing up and down randomly
def trans_image(image, steer, x_trans_range = 100, y_trans_range = 40):
    """
    Translation function provided by Vivek Yadav

    """
    # Translation
    tr_x = x_trans_range*np.random.uniform()-x_trans_range/2
    steer_ang = steer + tr_x/x_trans_range*2*.2
    tr_y = y_trans_range*np.random.uniform()-y_trans_range/2
    # tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(np.shape(image)[1], np.shape(image)[0]))
    return image_tr,steer_ang

def generator(image_files, steerings, batch_size=32, image_aug_en=1):
    num_samples = len(image_files)
    while 1: # Loop forever so the generator never terminates
        image_files, steerings = shuffle(image_files, steerings)
        for offset in range(0, num_samples, batch_size):
            batch_image_files = image_files[offset:offset+batch_size]
            batch_steerings = steerings[offset:offset+batch_size]
            images = []
            angles = []
            for i in range(len(batch_image_files)):
                name = "{}/{}".format(DATA_FOLDER, batch_image_files[i].strip())
                image = plt.imread(name)[40:135, :] # crop image to 320*, 40 pixels on top, and 25 pixels on bottom (320*160)
                angle = batch_steerings[i]
                if image_aug_en == 1: 
                    image, angle = trans_image(image, angle) # shift image and angle randomly
                    num_random = np.random.randint(1)
                    if (num_random  == 0):# flip the image with 50% possibility
                        image = np.fliplr(image)
                        angle = -1*angle
                images.append(image)
                angles.append(angle)
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            
            if 0: # for debugging
                print(name)
                print(np.shape(image))
                print(X_train.shape)
                if image_aug_en == 1: 
                    print(num_random)
                plt.figure()
                plt.imshow(image)
                plt.show
            yield shuffle(X_train, y_train)
# compile and train the model using the generator function
train_generator = generator(train_image_file, train_steering, batch_size=BATCH_SIZE, image_aug_en=1) # augmentation for training set
valid_generator = generator(valid_image_file, valid_steering, batch_size=BATCH_SIZE, image_aug_en=0) # no augmentation for validation set

# define models
IMAGE_SHAPE = (95, 320, 3)
def nvidia_model():
    """
    Model based on Nvidia paper
    http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    """
    # shape = (img[0], img[1], 3)
    model = Sequential()
    def image_resize(img):
        import tensorflow as tf
        img = tf.image.resize_images(img, (66, 200))
        return img
    model.add(Lambda(image_resize, input_shape = IMAGE_SHAPE))
    model.add(Lambda(lambda x: x/255.-0.5))
    model.add(Convolution2D(24, 5, 5, border_mode="same", subsample=(2,2), activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(36, 5, 5, border_mode="same", subsample=(2,2), activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(48, 5, 5, border_mode="valid", subsample=(2,2), activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
    model.add(SpatialDropout2D(0.2))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation="elu"))
    model.add(Dense(50, activation="elu"))
    model.add(Dense(10, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

def simplified_model():
    
    model = Sequential()
    def image_resize(img):
        import tensorflow as tf
        img = tf.image.resize_images(img, (64, 64))
        return img
    model.add(Lambda(image_resize, input_shape = IMAGE_SHAPE))
    model.add(Lambda(lambda x: x/255.-0.5))
    model.add(Convolution2D(3, 1, 1, border_mode="valid", subsample=(1,1), activation="elu"))
    
    model.add(Convolution2D(64, 3, 3, border_mode="valid", subsample=(1,1), activation="elu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(32, 3, 3, border_mode="valid", subsample=(1,1), activation="elu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(16, 3, 3, border_mode="valid", subsample=(1,1), activation="elu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="elu"))
    model.add(Dense(16, activation="elu"))
    model.add(Dense(1))
    return model

# select model
model = simplified_model()
# model = nvidia_model()
model.summary()
# compile and fit the model
model.compile(loss='mse', optimizer='adam')
history_obj = model.fit_generator(train_generator, samples_per_epoch=len(train_image_file), validation_data=valid_generator, nb_val_samples=len(valid_image_file), nb_epoch=EPOCHS)
model.save('model.h5')
print("My model saved!")
