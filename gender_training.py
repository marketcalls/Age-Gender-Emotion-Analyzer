import os
import cv2
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Activation, Flatten, Dense, Input
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from sklearn.model_selection import train_test_split

def load_data(path, img_size):
    """
    Load image data from directory, convert images to grayscale and resize.
    Split filename to get gender data.
    """
    pixels, gender = [], []

    for img in os.listdir(path):
        genders = img.split("_")[1]
        img = cv2.imread(str(path)+"/"+str(img))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (img_size, img_size))
        pixels.append(np.array(img))
        gender.append(np.array(genders))

    pixels = np.array(pixels)
    gender = np.array(gender, np.uint64)

    return pixels, gender

def create_model(input_shape, dropout_rate, regularizer_rate):
    """
    Create CNN model for gender prediction.
    """
    input_layer = Input(shape=input_shape)
    conv1 = Conv2D(32, (3, 3), padding='same', strides=(1, 1), kernel_regularizer=l2(regularizer_rate))(input_layer)
    conv1 = Dropout(dropout_rate)(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), padding='same', strides=(1, 1), kernel_regularizer=l2(regularizer_rate))(pool1)
    conv2 = Dropout(dropout_rate)(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), padding='same', strides=(1, 1), kernel_regularizer=l2(regularizer_rate))(pool2)
    conv3 = Dropout(dropout_rate)(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(256, (3, 3), padding='same', strides=(1, 1), kernel_regularizer=l2(regularizer_rate))(pool3)
    conv4 = Dropout(dropout_rate)(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    flatten = Flatten()(pool4)
    dense_1 = Dense(128, activation='relu')(flatten)
    drop_1 = Dropout(dropout_rate)(dense_1)
    output = Dense(2, activation="sigmoid")(drop_1)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer="adam", loss=["sparse_categorical_crossentropy"], metrics=['accuracy'])

    return model

def main(config):
    pixels, gender = load_data(config['dir_path'], config['img_size'])
    x_train, x_test, y_train, y_test = train_test_split(pixels, gender, random_state=config['random_state'])

    model = create_model((config['img_size'], config['img_size'], 1), config['dropout_rate'], config['regularizer_rate'])
    model.summary()

    checkpointer = ModelCheckpoint(config['model_path'], monitor='loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=config['epochs'], callbacks=[checkpointer])

    print("[+] LOSS = "+str(history.history['loss']))
    print("[+] VAL_LOSS = "+str(history.history['val_loss']))
    print("[+] ACCURACY = "+str(history.history['accuracy']))
    print("[+] VAL_ACCURACY = "+str(history.history['val_accuracy']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configurations')
    parser.add_argument('-c', '--config', help='Path to the configuration file', default='./configs/gender_config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config)
