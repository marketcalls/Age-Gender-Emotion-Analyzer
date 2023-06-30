import argparse
import yaml
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dropout, Flatten, Dense, MaxPooling2D, Conv2D, Activation, Input
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

def load_images_from_folder(folder):
    images = []
    labels = []
    sub_folders = os.listdir(folder)
    temp = sub_folders
    i=0
    last=[]

    for sub_folder in sub_folders:
        sub_folder_index = temp.index(sub_folder)
        label = sub_folder_index

        if  label in [4, 6]:
            new_label=0 # label = positive emotion
        elif label in [0,5]:
            new_label=1 # label = negative emotion
        else:
            new_label=2 # label = neutral emotion

        path = folder+'/'+sub_folder
        sub_folder_images = os.listdir(path)

        for image in sub_folder_images:
            image_path = path+'/'+image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (48,48))
            images.append(image)
            labels.append(new_label)
            i+=1
        last.append(i)
    return np.array(images), np.array(labels)

def create_model(input_shape, dropout_rate, regularizer_rate):

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
    output = Dense(3, activation="sigmoid")(drop_1)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer="adam", loss=["categorical_crossentropy"], metrics=['accuracy'])
    model.summary()

    return model

def main(config):

    images, labels = load_images_from_folder(config["dir_path"])

    images = images / 255.0

    labels = to_categorical(labels, num_classes=3)

    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.25, random_state=config["random_state"])

    model = create_model((48, 48, 1), config["dropout_rate"], config["regularizer_rate"])

    checkpointer = ModelCheckpoint(config["model_path"], monitor='loss',verbose=1,save_best_only=True,
                               save_weights_only=False, mode='auto',save_freq='epoch')
    
    callback_list=[checkpointer]

    history = model.fit(X_train,Y_train,batch_size=32,validation_data=(X_test,Y_test),epochs=config["epochs"],callbacks=[callback_list])

    print("[+] LOSS = "+str(history.history['loss']))
    print("[+] VAL_LOSS = "+str(history.history['val_loss']))
    print("[+] ACCURACY = "+str(history.history['accuracy']))
    print("[+] VAL_ACCURACY = "+str(history.history['val_accuracy']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configurations')
    parser.add_argument('-c', '--config', help='Path to the configuration file', default='./configs/emotion_config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config)
