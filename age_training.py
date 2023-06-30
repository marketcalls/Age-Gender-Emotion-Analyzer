import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, AveragePooling2D, GlobalAveragePooling2D
from keras import utils
from keras.callbacks import TensorBoard, ModelCheckpoint
import re

#GLOBAL VARIABLES 
DATASET_PATH = None
AUGMENTED_DATASET_PATH = None
NUM_CLASSES = None
LOG_DIR = None
MODEL_CHECKPOINT_PATH = None
OUTPUT_MODEL_PATH = None

def create_dataframe_from_directory(directory):
    """Creates dataframes from Images."""
    paths = []
    ages = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        pattern = r'(\d+)_\d+.jpg'
        match = re.search(pattern, filename)
        if match:
            number = int(match.group(1))
            paths.append(file_path)
            ages.append(number)
    data = {'filename': paths, 'age': ages}
    df = pd.DataFrame(data)
    return df

def assign_class_label(age):
    """Return the class label corresponding to the re-distributed 7 age-ranges."""
    if 1 <= age <= 2:
        return 0
    elif 3 <= age <= 9:
        return 1
    elif 10 <= age <= 20:
        return 2
    elif 21 <= age <= 27:
        return 3
    elif 28 <= age <= 45:
        return 4
    elif 46 <= age <= 65:
        return 5
    else:
        return 6

def parse_image(filename, label):
    """Read the image, decode it, and one-hot encode the label."""
    image_string = tf.io.read_file(filename)
    image_decoded = tf.io.decode_jpeg(image_string, channels=1)
    label = tf.one_hot(label, NUM_CLASSES)
    return image_decoded, label

def create_tensors(filenames, labels):
    """Create TensorFlow constants from filenames and labels."""
    filenames_tensor = tf.constant(filenames)
    labels_tensor = tf.constant(labels)
    return filenames_tensor, labels_tensor

def create_model():
    """Create the architecture of the sequential neural network."""
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(200, 200, 1)))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(132, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model

def plot_loss_accuracy(train_loss, test_loss, train_accuracy, test_accuracy):
    """Plot the loss and accuracy values."""
    fig, ax = plt.subplots(ncols=2, figsize=(15,7))
    ax = ax.ravel()
    ax[0].plot(train_loss, label='Train Loss', color='royalblue', marker='o', markersize=5)
    ax[0].plot(test_loss, label='Test Loss', color = 'orangered', marker='o', markersize=5)
    ax[0].set_xlabel('Epochs', fontsize=14)
    ax[0].set_ylabel('Categorical Crossentropy', fontsize=14)
    ax[0].legend(fontsize=14)
    ax[0].tick_params(axis='both', labelsize=12)
    ax[1].plot(train_accuracy, label='Train Accuracy', color='royalblue', marker='o', markersize=5)
    ax[1].plot(test_accuracy, label='Test Accuracy', color='orangered', marker='o', markersize=5)
    ax[1].set_xlabel('Epochs', fontsize=14)
    ax[1].set_ylabel('Accuracy', fontsize=14)
    ax[1].legend(fontsize=14)
    ax[1].tick_params(axis='both', labelsize=12)
    fig.suptitle(x=0.5, y=0.92, t="Lineplots showing loss and accuracy of CNN model by epochs", fontsize=16)
    plt.savefig(f'{LOG_DIR}/final_cnn_loss_accuracy.png', bbox_inches='tight')

def print_score_summary(metrics_names, score):
    """Print the score summary."""
    print(f'CNN model {metrics_names[0]} \t\t= {round(score[0], 3)}')
    print(f'CNN model {metrics_names[1]} \t= {round(score[1], 3)}')

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', export_as='confusion_matrix', cmap=plt.cm.Blues):
    """
    Print and plot the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True labels', fontsize=14)
    plt.xlabel('Predicted labels', fontsize=14)
    plt.savefig(f'{LOG_DIR}/{export_as}.png', bbox_inches='tight')

def main(config):
    
    global DATASET_PATH, AUGMENTED_DATASET_PATH, NUM_CLASSES, LOG_DIR, MODEL_CHECKPOINT_PATH, OUTPUT_MODEL_PATH

    DATASET_PATH = config['DATASET_PATH']
    AUGMENTED_DATASET_PATH = config['AUGMENTED_DATASET_PATH']
    NUM_CLASSES = config['NUM_CLASSES']
    LOG_DIR = config['LOG_DIR']
    MODEL_CHECKPOINT_PATH = config['MODEL_CHECKPOINT_PATH']
    OUTPUT_MODEL_PATH = config['OUTPUT_MODEL_PATH']

    # Importing the augmented training dataset and testing dataset
    train_augmented_df = create_dataframe_from_directory(AUGMENTED_DATASET_PATH)
    test_df = create_dataframe_from_directory(DATASET_PATH)

    train_augmented_df['target'] = train_augmented_df['age'].map(assign_class_label)
    test_df['target'] = test_df['age'].map(assign_class_label)

    train_augmented_filenames = list(train_augmented_df['filename'])
    train_augmented_labels = list(train_augmented_df['target'])
    test_filenames = list(test_df['filename'])
    test_labels = list(test_df['target'])

    train_augmented_filenames_tensor, train_augmented_labels_tensor = create_tensors(train_augmented_filenames, train_augmented_labels)
    test_filenames_tensor, test_labels_tensor = create_tensors(test_filenames, test_labels)

    train_augmented_dataset = tf.data.Dataset.from_tensor_slices((train_augmented_filenames_tensor, train_augmented_labels_tensor))
    train_augmented_dataset = train_augmented_dataset.map(lambda x, y: parse_image(x, y))
    train_augmented_dataset = train_augmented_dataset.batch(512)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames_tensor, test_labels_tensor))
    test_dataset = test_dataset.map(lambda x, y: parse_image(x, y))
    test_dataset = test_dataset.batch(512)

    age_model = create_model()
    age_model.summary()
    age_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    tensorboard_callback = TensorBoard(log_dir=LOG_DIR)
    checkpoint_callback = ModelCheckpoint(filepath=MODEL_CHECKPOINT_PATH,
                                           monitor='val_accuracy',
                                           save_best_only=True,
                                           save_weights_only=False,
                                           verbose=1
                                          )

    age_model_history = age_model.fit(train_augmented_dataset,
                                      batch_size=512,
                                      validation_data=test_dataset,
                                      epochs=12,
                                      callbacks=[tensorboard_callback, checkpoint_callback],
                                      shuffle=False
                                     )

    train_loss = age_model_history.history['loss']
    test_loss = age_model_history.history['val_loss']
    train_accuracy = age_model_history.history['accuracy']
    test_accuracy = age_model_history.history['val_accuracy']

    plot_loss_accuracy(train_loss, test_loss, train_accuracy, test_accuracy)

    age_model = tf.keras.models.load_model(MODEL_CHECKPOINT_PATH)
    age_model_score = age_model.evaluate(test_dataset, verbose=1)

    age_model_metrics_names = age_model.metrics_names
    print_score_summary(age_model_metrics_names, age_model_score)

    age_model.save(f"{OUTPUT_MODEL_PATH}.h5", save_format='h5')

    age_model_predictions = age_model.predict(test_dataset)
    age_model_predictions = age_model_predictions.argmax(axis=-1)
    confusion_mat = confusion_matrix(test_labels, age_model_predictions)

    class_labels = ['1-2', '3-9', '10-20', '21-27', '28-45', '46-65', '66-116']
    plt.figure(figsize=(16,8))
    plot_confusion_matrix(confusion_mat, class_labels, normalize=True,
                          title="Confusion Matrix based on predictions from CNN model",
                          export_as="final_cnn_conf_mat_norm"
                         )
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configurations')
    parser.add_argument('-c', '--config', help='Path to the configuration file', default='./configs/age_config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config)
