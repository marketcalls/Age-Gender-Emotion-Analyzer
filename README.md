# AGenR (Age & Gender Recognizer)
This repository contains two scripts for training and predicting age and gender using Convolutional Neural Network models.

## Prerequisites
- Python 3.6+
- OpenCV : ```python pip install opencv-python```
- TensorFlow : ```python pip install tensorflow```
- NumPy : ```python pip install numpy```
- pandas : ```python pip install pandas```
- scikit-learn : ```python pip install scikit-learn```
 ## pred.py
This script predicts the age and gender of a person from an image or real-time video using pre-trained deep learning models. It utilizes OpenCV for face detection and TensorFlow for model inference.

### Usage
Run the script from the command line with the following options:

-  Predict from an Image
```python 
python pred.py --image [image_path]
```
* Predict from Real-time Video (Webcam)
```python 
python pred.py
```
The script will open the webcam and continuously predict the age and gender of the people in the video stream. Press the 'q' key to exit the program.

## train.py
This script trains deep learning models for age and gender classification using the Adience Benchmark dataset. It performs data preprocessing, creates TensorFlow datasets, builds and trains the models, and saves the trained models.

### Dataset
To use this script, you need to download the [Adience Benchmark dataset](https://www.kaggle.com/datasets/ttungl/adience-benchmark-gender-and-age-classification) and place it in the _./archive/AdienceBenchmarkGenderAndAgeClassification/_ directory.

### Usage
Run the script from the command line:
``` python
python train.py
```
The script will load the dataset, perform data analysis, preprocess the data, split it into train and validation sets, create TensorFlow datasets, build and train the age and gender models, and save the trained models.

## Models
The script builds and trains two models:

+ Age Model: A convolutional neural network (CNN) model for age classification.
+ Gender Model: A CNN model for gender classification.

The trained models are saved in the _./models/age/_ and _./models/gender/_ directories, respectively.

**For further details, please refer to the code comments within each script.**
