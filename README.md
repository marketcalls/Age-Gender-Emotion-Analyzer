# AGenEmozer (Age, Gender, and Emotion Analyzer)

This repository contains code for training models to recognize age, gender, and emotion from images and analyzing faces from images or a webcam feed using these models implemented using CNN-keras.

## Description

The project includes four main Python scripts:

1. **age_training.py**: This script is for training a Convolutional Neural Network (CNN) model to recognize different age groups from images. The trained model is saved as 'age_model.h5'.

2. **gender_training.py**: This script is for training a CNN model to recognize gender from images. The trained model is saved as 'gender_model.h5'.

3. **emotion_training.py**: This script is for training a CNN model to recognize three classes of emotions from images: positive, negative, and neutral. The trained model is saved as 'emotion_model.h5'.

4. **AGenEmozer.py**: This script uses three pre-trained models to analyze faces in images or from a webcam feed. It predicts the age, gender, and emotion for each detected face.

## Installation

1. Clone this repository.
```bash
git clone https://github.com/Tejarsha-Arigila/Age-Gender-Emotion-Analyzer.git
```

2. Install required Python packages.
```bash
pip install -r requirements.txt
```

## Usage

### Training Datasets

Download the datasets and place them under _./data/_ folder before training. 

1. *Age* : [Merged & Augmented UTK Faces & Facial Age Dataset](https://www.kaggle.com/datasets/skillcate/merged-augmented-utk-faces-facial-age-dataset)
2. *Gender* : [UTKFace](https://susanqq.github.io/UTKFace/)
3. *Emotion* : [CK + 48](https://drive.google.com/drive/folders/1YEOBooxcTI4H8sIXhZ69sKPGhVRFDl_5?usp=drive_link)

### Train the Models

Each training script accepts a configuration file as an argument. These configuration files are expected to be in YAML format and specify details like the directory of the training data, the model path, the number of epochs for training, and other parameters.

1. Run the `age_training.py` script.

```bash
python age_training.py --config ./configs/age_config.yaml
```

2. Run the `gender_training.py` script.

```bash
python gender_training.py --config ./configs/gender_config.yaml
```

3. Run the `emotion_training.py` script.

```bash
python emotion_training.py --config ./configs/emotion_config.yaml
```

### Use the Age, Gender, and Emotion Analyzer

Run the `AGenEmozer.py` script. 

- To analyze an image, provide the image path as an argument:
```bash
python AGenEmozer.py --image <image_path.jpg>
```

- To analyze the webcam feed, run the script without the image argument:
```bash
python AGenEmozer.py
```

## Note

Please ensure that the models 'age_model.h5', 'gender_model.h5', and 'emotion_model.h5' are present in the root directory of the project. You can train these models using the provided scripts.

## Contribution

Contributions are welcome! Please create an issue for discussion before submitting a pull request.

## License

This project is licensed under the MIT License. Please see the [LICENSE](LICENSE) file for details.
