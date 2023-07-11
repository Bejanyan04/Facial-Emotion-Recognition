# Facial-Emotion-Recognition
This project is a facial emotion recognition system using machine learning. It includes code for preprocessing the data, training the models, and evaluating their accuracy on test data.
- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Preprocessing](#preprocessing)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Models](#models)

## Introduction

Facial emotion recognition is a task in computer vision that involves detecting and classifying emotions based on facial expressions. This project aims to provide a flexible and customizable solution for training facial emotion recognition models using a variety of architectures.

## Installation
Clone the repository
## Dataset

To use this project, you will need to download the `icml_face_data.csv` file from the official Kaggle website:

- [Download `icml_face_data.csv`](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

After downloading the file, place it in the `data` folder of this repository. The `data` folder should be created at the root directory of the repository if it does not already exist.


## Usage
### Preprocessing

Before training the models, you need to preprocess the data and convert the pixel values from a CSV file into images. Run the preprocess.py script without any arguments to perform this step.
<pre>
python preprocess.py
</pre>
This script will split the data into train, test, and validation sets and create images corresponding to the pixel information in the CSV file.

### Training
To train the facial emotion recognition models, use the train.py script. It supports both manually implemented models and models from the timm library. Specify the required command-line arguments to customize the training process.
<pre>

python train.py --model_name 'vgg19' --data_dir ./data --batch_size 32 --num_epochs 10 --lr_rate 0.001 --optimizer adam --device cuda --criterion &lt;loss_function&gt; --pretrained --timm_model --train_last_layer
</pre>

- `model_name`: Name of the model (default: CNNModel)
- `data_dir`: Path to the data directory (required)
- `batch_size`: Batch size for training (default: 32)
- `num_epochs`: Number of training epochs (default: 10)
- `lr_rate`: Learning rate (default: 0.001)
- `optimizer`: Optimizer for training (default: adam)
- `device`: Device for training (default: cuda)
- `criterion`: Loss function of the model (required)
- `pretrained`: Indicate if the training is done on a pretrained model
- `timm_model`: Indicate using a model from the timm library
- `train_last_layer`: If specified, only the last layer of the pretrained model is trained



### Evaluation
To evaluate the trained models on the test data, use the `eval.py` script. Specify the following command-line arguments:

- `-t` or `--timm_model`: Define if the model is a timm model or a manual model (optional, use `-t` flag to indicate timm model).
- `-n` or `--model_name`: Name of the model (default: CNNModel).
- `-d` or `--data_dir`: Directory of the data (default: ./data).
- `-p` or `--checkpoint_path`: Path to the checkpoint file (default: model_name+checkpoint.pth).

Run the following command to evaluate the models:

<pre>
python eval.py -t --model_name CNNModel --data_dir ./data --checkpoint_path <path_to_checkpoint>
</pre>

## Models

This project supports both manually implemented models and models from the `timm` library. You can specify the model architecture by choosing the appropriate `model_name` argument during training and evaluation. 

Transfer learning is also supported, allowing you to leverage pre-trained models and their learned features. By setting the `pretrained` flag, you can use a pre-trained model for training, fine-tuning, or transfer learning on your facial emotion recognition task. Additionally, if you want to train only the last layer of a pre-trained model, you can use the `train_last_layer` flag.

With this flexibility, you can easily experiment with different architectures, including custom models and pre-trained models, to achieve optimal performance in facial emotion recognition tasks.
