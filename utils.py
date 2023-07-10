
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt


import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def get_num_classes(data_dir):
    #return number of emotion classes 
    file_path  = os.path.join(data_dir,'icml_face_data.csv')
    data = pd.read_csv(file_path)
    labels = data['emotion'].values
    return len(np.unique(labels))

def count_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params



def save_model_info( hyperparameters, model_name, model_dir):
    # Create the directory for the model if it doesn't exist

   # model_dir = os.path.join(checkpoint_dir, model_name)


    # Save the model parameters
   # checkpoint_path = os.path.join(model_name, "checkpoint.pth")
    #torch.save(model.state_dict(), checkpoint_path)

    # Save the model information and hyperparameters
    info_path = os.path.join(model_name, "model_info.txt")
    with open(info_path, 'w') as file:
        file.write(f"Model: {model_name}\n\n")
        file.write("Hyperparameters:\n")
        for key, value in hyperparameters.items():
            file.write(f"{key}: {value}\n")

def save_checkpoint(model, optimizer, val_acc, epoch, directory):
    checkpoint_path = os.path.join(directory, 'checkpoint.pth')

    if not os.path.isfile(checkpoint_path) or val_acc > get_best_val_acc(directory):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_acc': val_acc,
            'epoch': epoch
        }

        torch.save(state, checkpoint_path)
        print(f"Checkpoint saved in '{checkpoint_path}'.")

def get_best_val_acc(directory):
    checkpoint_path = os.path.join(directory, 'checkpoint.pth')

    if not os.path.isfile(checkpoint_path):
        return float('-inf')  # Return negative infinity if the checkpoint file doesn't exist

    checkpoint = torch.load(checkpoint_path)
    return checkpoint['val_acc']

def VisualizeResults(train_lst, val_lst, metric):
  plt.title("Training and Validation "+ metric)
  plt.plot(val_lst,label="val")
  plt.plot(train_lst,label="train")
  plt.xlabel("iterations")
  plt.ylabel(metric)
  plt.legend()
  plt.show()

def convert_csv_to_image(csv_path):
    # Read the CSV file
    data = pd.read_csv(os.path.join('./data',csv_path))
    # Determine the directory based on the CSV file name
    directory = './data/'
    if 'train' in csv_path.lower():
        directory +=  'train/'
    elif 'test' in csv_path.lower():
        directory += 'test/'
    elif 'val' in csv_path.lower():
        directory += 'val/'

    # Iterate over the rows of the CSV file
    for index, row in data.iterrows():
        pixel_values = [int(value) for value in row.values[1].split()]
        # Reshape the pixel values into a 2D NumPy array
        pixels = np.array(pixel_values)
        sqrt_size = int(np.sqrt(pixels.shape[0]))
        pixels = pixels.reshape((sqrt_size, sqrt_size))

        # Create an image from the pixel values
        image = Image.fromarray(np.uint8(pixels))
        os.makedirs(directory, exist_ok=True)

        # Save the image with a unique filename in the appropriate directory
        image_path = os.path.join(directory, 'image_{}.png'.format(index))
        image.save(image_path)

