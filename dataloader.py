
import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.parallel as parallel
import torch.optim as optim
import argparse
import PIL.Image as Image



class EmotionsDatasetGrayscale(Dataset):
    def __init__(self, image_folder, labels_tensor):
        self.image_folder = image_folder  # image folder path
        self.labels_tensor = labels_tensor
        self.transform = transforms.Compose ([
            transforms.ToTensor(),  # Convert PIL image to tensor
           #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
            transforms.RandomHorizontalFlip(p=0.5) # p = probability of flip, 0.5 = 50% chance
            ])

    def __len__(self):
        return len(self.labels_tensor)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, f"image_{idx}.png")  # Assuming image names are image_0.jpg, image_1.jpg, ...
        if os.path.exists(img_path):
            image = Image.open(img_path)
            if image is not None:
                image = self.transform(image)
                label = self.labels_tensor[idx]
                return image, label


class EmotionsDatasetRGB(Dataset):
    def __init__(self, image_folder, labels_tensor):
        self.image_folder = image_folder  # image folder path
        self.labels_tensor = labels_tensor
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
            transforms.ToTensor()  # Convert PIL image to tensor
        ])

    def __len__(self):
        return len(self.labels_tensor)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, f"image_{idx}.png")  # Assuming image names are image_0.jpg, image_1.jpg, ...
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")  # Convert grayscale to RGB
            if image is not None:
                image = self.transform(image)
                label = self.labels_tensor[idx]
                return image, label

class RGBDataLoader:
    def __init__(self,data_dir,batch_size):

        train_df = pd.read_csv(os.path.join(data_dir,'train_data.csv'))
        val_df = pd.read_csv(os.path.join(data_dir,'val_data.csv'))
        test_df = pd.read_csv(os.path.join(data_dir,'test_data.csv'))

        train_labels = train_df['emotion'].values
        val_labels = val_df['emotion'].values
        test_labels = test_df['emotion'].values

        TrainDs = EmotionsDatasetRGB(os.path.join("data", "train"), train_labels) 
        self.train_loader = DataLoader(TrainDs,batch_size,shuffle=True)

        ValDs = EmotionsDatasetRGB(os.path.join("data", "val"), val_labels)
        self.val_loader = DataLoader(ValDs,batch_size,shuffle=False)

        TestDs = EmotionsDatasetRGB(os.path.join("data", "test"), test_labels)
        self.test_loader = DataLoader(TestDs,batch_size,shuffle=False)

class GrayscaleDataLoader():
    def __init__(self,data_dir,batch_size):

        train_df = pd.read_csv(os.path.join(data_dir,'train_data.csv'))
        val_df = pd.read_csv(os.path.join(data_dir,'val_data.csv'))
        test_df = pd.read_csv(os.path.join(data_dir,'test_data.csv'))

        train_labels = train_df['emotion'].values
        val_labels = val_df['emotion'].values
        test_labels = test_df['emotion'].values

        TrainDs = EmotionsDatasetGrayscale(os.path.join("data", "train"), train_labels) 
        self.train_loader=DataLoader(TrainDs,batch_size,shuffle=True)

        ValDs = EmotionsDatasetGrayscale(os.path.join("data", "val"), val_labels)
        self.val_loader=DataLoader(ValDs,batch_size,shuffle=False)

        TestDs = EmotionsDatasetGrayscale(os.path.join("data", "test"), test_labels)
        self.test_loader=DataLoader(TestDs,batch_size,shuffle=False)
