

# dataloader
import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.parallel as parallel
import torch.optim as optim
import timm
from torch.utils.tensorboard import SummaryWriter
import argparse
from dataloader import *
from models import *
from utils import *


def TrainModel(model_name, data_dir, batch_size, lr_rate, num_epochs,optimizer, criterion, device,num_classes):
    writer = SummaryWriter('logs')
    if device == 'gpu' and not torch.cuda.is_available():
        device = 'cpu'

    train_acc_lst = []
    val_acc_lst = []
    train_loss_lst = []
    val_loss_lst = []


    if args.timm_model:
        model = timm.create_model(model_name, pretrained=args.pretrained) #define model from timm library specifyng pretrained argument with command-line argument
        model.reset_classifier(num_classes=num_classes)
        #timm model architectures are for RGB images so images are converted to RGB 
        data_loader = RGBDataLoader(data_dir, batch_size)
        train_loader = data_loader.train_loader
        val_loader = data_loader.val_loader


    else:
        # Get the model class based on the model name
        model_class = globals()[model_name]
        # Instantiate the model
        model = model_class(num_classes)

        #in custom models Grayscale images are used not to increase computational complexity
        data_loader = GrayscaleDataLoader(data_dir, batch_size)
        train_loader = data_loader.train_loader
        val_loader = data_loader.val_loader
        if args.pretrained:
            checkpoint = torch.load(args.checkpoint_path)
            model_state = checkpoint['model_state']
            model.load_state_dict(model_state)

    model = model.to(device)
    optimizer_class = getattr(optim, args.optimizer)
    optimizer = optimizer_class(model.parameters(), lr=lr_rate)


    if args.train_last_layer:
    # Freeze the pre-trained layers
        for param in model.parameters():
            param.requires_grad = False
    # Make the parameters of the last layer trainable
        for param in model.get_classifier().parameters():
            param.requires_grad = True

    # Get the criterion class based on the criterion name
    criterion_class = getattr(nn, args.criterion)
    criterion = criterion_class()


    model_hypermarameters = {"model_name":model_name, "batch_size" :batch_size,"num_epochs":num_epochs, "loss":criterion, "optimizer":optimizer, "lr_rate":lr_rate, "device":device}
    model_dir_path = model_name
    os.makedirs(model_dir_path, exist_ok=True)  #create directory for the current model
    print(f"Creating model dir at {os.path.abspath(model_dir_path)}")

    #crete directory for the model, create txt file in that file for model to save info of hyperparameters
    save_model_info(model_hypermarameters,model_name, model_dir_path)
    # Training loop
    print("Training of NN")

    # print number of model parameters and number of learnable parameters
    total_params, trainable_params = count_model_parameters(model)
    print(f"Number of total parameters: {total_params}")
    print(f"Number of trainable parameters: {trainable_params}")

    for epoch in range(num_epochs):
        # Initialize the loss and accuracy for the current epoch
        epoch_train_loss = 0.0
        correct = 0
        total = 0

        # Training phase
        model.train()
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            # Forward pass
            outputs = model(data)
            train_loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            if batch_idx % 2 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Training Loss: {train_loss.item():.4f}")

            # Calculate accuracy on training data after each epoch
            epoch_train_loss += train_loss.item()
            _, predicted = torch.max(outputs.data.cpu(), 1)
            total += targets.size(0)
            correct += (predicted == targets.cpu()).sum().item()

        train_accuracy = 100 * correct / total
        train_acc_lst.append(train_accuracy)

        # Calculate the average loss per epoch
        epoch_train_loss /= len(train_loader)
        train_loss_lst.append(epoch_train_loss)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            for (data, targets) in val_loader:
                data, targets = data.to(device), targets.to(device)

                val_outputs = model(data)
                loss = criterion(val_outputs, targets)
                val_loss += loss.item()

                _, predicted = torch.max(val_outputs.data.cpu(), 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets.cpu()).sum().item()

            val_accuracy = 100 * val_correct / val_total
            val_acc_lst.append(val_accuracy)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Training Accuracy: {train_accuracy}")
            print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {val_accuracy}")

            val_loss /= len(val_loader)
            val_loss_lst.append(val_loss)

            writer.add_scalar('Loss/Training', epoch_train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)

            writer.add_scalar('Accuracy/Training', train_accuracy, epoch)
            writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)

            save_checkpoint(model, optimizer, val_accuracy, epoch,model_dir_path)

    # Close the SummaryWriter
    writer.close()
    VisualizeResults(train_acc_lst, val_acc_lst, "Accuracy")
    VisualizeResults(train_loss_lst, val_loss_lst, "Loss")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='FacialEmotionRecognition models training')
    # Add arguments
    parser.add_argument('--model_name', default='CNNModel',help='Name of the model')
    parser.add_argument('--data_dir', required=True, help='Path to the data directory', default = './data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', default='adam',help='Optimizer for training (default: adam)')
    parser.add_argument('--device', default='cuda',choices=['cpu', 'cuda'])
    parser.add_argument('--criterion', help ='Loss function of model')
    parser.add_argument('--pretrained', help='Indicate if the training is done on pretrained model',action="store_true")
    parser.add_argument('--timm_model', help ='Indicate using model from timm library or not',action="store_true")
    parser.add_argument('-last','--train_last_layer', help ='If specified only last layer of pretrained model is trained',action="store_true")
    

    args = parser.parse_args()

    num_classes = get_num_classes(args.data_dir)  #number of emotion classes in our dataset

    TrainModel(args.model_name, args.data_dir, args.batch_size, args.lr_rate, args.num_epochs, args.optimizer, args.criterion, args.device, num_classes)
    
    
