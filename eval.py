
import torch
import numpy as np
import pandas as pd 
import timm
from dataloader import *
from utils import get_num_classes
import os


def EvalTrainedModel(model_name,is_timm_model, data_dir):
  if args.checkpoint_path is None:
        # If checkpoint is not specified, use the default name
    checkpoint_path = os.path.join(model_name,'checkpoint.pth')
  else:
    checkpoint_path = args.checkpoint_path

  num_classes = get_num_classes(args.data_dir)  

  if is_timm_model:
    data_loader = RGBDataLoader(data_dir, batch_size=1)
    model = timm.create_model(model_name, pretrained='False') 
    model.reset_classifier(num_classes=num_classes)

  else:
    data_loader = GrayscaleDataLoader(data_dir, batch_size=1)

    model_class = globals()[args.model_name]
    model = model_class(num_classes)
  
  model.eval()
  with torch.no_grad():
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    test_loader = data_loader.test_loader
    for input, targets in test_loader:
      test_output = model(input)
      _, predicted =  torch.max(test_output.cpu(), axis=1)
      test_total = targets.size(0)
      test_correct = (predicted == targets.cpu()).sum().item()
    test_accuracy = test_correct/ test_total *100
    print("accuracy of model on test data")
    print(test_accuracy)
    #return test_accuracy
  
if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='FacialEmotionRecognition models trained')

  # Add arguments
  parser.add_argument('-t','--timm_model', help="Define if the model is timm or manual model", action="store_true")
  parser.add_argument('-n','--model_name', default='CNNModel',help='Name of the model')
  parser.add_argument('-d','--data_dir',help="directory of data",default ='./data')
  parser.add_argument('-p','--checkpoint_path', default=None,
                        help='Path to the checkpoint file (default: model_name+checkpoint.pth)')
    
  args = parser.parse_args()


  EvalTrainedModel(args.model_name, args.timm_model, args.data_dir)