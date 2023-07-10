
import pandas as pd
import os
from utils import convert_csv_to_image

def split_csv_files(data_path):
  df = pd.read_csv(os.path.join('data',data_path))
  # Get the 'Usage' column
  usage_column = df[' Usage']

  # Separate the DataFrame based on the unique values in the 'Usage' column
  train_df = df[usage_column == 'Training'].drop(' Usage', axis=1)
  val_df = df[usage_column == 'PublicTest'].drop(' Usage', axis=1)
  test_df = df[usage_column == 'PrivateTest'].drop(' Usage', axis=1)

  # Print the sizes of each split
  print("Train data size:", len(train_df))
  print("Validation data size:", len(val_df))
  print("Test data size:", len(test_df))

  # Save each split DataFrame as a CSV file
  train_df.to_csv('./data/train_data.csv', index=False)
  val_df.to_csv('./data/val_data.csv', index=False)
  test_df.to_csv('./data/test_data.csv', index=False)

  print("DataFrames saved as CSV files successfully.")

if __name__ == "__main__":
  split_csv_files('icml_face_data.csv')
  
  convert_csv_to_image('train_data.csv')  # Example usage for train data
  convert_csv_to_image('test_data.csv')  # Example usage for test data
  convert_csv_to_image('val_data.csv')  # Example usage for validation data