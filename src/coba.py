import random
import numpy as np
import torch
from parser import get_parser


def set_seed(seed):
    print("seednya adalah : " + seed)


if __name__ == "__main__":
    args = get_parser()
    set_seed(args["seed"])

"""
# Let's proceed by applying SMOTE to the train_dataset to balance the classes.

# Import necessary libraries for SMOTE
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Load the training dataset (since this was indicated to be exported in the notebook, I will simulate this process here)
train_dataset_path = '/content/drive/MyDrive/Skripsi/Dataset/Subclass/train_dataset_pisah.csv'
test_dataset_path = '/content/drive/MyDrive/Skripsi/Dataset/Subclass/test_dataset_pisah.csv'

# Since I can't load the data from external paths, I'll demonstrate using the assumed `train_dataset` DataFrame
# Here, assuming train_dataset has columns 'TEXT' for features and 'TARGET_LIST' for target labels

# Step 1: Separate features and target
X_train = train_dataset['TEXT']  # Assuming 'TEXT' column is the feature
y_train = train_dataset['TARGET_LIST']  # Assuming 'TARGET_LIST' column is the target

# Step 2: Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train.values.reshape(-1, 1), y_train)

# Output the resampled data dimensions and check balance
X_resampled.shape, y_resampled.shape