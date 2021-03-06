# -*- coding: utf-8 -*-
"""eslr-classifier-ml.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EAA0eecyZ4oqouvi_dwydWBAa3sEV1uI

## Classifier with machine learning (ESLR Recruitment Project 2019)

## Resources
- https://towardsdatascience.com/an-approach-to-choosing-the-number-of-components-in-a-principal-component-analysis-pca-3b9f3d6e73fe
- https://sdsawtelle.github.io/blog/output/week8-andrew-ng-machine-learning-with-python.html#PCA-in-sklearn
- https://bogotobogo.com/python/scikit-learn/scikit_machine_learning_Data_Compresssion_via_Dimensionality_Reduction_1_Principal_component_analysis%20_PCA.php

## Classification definition

Classification is a supervised learning approach. It learns from a given dataset and apply what it has learnt to classify other inputs.

## Naives Bayes classifier

The Naives Bayes classifier is based on the Bayes Theorem with an assumption of independence among predictors. It is easy to build and is useful in the case of large dataset.
Here is used the Bernoulli variant, which assumes that all classes are binary, either malignant or benign in our case.

## Link Google Drive to retrieve dataset
"""

from google.colab import drive
drive.mount('/content/drive')



"""## Import libraries and initialization
Set path to the directory that contains the ember dataset.
"""

import sys
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_files
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import preprocessing

# path to directory containing the ember directory
path = "/content/drive/My Drive/"

"""## Remove unlabeled vectors and create new file
**Only run this cell if label-stripped dataset does not exist already.**

Change path to the corresponding files if needed.
"""

# Load file in array.
features = np.memmap(path + "ember-dataset/Xtrain.dat", dtype=np.float32, mode='r', shape=(900000, 2351))

# Load label file.
labels = np.memmap(path + "ember-dataset/Ytrain.dat", dtype=np.float32, mode='r')

# Remove unlabeled vectors
unlabeled_index = np.argwhere(labels==-1).flatten()
labels = np.delete(labels, unlabeled_index, 0)
features = np.delete(features, unlabeled_index, 0)

# Generate files from dataset where unlabeled data are removed
# Files are saved to the format npy, use np.load() to load them
np.save(path + "Xtrain_no_unlabeled.npy", features)
np.save(path + "Ytrain_no_unlabeled.npy", labels)

del features
del labels

"""## Load dataset stripped from unlabeled vectors"""

# After stripping only 600000 vectors remain
features = np.load(path + "Xtrain_no_unlabeled.npy", mmap_mode='r+')
features = np.reshape(features, (-1, 2351))
labels = np.load(path + "Ytrain_no_unlabeled.npy", mmap_mode='r+')

"""## Compression using PCA (Principal Component Analysis)
- Test with PCA

- Use of IncrementalPCA

    - allow to compress using minibatches the dataset.

    - partial_fit seems to use less RAM than PCA's fit. However, when getting to transform operation it consumes a lot of RAM.

- Going back to PCA
"""

size = len(features)
# Define PCA and with the dimension to which it needs be reduced to
pca = PCA(n_components=500)
# Allows to center the points
# Tried StandardScaler and MinMaxScaler
scaler = StandardScaler(with_mean=False)
features_scaled = scaler.fit_transform(features)
features = pca.fit_transform(features_scaled)

"""## Split dataset for training and testing"""

# Split dataset into two subsets, one for training, another for testing
# test subset contains a third of the original dataset, train contains the rest
# Dataset is not shuffled before splitting
train, test, train_labels, test_labels = train_test_split(features,
                                                          labels,
                                                          test_size=0.3,
                                                          random_state=1,
                                                          shuffle=False)

print("Train shape:", train.shape)
print("Train_labels shape:", train_labels.shape)
print("Test shape:", test.shape)
print("Test_labels shape:", test_labels.shape)

"""## Build classifier and evaluate performance"""

# Initialize our classifier
# Use Bernoulli distribution as only 2 outputs remain after stripping unlabeled data
# either malignant or benign
gnb = BernoulliNB()
# Train our classifier
for i in range(0, len(train), len(train) // 4):
    train_subset = train[i : i + len(train) // 4]
    train_labels_subset = train_labels[i : i + len(train) // 4]
    gnb.partial_fit(train_subset, train_labels_subset, np.unique(train_labels_subset))

# Make predictions
preds = gnb.predict(test)

# Evaluate accuracy
print("Accuracy :", accuracy_score(test_labels, preds))
