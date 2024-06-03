# Fichier principal

import pandas as pd
import opendatasets as od
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image
import os
import numpy as np


import data
import pca

# https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

od.download("https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")

data_train = pd.read_csv("gtsrb-german-traffic-sign/Train.csv")
data_test = pd.read_csv("gtsrb-german-traffic-sign/Test.csv")
data_meta = pd.read_csv("gtsrb-german-traffic-sign/Meta.csv")

# Reading csv heads
print(data_train.head(10))
print(data_test.head(10))
print(data_meta.head(10))

# Check null values
print(data_train.isnull().any())
print(data_test.isnull().any())
print(data_meta.isnull().any())

# Meta has a null value, we display it
print(data_meta[data_meta.isnull().any(axis=1)])

# Count image number for each classId
compte_classid = data_train['ClassId'].value_counts()

# Display results in graph
plt.figure(figsize=(12, 6))
compte_classid.sort_index().plot(kind='bar')

plt.xlabel('ClassId')
plt.ylabel('Nombre d\'images')
plt.title('Nombre d\'images par type de panneau')
plt.xticks(rotation=90)
plt.show()


# Image to duplicate
data.blackAndWhiteImage('gtsrb-german-traffic-sign/Train/0/00000_00000_00000.png')

image = Image.open('gtsrb-german-traffic-sign/Train/0/00000_00000_00000.png')
image_nb = Image.open('gtsrb-german-traffic-sign/Train/0/00000_00000_00000_bw.png')

# Display image and its duplicate
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Image originale')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Image en noir et blanc')
plt.imshow(image_nb, cmap='gray')
plt.axis('off')

plt.show()

data.resize(["Train/0/00000_00000_00000.png"])