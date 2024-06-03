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

# od.download("https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")

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

# Dossier racine contenant les images
rootImageDirTrain = 'gtsrb-german-traffic-sign/Train'
rootImageDirTest = 'gtsrb-german-traffic-sign/Test'

# Lister toutes les images dans les dossiers
def listAllImages(root_dir):
    image_paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.png'):
                image_paths.append(os.path.join(root.split("/")[1], file).replace('\\', '/'))
    return image_paths

# Vérifier que chaque image dans le dossier est étiquetée
def checkAllImagesLabeled(image_paths, labeled_paths):
    unlabeled_images = []
    labeled_set = set(labeled_paths)  # Convertir les chemins étiquetés en set pour une recherche rapide
    for image_path in image_paths:
        if image_path not in labeled_set:
            unlabeled_images.append(image_path)
    return unlabeled_images

def areImagesLabeled(root_dir, dataset):
    # Obtenir la liste de toutes les images dans les dossiers
    all_image_paths = listAllImages(root_dir)

    # Obtenir la liste des chemins d'images dans le fichier CSV
    csv_image_paths = dataset['Path'].tolist()

    # Vérifier les images non étiquetées
    unlabeled_images = checkAllImagesLabeled(all_image_paths, csv_image_paths)

    # Afficher les résultats
    if len(unlabeled_images) == 0:
        print("Toutes les images dans les dossiers sont bien étiquetées.")
    else:
        print("Les images suivantes ne sont pas étiquetées :")
        for img in unlabeled_images:
            print(img)

areImagesLabeled(rootImageDirTrain, dataset=data_train)
areImagesLabeled(rootImageDirTest, dataset=data_test)

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