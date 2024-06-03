# Pre traitement

import pandas as pd
import opendatasets as od
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image
import os
import numpy as np


def addLineToCsv(dataset, datasetPath, oldImagePath, newImagePath):
    matched_row = dataset[dataset['Path'] == oldImagePath[oldImagePath.find('Train'):]]
    new_row = {
      'Width': matched_row['Width'].values[0],
      'Height': matched_row['Height'].values[0],
      'Roi.X1': matched_row['Roi.X1'].values[0],
      'Roi.Y1': matched_row['Roi.Y1'].values[0],
      'Roi.X2': matched_row['Roi.X2'].values[0],
      'Roi.Y2': matched_row['Roi.Y2'].values[0],
      'ClassId': matched_row['ClassId'].values[0],
      'Path': newImagePath[newImagePath.find('Train'):]
  }
    with open(datasetPath,'a') as fd:
      fd.write(f"{new_row['Width']},{new_row['Height']},{new_row['Roi.X1']},{new_row['Roi.Y1']},{new_row['Roi.X2']},{new_row['Roi.Y2']},{new_row['ClassId']},{new_row['Path']}\n")
    

def blackAndWhiteImage(path, dataset, datasetPath):
  # Load image
  image = Image.open(path)

  # Convert to b&l
  image_bl = image.convert('L')
  
  # Add new label to image
  path_bl = path.split(".")[0] + '_bw.png'
  addLineToCsv(dataset, datasetPath, path, path_bl)
  
  # Save duplicate
  image_bl.save(path_bl)


def resize(path_list):
  for p in path_list:
    image = Image.open("gtsrb-german-traffic-sign/" + p)
    new_size = (image.width * 2, image.height * 2)
    resized_image = image.resize(new_size, Image.LANCZOS)
    path_bl = p.split(".")[0] + '_resized.png'
    resized_image.save("gtsrb-german-traffic-sign/" + path_bl)