# Pre traitement

import pandas as pd
import opendatasets as od
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image
import os
import numpy as np


def blackAndWhiteImage(path):
  # Load image
  image = Image.open(path)

  # Convert to b&l
  image_bl = image.convert('L')

  # Save duplicate
  path_bl = path.split(".")[0] + '_bw.png'
  image_bl.save(path_bl)


def resize(path_list):
  for p in path_list:
    image = Image.open("gtsrb-german-traffic-sign/" + p)
    new_size = (image.width * 2, image.height * 2)
    resized_image = image.resize(new_size, Image.LANCZOS)
    path_bl = p.split(".")[0] + '_resized.png'
    resized_image.save("gtsrb-german-traffic-sign/" + path_bl)