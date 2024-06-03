import pandas as pd
import opendatasets as od
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image
import os
import numpy as np

class PCAReductor():
  def __init__(self, n, path_list):
    self.n_comp = n
    self.path_list = path_list

    for p in path_list:
      image = Image.open("")