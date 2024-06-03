import pandas as pd
import opendatasets as od
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image
import os
import numpy as np
import csv
import pca

class PathListGenerator():
    def __init__(self, csv_file):
        self.file = csv_file
        self.csv_lines = []
        self.path_list = []
        with open(self.file, 'r') as f:
            line = csv.reader(f)
            data = list(line)
            self.csv_lines.extend(line)
            for i in range(1, len(data)):
                self.path_list.append(data[i][-1])
# pedrovanhylckama
# 71b9da1af5b7f28a485d5ddba5f6f268
od.download("https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
r = PathListGenerator("gtsrb-german-traffic-sign/Train.csv")
p = pca.PCAReductor(10, r.path_list)
p.pca_reduction()
p.reconstruct_image()