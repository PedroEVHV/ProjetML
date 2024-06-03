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
        self.pca = PCA(n_components=self.n_comp)
        self.path_list = path_list
        self.images = []
        self.n_images = len(path_list)
        for p in range(0, len(path_list)):
            try:
                image = Image.open("gtsrb-german-traffic-sign/" + path_list[p])
                image = np.array(image)
                self.images.append(image)
            except:
                print("An exception occurred")


    # def standardize(self):
    #     self.scaler = StandardScaler()
    #     self.images_scaled = self.scaler.fit_transform(self.images_res)

    def pca_reduction(self):
        self.pca.fit(self.images)
        img_transformed = self.pca.transform(self.images)

    def plot_var(self):
        # Variance expliquée par chaque composante principale
        explained_variance = self.pca.explained_variance_ratio_

        # Visualisation de la variance expliquée
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(explained_variance))
        plt.xlabel('Nombre de composantes')
        plt.ylabel('Variance expliquée cumulée')
        plt.title('Variance expliquée par PCA')
        plt.show()

    def reconstruct_image(self, index):

        images_reconstructed = self.pca.inverse_transform(self.img_transformed)

        plt.figure(figsize=(12, 6))

        # Image originale
        plt.subplot(1, 2, 1)
        plt.imshow(self.images[index], cmap='gray')
        plt.title('Image originale')

        # Image reconstruite
        plt.subplot(1, 2, 2)
        plt.imshow(images_reconstructed[index], cmap='gray')
        plt.title('Image reconstruite')

        plt.show()

    

