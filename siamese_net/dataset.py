
from PIL import Image
import pandas as pd
import os
import torch
import numpy as np
import cv2
def binarize_signature_image(image_path, threshold_value=128):
    # Read the image
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mean_intensity = np.mean(original_image)
    # Apply thresholding
    _, binary_image = cv2.threshold(original_image, mean_intensity, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pil_image = Image.fromarray(binary_image)
    return pil_image


    return binary_image
class SiameseDataset:
    def __init__(self, training_csv=None, training_dir=None, transform=None):
        # used to prepare the labels and images path
        self.train_df = pd.read_csv(training_csv)
        self.train_df.columns = ["image1", "image2", "label"]
        self.train_dir = training_dir
        self.transform = transform

    def __getitem__(self, index):

        # getting the image path
        image1_path = os.path.join(self.train_dir, self.train_df.iat[index, 0])
        image2_path = os.path.join(self.train_dir, self.train_df.iat[index, 1])

        # Loading the image
       
        img0 = binarize_signature_image(image1_path,200)
        img1 = binarize_signature_image(image2_path,200)
    

        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return (
            img0,
            img1,
            torch.from_numpy(
                np.array([int(self.train_df.iat[index, 2])], dtype=np.float32)
            ),
        )

    def __len__(self):
        return len(self.train_df)