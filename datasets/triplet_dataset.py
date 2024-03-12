
from PIL import Image
import pandas as pd
import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision.transforms.functional import pad


class SquarePad(torch.nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return pad(image, padding, 255, 'constant')

def binarize_signature_image(image_path, threshold_value=128):
    # Read the image
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mean_intensity = np.mean(original_image)
    # Apply thresholding
    _, binary_image = cv2.threshold(original_image, mean_intensity, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_bgr = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    pil_image = Image.fromarray(img_bgr)
    return pil_image


    return binary_image
class SiameseDataset(Dataset):
    def __init__(self, training_csv=None, training_dir=None, transform=None):
        # used to prepare the labels and images path
        self.train_df = pd.read_csv(training_csv)
        self.train_df.columns = ["anchor", "positive", "negative"]
        self.train_dir = training_dir
        self.transform = transform

    def __getitem__(self, index):

        # getting the image path
        image1_path = os.path.join(self.train_dir, self.train_df.iat[index, 0])
        image2_path = os.path.join(self.train_dir, self.train_df.iat[index, 1])
        image3_path = os.path.join(self.train_dir, self.train_df.iat[index, 2])

        # Loading the image
       
        img0 = binarize_signature_image(image1_path,200)
        img1 = binarize_signature_image(image2_path,200)
        img2 = binarize_signature_image(image3_path,200)
    

        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return (
            img0,
            img1,
            img2,
        )

    def __len__(self):
        return len(self.train_df)