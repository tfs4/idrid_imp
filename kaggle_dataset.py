from torchvision.transforms import transforms
import PIL.Image as Image
from torch.utils.data import ConcatDataset
import torch
from sklearn.utils import class_weight
from torch.utils.data import (
    Dataset,
    DataLoader,
)
import config
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.io import imread
from tqdm import tqdm




class dataset(Dataset):

    def __init__(self, df, data_path, image_transform=None, train=True):  # Constructor.
        super(Dataset, self).__init__()  # Calls the constructor of the Dataset class.
        self.df = df
        self.data_path = data_path
        self.image_transform = image_transform
        self.train = train

    def __len__(self):
        return len(self.df)  # Returns the number of samples in the dataset.

    def __getitem__(self, index):
        image_id = self.df['id_code'][index]

        try:
            image = Image.open(f'{self.data_path}/{image_id}.jpg')  # Image.
        except FileNotFoundError:
            image = Image.open(f'{self.data_path}/{image_id}.jpeg')  # Image.

        if self.image_transform:
            image = self.image_transform(image)  # Applies transformation to the image.

        if self.train:
            label = self.df['level'][index]  # Label.
            return image, label  # If train == True, return image & label.

        else:
            return image  # If train != True, return image.

def get_data_loader_2_classes(path, data_lebel, size):

    test_transform = transforms.Compose([transforms.Resize([size, size]),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                         ])

    data_lebel["level"].replace({2: 1, 3: 1, 4: 1}, inplace=True)
    train_df = data_lebel.reset_index()
    data_set = dataset(train_df, f'{path}train', image_transform=test_transform)

    train_size = int(0.8 * len(data_set))
    val_size = len(data_set) - train_size

    train_set, valid_set = torch.utils.data.random_split(data_set, [train_size, val_size],
                                                         generator=torch.Generator().manual_seed(42))
    train = DataLoader(train_set, batch_size=config.BATCH, shuffle=True)
    valid = DataLoader(valid_set, batch_size=config.BATCH, shuffle=False)

    return train, valid
