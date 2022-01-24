import pandas as pd
import torch
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
torch.manual_seed(0)

import idrid_dataset
import config
import model
import run


def test_model_kaggle(model, model_path):
    path = '/home/thiago/PycharmProjects/datasets/kaggle/500/'


    path_loader = torch.load(model_path)
    model.load_state_dict(path_loader)
    test_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/kaggle/test.csv')
    data_test = idrid_dataset.get_test_dataset_2_classes(path, test_lebel, 512)
    run.test(data_test, model)


def test_model(model, model_path):
    path = '/home/thiago/PycharmProjects/datasets/IDRI/500/'
    data_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/train.csv')


    path_loader = torch.load(model_path)
    model.load_state_dict(path_loader)
    test_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/test.csv')
    data_test = idrid_dataset.get_test_dataset_2_classes(path, test_lebel, 1024)
    run.test(data_test, model)


def test_model_2(model, size):
    path = '/home/thiago/PycharmProjects/datasets/IDRI/500/'
    data_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/train.csv')



    test_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/test.csv')
    data_test = idrid_dataset.get_test_dataset_2_classes(path, test_lebel, size)
    run.test(data_test, model)



if __name__ == '__main__':
  # best result binary 82.52
    model = model.get_densenet121_2_classes()
    test_model(model, 'models/model1024_binary.pt')
