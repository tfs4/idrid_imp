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


def test_final_model(model_binary, model_mc, size_binary, size_mc):
    path = '/home/thiago/PycharmProjects/datasets/IDRI/500/'
    data_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/train.csv')

    test_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/test.csv')
    data_test_binary = idrid_dataset.get_test_dataset_2_classes(path, test_lebel, size_binary)


    #run.test_final(data_test, model)


def test_model_4(model, size):
    path = '/home/thiago/PycharmProjects/datasets/IDRI/500/'
    data_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/train.csv')



    test_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/test.csv')
    data_test = idrid_dataset.get_test_dataset_4_classes(path, test_lebel, size)
    run.test(data_test, model)


def do_binary():
    path = '/home/thiago/PycharmProjects/datasets/IDRI/500/'
    data_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/train.csv')
    class_weights = idrid_dataset.get_weight(data_lebel, n_classes=2)


    x = 1024
    train, valid = idrid_dataset.get_data_loader_2_classes(path, data_lebel, x)
    classificador = model.get_densenet121_2_classes()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(classificador.parameters(), lr=config.LR)

    train_losses, valid_losses = run.optimize(train, valid, classificador, criterion, optimizer, config.EPOCHS)

    epochs = range(config.EPOCHS)
    plt.plot(epochs, train_losses, 'g', label='Training loss')
    plt.plot(epochs, valid_losses, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    test_model_2(classificador, x)
    torch.save(classificador.state_dict(), 'models/experimento_1_classes' + str(x) + '.pt')


def do_mc():
    path = '/home/thiago/PycharmProjects/datasets/IDRI/500/'


    x = 512

    data_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/train.csv')
    data_augmentation = pd.read_csv('augmentation.csv')
    class_weights = idrid_dataset.get_weight(data_lebel, n_classes=4, graph=True, data_aug=data_augmentation)


    train, valid = idrid_dataset.get_data_loader_4_classes(path, data_lebel, x, aug = data_augmentation,  aug_path='augmentation')



    classificador = model.get_densenet121_mc()
    #path_loader = torch.load('models/model_mcpt')
    #classificador.load_state_dict(path_loader)



    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(classificador.parameters(), lr=config.LR)

    print(train)

    train_losses, valid_losses = run.optimize(train, valid, classificador, criterion, optimizer, config.EPOCHS)

    epochs = range(config.EPOCHS)
    plt.plot(epochs, train_losses, 'g', label='Training loss')
    plt.plot(epochs, valid_losses, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    test_model_4(classificador, x)
    torch.save(classificador.state_dict(), 'models/experimento_1_classes' + str(x) + '.pt')


if __name__ == '__main__':
    #do_binary()


    #
    #data = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/train.csv')
    #path = '/home/thiago/PycharmProjects/datasets/IDRI_teste/500/train/'
    #idrid_dataset.do_augmentation(data, path)
    do_mc()

'''
    # binary test
    classificador = model.get_densenet121_2_classes()
    path_loader = torch.load('models/binary/model_binary.pt')
    classificador.load_state_dict(path_loader)
    test_model_2(classificador, 1024)

    #mc test
    classificador = model.get_densenet121_mc()
    path_loader = torch.load('models/mc/model_mc.pt')
    classificador.load_state_dict(path_loader)
    test_model_4(classificador, 512)
'''




