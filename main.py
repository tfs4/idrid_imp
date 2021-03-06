import pandas as pd
import torch
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
torch.manual_seed(0)

import idrid_dataset, kaggle_dataset
import config
import model
import run



def do_3_last_classes():
    path = '/home/thiago/PycharmProjects/datasets/IDRI/500/'


    x = 1024

    data_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/train.csv')
    data_augmentation = pd.read_csv('augmentation.csv')
    class_weights = idrid_dataset.get_weight(data_lebel, n_classes=-3, graph=True, data_aug=data_augmentation)

    train, valid = idrid_dataset.get_data_loader_3_last_classes(path, data_lebel, x, aug = data_augmentation,  aug_path='augmentation')
    classificador = model.get_densenet121_last_3()


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
    test_model_3_last(classificador, x)
    torch.save(classificador.state_dict(), 'models/experimento_1_classes' + str(x) + '.pt')

def do_3_classes():
    path = '/home/thiago/PycharmProjects/datasets/IDRI/500/'
    data_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/train.csv')
    data_augmentation = pd.read_csv('augmentation.csv')

    class_weights = idrid_dataset.get_weight(data_lebel, n_classes=3, data_aug=data_augmentation)
    x = 1024
    train, valid = idrid_dataset.get_data_loader_3_classes(path, data_lebel, x, aug = data_augmentation,  aug_path='augmentation')
    classificador = model.get_densenet121_3_classes()



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
    test_model_3(classificador, x)
    torch.save(classificador.state_dict(), 'models/experimento_3_classes' + str(x) + '.pt')


def do_binary():
    path = '/home/thiago/PycharmProjects/datasets/IDRI/300/'
    data_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/train.csv')
    class_weights = idrid_dataset.get_weight(data_lebel, n_classes=2)
    x = 1024
    train, valid = idrid_dataset.get_data_loader_2_classes(path, data_lebel, x)
    #classificador = model.get_densenet121_2_classes()
    classificador = model.get_densenet201_2_classes()


    path_loader = torch.load('models/experimento_1_classes1024.pt')
    classificador.load_state_dict(path_loader)


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



def test_kaggle_model_2(model, size):
    path = '/home/thiago/PycharmProjects/datasets/kaggle/500/'
    data_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/kaggle/train.csv')

    test_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/kaggle/test.csv')
    data_test = idrid_dataset.get_test_dataset_2_classes(path, test_lebel, size)
    run.test(data_test, model)




def test_full_models_3_classes(model_binary, model_mc):
    path = '/home/thiago/PycharmProjects/datasets/IDRI/500/'
    data_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/train.csv')

    test_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/test.csv')
    data_test_binary = idrid_dataset.get_test_dataset(path, test_lebel, 1024)
    data_test_mc = idrid_dataset.get_test_dataset(path, test_lebel, 512)

    run.test_full(data_test_binary, data_test_mc, model_binary, model_mc)


def test_full_models(model_binary, model_mc, limit, size_1, size_2):
    path = '/home/thiago/PycharmProjects/datasets/IDRI/500/'
    data_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/train.csv')

    test_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/test.csv')
    data_test_binary = idrid_dataset.get_test_dataset(path, test_lebel, size_1)
    data_test_mc = idrid_dataset.get_test_dataset(path, test_lebel, size_2)

    run.test_full(data_test_binary, data_test_mc, model_binary, model_mc, limit)


def test_model_2(model, size):
    path = '/home/thiago/PycharmProjects/datasets/IDRI/500/'
    data_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/train.csv')

    test_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/test.csv')
    data_test = idrid_dataset.get_test_dataset_2_classes(path, test_lebel, size)
    run.test(data_test, model)

def test_model_3(model, size):
    path = '/home/thiago/PycharmProjects/datasets/IDRI/500/'
    data_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/train.csv')

    test_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/test.csv')
    data_test = idrid_dataset.get_test_dataset_3_classes(path, test_lebel, size)
    run.test(data_test, model)


def test_model_3_last(model, size):
    path = '/home/thiago/PycharmProjects/datasets/IDRI/500/'
    data_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/train.csv')

    test_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/test.csv')
    data_test = idrid_dataset.get_test_dataset_last_3(path, test_lebel, size)
    run.test(data_test, model)



def test_model_4(model, size):
    path = '/home/thiago/PycharmProjects/datasets/IDRI/500/'
    data_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/train.csv')

    test_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/test.csv')
    data_test = idrid_dataset.get_test_dataset_4_classes(path, test_lebel, size)
    run.test(data_test, model)






############################ kaggle


def do_kaggle_binary():
    path = '/home/thiago/PycharmProjects/datasets/kaggle/500/'
    data_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/kaggle/train.csv')
    class_weights = idrid_dataset.get_weight(data_lebel, n_classes=2, graph=True)

    x = 1024
    train, valid = kaggle_dataset.get_data_loader_2_classes(path, data_lebel, x)
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



'''
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
'''



if __name__ == '__main__':

    do_binary()


'''
    classificador_first = model.get_densenet121_3_classes()
    path_loader = torch.load('models/3_first_classes/experimento_3_classes1024.pt')
    classificador_first.load_state_dict(path_loader)

    classificador_last = model.get_densenet121_last_3()
    path_loader = torch.load('models/3_lasts_classes/experimento_1_classes1024.pt')
    classificador_last.load_state_dict(path_loader)

    test_model_3(classificador_first, 1024)
    test_model_3_last(classificador_last, 1024)
    test_full_models(classificador_first, classificador_last, 2, 1024, 1024)

    # binary test
    classificador_binary = model.get_densenet121_2_classes()
    path_loader = torch.load('models/binary/model_binary.pt')
    classificador_binary.load_state_dict(path_loader)

    classificador_mc = model.get_densenet121_mc()
    path_loader = torch.load('models/mc/model_mc.pt')
    classificador_mc.load_state_dict(path_loader)

    test_model_2(classificador_binary, 1024)
    test_model_4(classificador_mc, 512)
    test_full_models(classificador_binary, classificador_mc, 1, 1024, 512)

'''




