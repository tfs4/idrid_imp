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
    #classificador = model.get_densenet121_last_3()
    classificador = model.get_vgg_last_3()

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
    class_weights = idrid_dataset.get_weight(data_lebel, n_classes=3)
    x = 1024
    train, valid = idrid_dataset.get_data_loader_3_classes(path, data_lebel, x)
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
    classificador = model.get_densenet121_2_classes()

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



def test_full_models(model_binary, model_mc, size):
    path = '/home/thiago/PycharmProjects/datasets/IDRI/500/'
    data_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/train.csv')

    test_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/test.csv')
    data_test = idrid_dataset.get_test_dataset(path, test_lebel, size)

    run.test_full(data_test, model_binary, model_mc)


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
   #do_kaggle_binary()
   #classificador = model.get_densenet121_2_classes()
   #path_loader = torch.load('models/kaggle_no_agumentation.pt')
   #classificador.load_state_dict(path_loader)
   #test_kaggle_model_2(classificador, 1024)



    #data = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/train.csv')
    #path = '/home/thiago/PycharmProjects/datasets/IDRI_teste/500/train/'
    #idrid_dataset.do_augmentation(data, path)
    #do_mc()


    # binary test
    classificador_binary = model.get_densenet121_2_classes()
    path_loader = torch.load('models/binary/model_binary.pt')
    classificador_binary.load_state_dict(path_loader)

    classificador_mc = model.get_densenet121_mc()
    path_loader = torch.load('models/mc/model_mc.pt')
    classificador_mc.load_state_dict(path_loader)

    test_full_models(classificador_binary, classificador_mc, 1024)
    #test_model_2(classificador, 1024)




    #do_3_last_classes()
    #classificador = model.get_densenet121_last_3()
    #path_loader = torch.load('models/experimento_1_classes1024.pt')
    #classificador.load_state_dict(path_loader)
    #test_model_3_last(classificador, 1024)

    #mc test
    #classificador = model.get_densenet121_mc()
    #path_loader = torch.load('models/mc/model_mc.pt')
    #classificador.load_state_dict(path_loader)
    #test_model_4(classificador, 512)





