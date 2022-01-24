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


def test_model_4(model, size):
    path = '/home/thiago/PycharmProjects/datasets/IDRI/500/'
    data_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/train.csv')



    test_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/test.csv')
    data_test = idrid_dataset.get_test_dataset_4_classes(path, test_lebel, size)
    run.test(data_test, model)





if __name__ == '__main__':
    #best binary result
    #model = model.get_densenet121_2_classes()
    #test_model(model, 'models/binary/model_binary.pt')


    path = '/home/thiago/PycharmProjects/datasets/IDRI/500/'
    data_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/train.csv')
    class_weights = idrid_dataset.get_weight(data_lebel, n_classes=4)
    train, valid = idrid_dataset.get_data_loader_4_classes(path, data_lebel, 1024)

    for x in [1024]:
        print(x)
        classificador = model.get_densenet121_4_classes()
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
        test_model_4(classificador, x)
        torch.save(classificador.state_dict(), 'models/experimento_1_classes' + str(x) + '.pt')
