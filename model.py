import torchvision.models as models
import torch
from torch import nn

import config
import random
import numpy as np

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


def get_densenet121_4_classes():
    densenet121 = models.densenet121(pretrained=True)

    for param in densenet121.parameters():
        param.requires_grad = False

    densenet121.classifier = nn.Sequential(
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(32, 4),
    )
    densenet121.cuda()

    return densenet121

def get_densenet121_2_classes():
    densenet121 = models.densenet121(pretrained=True)

    for param in densenet121.parameters():
        param.requires_grad = False

    densenet121.classifier = nn.Sequential(
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2),
    )
    densenet121.cuda()

    return densenet121

def get_densenet121(class_weights):
    densenet121 = models.densenet121(pretrained=True)

    for param in densenet121.parameters():
        param.requires_grad = False

    densenet121.classifier = nn.Sequential(
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 5),
    )
    densenet121.cuda()
    torch.nn.CrossEntropyLoss()
    torch.optim.Adam(densenet121.parameters(), lr=config.LR)

    return densenet121





def get_vgg11_binary():
    model = models.vgg11(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(

        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(4096, 1024),
        nn.ReLU(),
        nn.Dropout(0.1),

        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.1),


        nn.Linear(512, 2),


    )
    model.cuda()

    return model



def get_vgg11_mc():
    model = models.vgg11(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(

        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(4096, 1024),
        nn.ReLU(),
        nn.Dropout(0.1),

        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.1),


        nn.Linear(512, 4),


    )
    model.cuda()

    return model
