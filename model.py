import torchvision.models as models
import torch
from torch import nn

import config
import random
import numpy as np

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


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


def get_inception_v3(class_weights):
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 5),
    )
    model.cuda()
    torch.nn.CrossEntropyLoss()
    torch.optim.Adam(model.parameters(), lr=config.LR)

    return model