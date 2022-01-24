import torch

DEVICE =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 0.0003
BATCH = 10
EPOCHS = 100