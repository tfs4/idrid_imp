import torch

DEVICE =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 0.0001
BATCH = 4
EPOCHS = 100


