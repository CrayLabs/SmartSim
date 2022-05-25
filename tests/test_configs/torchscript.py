# import torch


def average(x):
    return torch.tensor(torch.mean(x)).unsqueeze(0)
