import torch.nn as nn
import torch
class AnchorGenerator(nn.module):
    def __init__(self, scale, ratio):
