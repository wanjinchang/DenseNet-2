import torch
from torch import nn
import torch.nn.parallel
from torch.autograd import Variable

from model import DenseNet
from data_loader import get_loader

class Trainer(object):
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader

        self.num_gpu = config.num_gpu
        self.lr = config.lr
        self.momentum = config.momentum
        self.batch_size = config.batch_size

        # build model
        self.model = DenseNet()

        if self.num_gpu > 0:
            self.model.cuda()

        if self.load_path
