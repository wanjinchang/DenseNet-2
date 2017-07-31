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

        # network params
        self.num_blocks = config.num_blocks
        self.num_layers_total = config.num_layers_total
        self.growth_rate = config.growth_rate
        self.bottleneck = config.bottleneck
        self.compression = config.compression

        # training params
        self.epochs = config.epochs
        self.lr = config.lr
        self.momentum = config.momentum
        self.weight_decay = config.weight_decay
        self.dropout_rate = config.dropout_rate

        # other params
        self.ckpt_dir = config.ckpt_dir
        self.num_gpu = config.num_gpu
        self.use_tensorboard = config.use_tensorboard

        # build model
        self.model = DenseNet(self.num_blocks, self.num_layers_total,
                self.growth_rate, 10, self.bottleneck, self.dropout_rate, self.theta)

        if self.num_gpu > 0:
            self.model.cuda()

        if self.load_path:
            self.load_model()


    def train(self):

    def test(self):

    def save_model(self):

    def load_model(self):

    def adjust_learning_rate(optimizer, epoch):


