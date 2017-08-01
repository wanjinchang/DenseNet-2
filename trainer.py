import os
import shutil

import torch
from torch import nn
import torch.nn.parallel
from torch.autograd import Variable

from model import DenseNet
from data_loader import get_loader
from tensorboard_logger import configure, log_value

class Trainer(object):
    """


    """
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader

        # network params
        self.num_blocks = config.num_blocks
        self.num_layers_total = config.num_layers_total
        self.growth_rate = config.growth_rate
        self.bottleneck = config.bottleneck
        self.theta = config.compression

        # training params
        self.epochs = config.epochs
        self.lr = config.lr
        self.momentum = config.momentum
        self.weight_decay = config.weight_decay
        self.dropout_rate = config.dropout_rate

        # other params
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.load_path = config.load_path
        self.num_gpu = config.num_gpu
        self.use_tensorboard = config.use_tensorboard
        self.resume = config.resume

        # build model
        self.model = DenseNet(self.num_blocks, self.num_layers_total,
                self.growth_rate, 10, self.bottleneck, self.dropout_rate, self.theta)

        print('Number of model parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))

        if self.num_gpu > 0:
            self.model.cuda()

        # load a model if resume or if testing
        if self.load_path or self.resume:
            self.load_model()

        # configure tensorboard logging
        if self.use_tensorboard:
            configure(self.logs_dir + get_name())

    def train(self):
        pass

    def test(self):
        pass

    def save_checkpoint(self, state, is_best, filename='ckpt.pth.tar'):
        """
        Save a copy of the model so that it can be loaded at a future
        date. This function is used when the model is being evaluated 
        on the test data.

        Furthermore, the model with the highest accuracy is saved as
        with a special name.
        """
        print("[*] Saving model to {}".format(self.ckpt_dir))

        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            shutil.copyfile(ckpt_path, 
                os.path.join(self.ckpt_dir, 'model_best.pth.tar'))

    def load_checkpoint(self, best=False):
        """
        Load the best copy of a model. This is useful for 2 cases:

        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Params
        ------
        - best: if set to True, loads the best model. Use this if you want
          to evaluate your model on the test data. Else, set to False in 
          which case the most recent version of the checkpoint is used.
        """
        print("[*] Loading model from {}".format(self.ckpt_dir))

        filename = 'ckpt.pth.tar'
        if best:
            filename = 'model_best.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)

        ckpt = torch.load(ckpt_path)
        self.start_epoch = ckpt['epoch']
        self.best_prec1 = ckpt['best_prec1']
        self.model.load_state_dict(ckpt['state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        
        print("[*] Loaded checkpoint @ epoch {}".format(ckpt['epoch']))

    def adjust_learning_rate(optimizer, epoch):
        pass

    def get_name(self):
        """
        Returns the name of the model based on the configuration
        parameters.

        The name will take the form DenseNet-Y-X, where X is the total
        number of layers specified by `config.total_num_layers` and
        Y is either an empty string or BC based on whether `config.bottleneck`
        is set to false or true respectively.

        For example, given 169 layers with bottleneck, this function
        will return DenseNet-BC-169.
        """
        if self.bottleneck:
            return 'DenseNet-BC-{}'.format(self.total_num_layers)
        return 'DenseNet-{}'.format(self.total_num_layers)
