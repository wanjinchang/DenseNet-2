import os
import time
import shutil
from tqdm import trange

import torch
from torch import nn
import torch.nn.parallel
from torch.autograd import Variable

from model import DenseNet
from tensorboard_logger import configure, log_value

class Trainer(object):
    def __init__(self, config, data_loader):
        self.config = config
        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
        else:
            self.test_loader = data_loader

        # network params
        self.num_blocks = config.num_blocks
        self.num_layers_total = config.num_layers_total
        self.growth_rate = config.growth_rate
        self.bottleneck = config.bottleneck
        self.theta = config.compression

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.best_valid_acc = 0.
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
        self.print_freq = config.print_freq

        # build densenet model
        self.model = DenseNet(self.num_blocks, self.num_layers_total,
                self.growth_rate, 10, self.bottleneck, self.dropout_rate, self.theta)

        print('[*] Number of model parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))

        # define loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr,
                momentum=self.momentum, weight_decay=self.weight_decay)

        if self.num_gpu > 0:
            # model = torch.nn.DataParallel(model).cuda()
            self.model.cuda()
            self.criterion.cuda()

        # finally configure tensorboard logging
        if self.use_tensorboard:
            configure(self.logs_dir + self.get_name())

    def train(self):
        # load the most recent checkpoint
        if self.resume:
            self.load_checkpoint(best=False)

        for epoch in trange(self.start_epoch, self.epochs):
            # decay learning rate
            self.adjust_learning_rate(epoch)

            # train for 1 epoch
            self.train_one_epoch(epoch)

            # evaluate on validation set
            valid_acc = self.validate()

            is_best = valid_acc > self.best_valid_acc
            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_valid_acc': self.best_valid_acc}, is_best)

    def train_one_epoch(self, epoch):
        """
        Train the model on 1 epoch of the training set. An epoch
        corresponds to one full pass through the entire training
        set in successive mini-batches.
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()

        tic = time.time()
        for i, (image, target) in enumerate(self.train_loader):
            if self.num_gpu > 0:
                image = image.cuda()
                target = target.cuda(async=True)
            input_var = torch.autograd.Variable(image)
            target_var = torch.autograd.Variable(target)

            # forward pass
            output = self.model(input_var)

            # compute loss & accuracy 
            loss = self.criterion(output, target_var)
            acc = self.accuracy(output.data, target)
            losses.update(loss.data[0], image.size()[0])
            accs.update(acc, image.size()[0])

            # compute gradients and update SGD
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            toc = time.time()
            batch_time.update(toc-tic)

            # print to screen
            if i % self.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Train Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Train Acc: {acc.val:.3f} ({acc.avg:.3f})'.format(
                        epoch, i, len(self.train_loader), batch_time=batch_time,
                        loss=losses, acc=accs))

        # log to tensorboard
        if self.use_tensorboard:
            log_value('train_loss', losses.avg, epoch)
            log_value('train_acc', accs.avg, epoch)


    def validate(self):
        """
        Evaluate the model loss and accuracy on the validation
        set.
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()

        tic = time.time()
        for i, (image, target) in enumerate(self.valid_loader):
            if self.num_gpu > 0:
                image = image.cuda()
                target = target.cuda(async=True)
            input_var = torch.autograd.Variable(image)
            target_var = torch.autograd.Variable(target)

            # forward pass
            output = self.model(input_var)

            # compute loss & accuracy 
            loss = self.criterion(output, target_var)
            acc = self.accuracy(output.data, target)
            losses.update(loss.data[0], image.size()[0])
            accs.update(acc, image.size()[0])

            # measure elapsed time
            toc = time.time()
            batch_time.update(toc-tic)

            # print to screen
            if i % self.print_freq == 0:
                print('Valid: [{0}/{1}]\t'
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Valid Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Valid Acc: {acc.val:.3f} ({acc.avg:.3f})'.format(
                        i, len(self.valid_loader), batch_time=batch_time,
                        loss=losses, acc=accs))

        print('[*] Valid Acc: {acc.avg:.3f}'.format(acc=accs))

        # log to tensorboard
        if self.use_tensorboard:
            log_value('val_loss', losses.avg, epoch)
            log_value('val_acc', accs.avg, epoch)

        return accs.avg


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

        # load variables from checkpoint
        self.start_epoch = ckpt['epoch']
        self.best_valid_acc = ckpt['best_valid_acc']
        self.model.load_state_dict(ckpt['state_dict'])
        
        print("[*] Loaded checkpoint @ epoch {}".format(ckpt['epoch']))

    def adjust_learning_rate(self, epoch):
        """
        Learning rate schedule defined in the paper.

        The initial learning rate is set to 0.1, and is 
        divided by 10 at 50% and 75% of the total number of 
        training epochs.
        """
        sched1 = int(0.5 * self.epochs)
        sched2 = int(0.75 * self.epochs)
        if epoch < sched1:
            lr = self.lr
        elif epoch >= sched1 and epoch <= sched2:
            lr /= 10
        else:
            lr /= 10

        # log to tensorboard
        if self.use_tensorboard:
            log_value('learning_rate', lr, epoch)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

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
            return 'DenseNet-BC-{}'.format(self.num_layers_total)
        return 'DenseNet-{}'.format(self.num_layers_total)

    def accuracy(self, predicted, ground_truth):
        """
        Utility function for calculating the accuracy of the model.

        Params
        ------
        - predicted: (torch.FloatTensor)
        - ground_truth: (torch.LongTensor)

        Returns
        -------
        - acc: (float) % accuracy.
        """
        predicted = torch.max(predicted, 1)[1]
        total = len(ground_truth)
        correct = (predicted == ground_truth).sum()
        acc = 100 * (correct / total)
        return acc

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
