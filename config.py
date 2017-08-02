import argparse
from utils import str2bool

arg_lists = []
parser = argparse.ArgumentParser(description='DenseNet')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# network params
net_arg = add_argument_group('Network')
net_arg.add_argument('--num_blocks', type=int, default=3,
                        help='# of Dense blocks to use in the network')
net_arg.add_argument('--num_layers_total', type=int, default=40,
                        help='Total # of layers in the network')
net_arg.add_argument('--growth_rate', type=int, default=12,
                        help='Growth rate (k) of the network')
net_arg.add_argument('--bottleneck', type=str2bool, default=False,
                        help='Whether to use bottleneck layers')
net_arg.add_argument('--compression', type=float, default=1.0,
                        help='Compression factor theta in the range [0, 1]')

# data params
data_arg = add_argument_group('Data')
data_arg.add_argument('--valid_size', type=float, default=0.1,
                        help='% split of training set used for validation set')
data_arg.add_argument('--batch_size', type=int, default=64,
                        help='# of images in each batch of data')
data_arg.add_argument('--num_worker', type=int, default=1,
                        help='# of subprocesses to use for data loading')
data_arg.add_argument('--augment', type=str2bool, default=True,
                        help='Whether to apply data augmentation or not')
data_arg.add_argument('--shuffle', type=str2bool, default=True,
                        help='Whether to shuffle the dataset after every epoch')
data_arg.add_argument('--show_sample', type=str2bool, default=False,
                        help='Whether to visualize a sample grid of the data')

# training params
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True,
                            help='Whether to train or test the model')
train_arg.add_argument('--epochs', type=int, default=300,
                            help='# of epochs to train for')
train_arg.add_argument('--lr', type=float, default=0.1,
                            help='Initial learning rate value')
train_arg.add_argument('--momentum', type=float, default=0.9,
                            help='Nesterov momentum value')
train_arg.add_argument('--weight_decay', type=float, default=1e-4,
                            help='weight decay penalty')
train_arg.add_argument('--dropout_rate', type=float, default=0.0,
                            help='Dropout rate used with non-augmented datasets')

# other params
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--random_seed', type=int, default=4242,
                        help='Seed to ensure reproducibility')
misc_arg.add_argument('--load_path', type=str, default='',
                        help='Path to pretrained model. Should be specified if testing the model')
misc_arg.add_argument('--data_dir', type=str, default='./data',
                        help='Directory in which data is stored')
misc_arg.add_argument('--ckpt_dir', type=str, default='./ckpt',
                        help='Directory in which to save model checkpoints')
misc_arg.add_argument('--logs_dir', type=str, default='./logs/',
                        help='Directory in which Tensorboard logs wil be stored')
misc_arg.add_argument('--num_gpu', type=int, default=0,
                        help="# of GPU's to use. A value of 0 will run on the CPU")
misc_arg.add_argument('--use_tensorboard', type=str2bool, default=False,
                        help='Whether to use tensorboard for visualization')
misc_arg.add_argument('--resume', type=str2bool, default=True,
                        help='Whether to resume training from most recent checkpoint')
misc_arg.add_argument('--print_freq', type=int, default=10,
                        help='How frequently to display training details on screen')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed