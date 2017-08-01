import torch

from trainer import Trainer
from config import get_config
from data_loader import get_loader
from utils import prepare_dirs, save_config

def main(config):

    # ensure directories are setup
    prepare_dirs(config)

    if config.num_gpu > 0:
        torch.cuda.manual_seed(config.random_seed)
        kwargs = {'num_workers': 1, 'pin_memory': True}
    else:
        torch.manual_seed(config.random_seed)
        kwargs = {}

    kwargs['shuffle'] = config.shuffle
    kwargs['show_sample'] = config.show_sample

    # instantiate data loader
    if config.is_train:
        train_loader = get_loader(config.data_dir, 'train', 
            config.batch_size, config.augment, **kwargs)
        valid_loader = get_loader(config.data_dir, 'valid',
            config.batch_size, config.augment, **kwargs)
        data_loader = (train_loader, valid_loader)
    else:
        data_loader = get_loader(config.data_dir, 'test', 
            config.batch_size, config.augment, **kwargs)

    # instantiate trainer
    trainer = Trainer(config, data_loader)

    # either train
    if config.is_train:
        save_config(config)
        trainer.train()

    # # or load a pretrained model and test
    # else:
    #     if not config.load_path:
    #         raise Exception("[!] You should specify `load_path` to load a pretrained model")
    #     trainer.test()

if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
