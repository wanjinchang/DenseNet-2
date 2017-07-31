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
    data_loader = get_loader(config.data_dir, config.is_train, 
         config.batch_size, config.augment, **kwargs)

    # instantiate trainer
    trainer = Trainer(config, data_loader)

    # either train
    if config.is_train:
        save_config(config)
        trainer.train()

    # or load a pretrained model and test
    else:
        if not config.load_path:
            raise Exception("[!] You should specify `load_path` to load a pretrained model")
        trainer.test()

if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
