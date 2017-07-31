import torch

from trainer import Trainer
from config import get_config
from data_loader import get_loader

def main(config):

    # ensure directories are setup
    prepare_dirs(config)

    # fix a seed
    torch.manual_seed(config.random_seed)
    if config.num_gpu > 0:
        torch.cuda.manual_seed(config.random_seed)

    # only augment on training set
    augment = False
    if config.is_train:
        augment = config.augment

    # load the data
    data_loader = get_loader(config.data_dir, 
                             config.is_train, 
                             config.batch_size, 
                             config.num_workers,
                             augment, 
                             do_shuffle, 
                             config.show_sample)

    # instantiate trainer
    trainer = Trainer(config, data_loader)

    # dump train config and train
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
