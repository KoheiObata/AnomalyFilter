from loaders.load import load_data
from loaders.loader import Loader, Loader_batch
import numpy as np


def load_dataloader(dataparams, group='train'):
    dataset_path = './dataset'
    if group=='train':
        train_dataset, val_dataset = load_data(dataset=dataparams.dataset,
                    group='train',
                    entities=dataparams.entities,
                    downsampling=dataparams.downsampling,
                    min_length=None,
                    root_dir=dataset_path,
                    verbose=True,
                    validation=True)

        train_dataloader = Loader(dataset=train_dataset,
                                    batch_size=dataparams.batch_size,
                                    window_size=dataparams.window_size,
                                    window_step=dataparams.window_step,
                                    shuffle=True,
                                    padding_type='None',
                                    sample_with_replace=False,
                                    verbose=True,
                                    mask_position='None',
                                    n_masked_timesteps=0)

        val_dataloader = Loader(dataset=val_dataset,
                                    batch_size=dataparams.batch_size,
                                    window_size=dataparams.window_size,
                                    window_step=dataparams.window_step,
                                    shuffle=True,
                                    padding_type='None',
                                    sample_with_replace=False,
                                    verbose=True,
                                    mask_position='None',
                                    n_masked_timesteps=0)
        return train_dataloader, val_dataloader
    elif group=='test':
        test_dataset = load_data(dataset=dataparams.dataset,
                    group='test',
                    entities=dataparams.entities,
                    downsampling=dataparams.downsampling,
                    min_length=None,
                    root_dir=dataset_path,
                    verbose=True,
                    validation=False)

        test_dataloader = Loader(dataset=test_dataset,
                                    batch_size=dataparams.batch_size,
                                    window_size=dataparams.window_size,
                                    window_step=dataparams.window_size,
                                    shuffle=False,
                                    padding_type='None',
                                    sample_with_replace=False,
                                    verbose=True,
                                    mask_position='None',
                                    n_masked_timesteps=0)
        return test_dataloader