""""
A default data loader
"""

import torch

from data import find_dataset_using_name

class DefaultDataLoader:
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class
        Step 1: create a dataset instance given the name [dataset_mode]
        """
        self.opt = opt
        self.data_loader = None
        self.dataset_class = find_dataset_using_name(opt.dataset)

    def load_data(self):
        '''To be called before using the dataloader.
            Creates multi-threaded data loader/s.
        '''
        self.dataset = self.dataset_class(self.opt)

        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.data_loader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=self.opt.batch_size,
            shuffle=self.opt.shuffle_data,
            num_workers=int(self.opt.num_workers),
            pin_memory=True)

        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data"""
        for data in self.data_loader:
            yield data
