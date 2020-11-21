""" LOL dataset """

import os
from random import shuffle

from data.base_dataset import BaseDataset
from data import utils

from torchvision import transforms
from PIL import Image
import numpy as np

class LolDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        """
        parser.add_argument('--dim', type=int, default=256, help='Image dimension for input to model')
        parser.add_argument('--random_horz_flip', type=float, default=0.,
                            help='Probability of random horizontal flip of image.')
        return parser

    def __init__(self, opt, data_list):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir = os.path.join(opt.dataroot, opt.data_path_suffix)

        self.data_list = data_list

        self.transforms = transforms.Compose([
            transforms.Lambda(self.makeSquared),
            transforms.ToTensor(),
            # transforms.Normalize([0., 0., 0.], [255., 255., 255.])
        ])

        if opt.shuffle_data:
            shuffle(self.data_list)

    def __getitem__(self, idx):
        """Return a data point and its metadata information.

        Parameters:
            idx - - a random integer for data indexing
        """

        l, h = Image.open(os.path.join(self.dir, 'low', self.data_list[idx])), Image.open(
            os.path.join(self.dir, 'high', self.data_list[idx]))

        return {
            'high': self.transforms(h),
            'low': self.transforms(l),
            'image': self.data_list[idx]
        }

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.data_list)

    def makeSquared(self, img):
        """
        Resize rectangular image to square by maintaining the aspect ratio.
        """
        return utils.MakeSquared(img, dim=self.opt.dim)
