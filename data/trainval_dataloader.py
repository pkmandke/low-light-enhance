"""
Data Loader for Train-Val-Test data for auto-encoder training
"""

import os, csv
from data.default_dataloader import DefaultDataLoader

import torch


class TrainvalDataLoader(DefaultDataLoader):

    @staticmethod
    def modify_commandline_options(parser, is_train):

        parser.add_argument('--csv_path', type=str, default='../datasets/lol/',
                            help='Path to dir where dataset metadata is saved.')
        if is_train:
            parser.add_argument('--train_filename', type=str, default='train_450.csv',
                                help='Name of the train metadata csv file.')
            parser.add_argument('--val_filename', type=str, default='val_35.csv',
                                help='Name of the validation metadata csv file.')
        else:
            parser.add_argument('--test_filename', type=str, default='test_15.csv',
                                help='Name of the test metadata csv file.')

        return parser

    def __init__(self, opt):

        super(TrainvalDataLoader, self).__init__(opt)

        self.train_loader, self.test_loader, self.val_loader = None, None, None
        if opt.isTrain:
            self.len_train_set = None
            self.len_val_set = None
        else:
            self.len_test_set = None

    def load_data(self):
        """
        Load the CSV files with image names and trait labels
        Create trainval or test dataset based on self.opt.is_train
        """

        if self.opt.isTrain:
            with open(os.path.join(self.opt.csv_path, self.opt.train_filename), 'r') as csvfile:
                csvr = csv.reader(csvfile, delimiter=',')
                for l in csvr:
                    train_list = l
            train_set = self.dataset_class(self.opt, train_list)

            with open(os.path.join(self.opt.csv_path, self.opt.val_filename), 'r') as csvfile:
                csvr = csv.reader(csvfile, delimiter=',')
                for l in csvr:
                    val_list = l
            val_set = self.dataset_class(self.opt, val_list)

            self.dataset = train_set

            self.len_train_set = len(train_set)
            self.len_val_set = len(val_set)

            self.train_loader = torch.utils.data.DataLoader(train_set,
                                                            batch_size=self.opt.batch_size,
                                                            shuffle=self.opt.shuffle_data,
                                                            num_workers=int(self.opt.num_threads))
            self.val_loader = torch.utils.data.DataLoader(val_set,
                                                          batch_size=self.opt.batch_size,
                                                          shuffle=self.opt.shuffle_data,
                                                          num_workers=int(self.opt.num_threads))

            self.data_loader = self.train_loader
        else:
            # At test time
            with open(os.path.join(self.opt.csv_path, self.opt.test_filename), 'r') as csvfile:
                csvr = csv.reader(csvfile, delimiter=',')
                for l in csvr:
                    test_list = l
            test_set = self.dataset_class(self.opt, test_list)

            self.len_test_set = len(test_set)

            self.dataset = test_set

            self.data_loader = torch.utils.data.DataLoader(test_set,
                                                           batch_size=1,
                                                           shuffle=False,
                                                           num_workers=int(self.opt.num_threads))

        return self
