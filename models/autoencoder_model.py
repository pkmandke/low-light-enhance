"""Model class for auto-encoders
"""
from collections import OrderedDict
import os

import torch
import numpy as np
import torch.nn as nn
from .base_model import BaseModel
import ssim
from . import utils as model_utils
from util.util import compute_psnr
from skimage.metrics import structural_similarity


class AutoencoderModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(dataset_mode='aligned')  # You can rewrite default values for this model. For example, this model usually uses aligned dataset as its dataset.
        if is_train:
            parser.add_argument('--lambda_rmse', type=float, default=1.0, help='weight for the rmse loss')  # You can define new arguments for this model.
            parser.add_argument('--lambda_ssim', type=float, default=0., help='weight for the ssim loss')

        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.losses = {'RMSE': nn.MSELoss(), 'SSIM': ssim.SSIM(window_size=1, size_average=True)}
        self.loss_names = list(self.losses.keys())
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['data_A', 'data_B', 'output']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        self.models = {'auto_encoder': model_utils.load_network(opt, gpu_ids=self.gpu_ids)}
        self.model_names = ['auto_encoder']
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        if self.isTrain:  # only defined during training time
            # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
            # We also provide a GANLoss class "networks.GANLoss". self.criterionGAN = networks.GANLoss().to(self.device)
            # define and initialize optimizers. You can define one optimizer for each network.
            # If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
            self.optimizer = torch.optim.Adam(self.models['auto_encoder'].parameters(), lr=opt.lr,
                                              weight_decay=opt.l2_decay)
            self.optimizers = [self.optimizer]

            # self.schedulers = [model_utils.get_scheduler(self.optimizer, opt)]

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            data: a dictionary that contains the data and its metadata information.
        """
        self.input = data['low'].to(self.device)

        self.target = data['high'].to(self.device)
        self.image_paths = data['image']
        self.data = data

    def forward(self, input=None):
        """Run forward pass. This will be called by both functions."""
        if input is not None:
            self.output = self.models['auto_encoder'](input)
            return self.output

        self.output = self.models['auto_encoder'](self.input)
        return self.output

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss_rmse = self.losses['RMSE'](self.target, self.output)
        self.loss_ssim = self.losses['SSIM'](self.target, self.output)
        self.loss = self.loss_rmse * self.opt.lambda_rmse + (1. - self.loss_ssim) * self.opt.lambda_ssim
        self.loss.backward()       # calculate gradients of network

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()               # first call forward to calculate intermediate results
        self.optimizer.zero_grad()   # clear network's existing gradients
        self.backward()              # calculate gradients
        self.optimizer.step()        # update gradients for network

    def init_epoch(self):
        """ Init placeholders for loss and metrics for current epoch """

        self.epoch_metadata = {'num_train_images': 0,
                               'num_val_images': 0}

        self.epoch_losses = {'train_loss': 0.,
                             'val_loss': 0.,
                             'train_rmse': 0.,
                             'val_rmse': 0.}

        self.epoch_metrics = {'train_ssim': 0.,
                              'val_ssim': 0.,
                              'train_psnr': 0.,
                              'val_psnr': 0.}
        self.epoch_init = True

    def train_epoch(self, data_loader):
        """ Given dataloader, train one epoch with all images """

        assert self.epoch_init  # Ensure that the variables have been reset
        self.epoch_init = False

        self.models['auto_encoder'].train()

        for data in data_loader:
            self.set_input(data)
            self.epoch_metadata['num_train_images'] += self.input.shape[0]  # Accumulate each batch size for the entire epoch

            self.optimize_parameters()

            self.epoch_losses['train_rmse'] += (self.loss_rmse.item() * self.input.shape[0])
            self.epoch_losses['train_loss'] += (self.loss.item() * self.input.shape[0])
            self.epoch_metrics['train_ssim'] += (self.loss_ssim.item() * self.input.shape[0])
            self.epoch_metrics['train_psnr'] += (compute_psnr(torch.clamp(self.output, 0., 1.), self.target,
                                                              max_pixel_value=1.) * self.input.shape[0])

        self.epoch_losses['train_rmse'] /= self.epoch_metadata['num_train_images']
        self.epoch_losses['train_loss'] /= self.epoch_metadata['num_train_images']
        self.epoch_metrics['train_ssim'] /= self.epoch_metadata['num_train_images']
        self.epoch_metrics['train_psnr'] /= self.epoch_metadata['num_train_images']

    def validate(self, dataloader):
        """ Validate model with given dataloader """

        total_imgs = dataloader.len_val_set
        self.models['auto_encoder'].eval()

        for data in dataloader.val_loader:

            self.input = data['low'].to(self.device)
            with torch.no_grad():
                self.forward()

            # Accumulate losses
            rmse = self.losses['RMSE'](self.output, data['high'].to(self.device)).item()
            ssim = self.losses['SSIM'](self.output, data['high'].to(self.device)).item()
            self.epoch_losses['val_rmse'] += (rmse * self.input.shape[0])
            self.epoch_losses['val_loss'] += ((self.opt.lambda_rmse * rmse + (1. - ssim) * self.opt.lambda_ssim)
                                              * self.input.shape[0])
            self.epoch_metrics['val_ssim'] += (ssim * self.input.shape[0])
            self.epoch_metrics['val_psnr'] += (compute_psnr(torch.clamp(self.output, 0., 1.), data['high'].to(self.device),
                                                            max_pixel_value=1.) * self.input.shape[0])

        # Compute average loss and metrics for the full val set
        self.epoch_losses['val_rmse'] /= total_imgs
        self.epoch_losses['val_loss'] /= total_imgs
        self.epoch_metrics['val_ssim'] /= total_imgs
        self.epoch_metrics['val_psnr'] /= total_imgs

    def log_parameters(self, epoch=None):
        """ Log parameters after epoch """

        if self.opt.logging:

            self.logger.info('Epoch {} complete.'.format(epoch))

            for k, v in self.epoch_losses.items():
                self.logger.info('{} = {}'.format(k.capitalize(), v))
            for k, v in self.epoch_metrics.items():
                self.logger.info('{} = {}'.format(k.capitalize(), v))

    def log_model_info(self, verbose=True):
        """ Log model parameter count, metadata and (optionally) network architecture if logging is enabled """

        if not self.opt.logging:
            return

        self.logger.info('---------- Networks initialized -------------\n')
        for name in self.model_names:
            if isinstance(name, str):
                net = self.models[name]
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    self.logger.info(net)
                self.logger.info('\n[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        self.logger.info('-----------------------------------------------\n')


    def get_epoch_losses(self):
        """ Return a dict with current epoch losses """

        return OrderedDict({k: float(v) for k, v in self.epoch_losses.items()})

    def get_epoch_metrics(self):
        """ Return a dict containing current epochs metrics """

        return OrderedDict({k: float(v) for k, v in self.epoch_metrics.items()})

    def get_plotting_artifacts(self):
        """ Return dict with values to plot with visdom for current epoch """

        return {
            'Loss': OrderedDict({k: float(v) for k, v in self.epoch_losses.items() if k.endswith('_loss')}),
            'SSIM': OrderedDict({k: float(v) for k, v in self.epoch_metrics.items() if k.endswith('_ssim')}),
            'PSNR': OrderedDict({k: float(v) for k, v in self.epoch_metrics.items() if k.endswith('_psnr')})
        }

    def initialize_test(self):
        """ Initialize params for test """

        self.test_params = {
            'num_images': 0,
            'rmse': 0.,
            'ssim': 0.,
            'psnr': 0.
        }

    def test(self):
        """ Run forward pass at test time """

        self.models['auto_encoder'].eval()

        with torch.no_grad():
            self.forward()

    def accumulate_test_metrics_and_get_visuals(self):
        """ Accumulate metrics at test time """

        self.test_params['num_images'] += 1
        rmse = np.format_float_positional(self.losses['RMSE'](self.output, self.target).item(), precision=9,
                                          unique=False)
        self.test_params['rmse'] += float(rmse)
        # ssim = np.format_float_positional(self.losses['SSIM'](self.output, self.target).item(), precision=9,
        #                                   unique=False)
        sq = torch.squeeze
        tg = sq(self.target).detach().cpu().numpy()
        ssim = np.format_float_positional(structural_similarity(np.transpose(sq(self.output).detach().cpu().numpy(), (1,2,0)), np.transpose(tg, (1,2,0)),
                                                                data_range=tg.max() - tg.min(), multichannel=True, win_size=11),
                                          precision=9,
                                          unique=False)
        self.test_params['ssim'] += float(ssim)
        psnr = np.format_float_positional(
                compute_psnr(torch.clamp(self.output, 0., 1.), self.target, max_pixel_value=1.),
                precision=9, unique=False)
        self.test_params['psnr'] += float(psnr)

        return {
            'image': self.image_paths,
            'low_path': os.path.join(self.opt.dataroot, self.opt.data_path_suffix, 'low'),
            'Predicted': self.output.detach(),
            'target_path': os.path.join(self.opt.dataroot, self.opt.data_path_suffix, 'high'),
            'metrics': {'SSIM': ssim, 'RMSE': rmse, 'PSNR': psnr}
        }

    def get_test_metrics(self):
        """ Return a summary of test metrics """

        num_images = self.test_params['num_images']
        return {
            'Mean SSIM': self.test_params['ssim'] / num_images,
            'Mean RMSE': self.test_params['rmse'] / num_images,
            'Mean PSNR': self.test_params['psnr'] / num_images,
        }
