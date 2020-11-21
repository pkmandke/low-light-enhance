"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model') and different datasets.
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

See options/base_options.py and options/train_options.py for more training options.
"""
import time
from datetime import timedelta

from options.train_options import TrainOptions
from data import get_data_loaders
from models import create_model
from util.visualizer import Visualizer


def main():
    """ Main function to train a model with given cmdline options """

    opt = TrainOptions().parse()   # get training options
    dataloader = get_data_loaders(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataloader)    # get the number of images in the train set.

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.log_model_info(opt.verbose) # log model metadata to log file iff opt.logging is enabled
    model.logger.info('The number of training images = {}'.format(dataset_size))
    model.logger.info('Num val images = {}'.format(dataloader.len_val_set))
    print('The number of training images = {}'.format(dataset_size))
    print('Num val images = {}'.format(dataloader.len_val_set))
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots

    start_time = time.monotonic()
    for epoch in range(opt.epoch_count, opt.n_epochs + 1):

        # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        visualizer.reset()
        # model.update_learning_rate()    # update learning rates in the beginning of every epoch.

        epoch_start_time = time.monotonic()
        model.init_epoch()
        model.train_epoch(dataloader.train_loader)
        model.validate(dataloader)
        epoch_end_time = time.monotonic()

        model.log_parameters(epoch)

        if epoch % opt.log_freq == 0:    # print training losses and save logging information to the disk
            losses = model.get_epoch_losses()
            metrics = model.get_epoch_metrics()
            epoch_time = timedelta(seconds=epoch_end_time - epoch_start_time)
            visualizer.print_current_losses_and_metrics(epoch, losses, metrics, epoch_time)
            if opt.display_id > 0:
                for n, y_dict in model.get_plotting_artifacts().items():
                    visualizer.line_plot(n, epoch, y_dict, xlabel='epochs', ylabel=n)

        if epoch % opt.save_epoch_freq == 0:              # cache model every <save_epoch_freq> epochs
            model.logger.info('saving the model at the end of epoch %d' % epoch)
            model.save_networks('latest')
            model.save_networks('epoch_%d' % epoch)
        if opt.verbose:
            print('End of epoch {} / {} \t Time Taken: {} sec'.format(epoch, opt.n_epochs,
                                                                      timedelta(seconds=epoch_end_time - epoch_start_time)))
        model.logger.info('End of epoch {} / {} \t Time Taken: {} sec'.format(epoch, opt.n_epochs,
                                                                              timedelta(seconds=epoch_end_time - epoch_start_time)))

        model.update_learning_rate()

    model.logger.info('Total training time for {} epochs = {}s'.format(opt.n_epochs,
                                                                       timedelta(seconds=time.monotonic() - start_time)))

    model.save_networks('epoch_{}_final'.format(opt.n_epochs))


if __name__ == '__main__':
    main()
