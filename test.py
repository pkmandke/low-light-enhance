"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

Example (You need to train models first):
    The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

See options/base_options.py and options/test_options.py for more test options.
"""

import os
from options.test_options import TestOptions
from data import get_data_loaders
from models import create_model
from util import html


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.shuffle_data = False  # disable data shuffling;
    opt.display_id = -1   # no visdom display; the test code optionally saves the results to a HTML file.
    dataloader = get_data_loaders(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}'.format(opt.epoch))  # define the website directory

    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Epoch = %s' % (opt.name, opt.epoch))

    # test with eval mode. This only affects layers like batchnorm and dropout.
    if opt.eval:
        model.eval()
    model.initialize_test() # initialize variables to hold test metrics
    for i, data in enumerate(dataloader):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.accumulate_test_metrics_and_get_visuals()  # get image results
        if i % opt.save_nth_image == 0:
            print('processing (%04d)-th image... %s' % (i, visuals['image']))
            webpage.add_image_with_text(visuals, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    webpage.add_summary(model.get_test_metrics())
    webpage.save()  # save the HTML
