import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE


if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(webpage, visuals, txt, webpage_header='Test', aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()

    webpage.add_header(webpage_header)
    ims, txts, links = [], [], []

    for name, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            self.name_to_disp_dict = {}
            self.next_disp_id = opt.display_id
            import visdom
            self.ncols = opt.display_ncols
            if opt.display_env == '':
                if opt.name:
                    opt.display_env = opt.name
                else:
                    opt.display_env = 'main'

            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def line_plot(self, plot_name, x, y_dict, xlabel='epochs', ylabel='Metric/Loss'):
        """ Create line plots.
        Update existing plot with same name.
        Else create new plot with `name`
        """

        if not hasattr(self, plot_name):
            setattr(self, plot_name, {'X': [], 'Y': [], 'legend': list(y_dict.keys())})
            self.name_to_disp_dict[plot_name] = self.next_disp_id
            self.next_disp_id += 1
        plot_data = getattr(self, plot_name)
        plot_data['X'].append(x)
        plot_data['Y'].append([y_dict[k] for k in y_dict.keys()])
        try:
            self.vis.line(
                X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
                Y=np.array(plot_data['Y']),
                opts={
                    'title': self.name + ' ' + plot_name,
                    'legend': plot_data['legend'],
                    'xlabel': xlabel,
                    'ylabel': ylabel},
                win=self.name_to_disp_dict[plot_name])
        except VisdomExceptionBase:
            self.create_visdom_connections()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses_and_metrics(self, epoch, losses, metrics, epoch_time):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        """
        message = '(epoch: {}, time: {}, ) '.format(epoch, epoch_time)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
        for k, v in metrics.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
