import dominate
from dominate.tags import meta, hr, h3, table, tr, td, p, a, img, br, b, u, figure, figcaption
import os
from . import util

class HTML:
    """This HTML class allows us to save images and write texts into a single HTML file.

     It consists of functions such as <add_header> (add a text header to the HTML file),
     <add_images> (add a row of images to the HTML file), and <save> (save the HTML to the disk).
     It is based on Python library 'dominate', a Python library for creating and manipulating HTML documents using a DOM API.
    """

    def __init__(self, web_dir, title, refresh=0):
        """Initialize the HTML classes

        Parameters:
            web_dir (str) -- a directory that stores the webpage. HTML file will be created at <web_dir>/index.html; images will be saved at <web_dir/images/
            title (str)   -- the webpage name
            refresh (int) -- how often the website refresh itself; if 0; no refreshing
        """
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, 'images')
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def get_image_dir(self):
        """Return the directory that stores images"""
        return self.img_dir

    def add_header(self, text):
        """Insert a header to the HTML file

        Parameters:
            text (str) -- the header text
        """
        with self.doc:
            h3(text)

    def add_image_with_text(self, visuals, aspect_ratio=1.0, width=256):
        """ Add images along with textual metadata """

        image_dir = self.get_image_dir()

        visuals['image'] = visuals['image'][0]
        self.add_header(visuals['image'])

        pred = util.tensor2im(visuals['Predicted'])
        image_name = visuals['image']

        save_path = os.path.join(image_dir, 'pred_' + image_name)
        util.save_image(pred, save_path, aspect_ratio=aspect_ratio)

        t = table(border=1, style="table-layout: fixed; width: 1200px;")  # Insert a table
        self.doc.add(t)
        with t:
            with tr():
                with td(style="word-wrap: break-word;", halign="center", valign="top"):
                    with p():
                        hr()
                        p(b('ImageID: '), visuals['image'])
                        br()
                        #### Add image and feature maps ####
                        with figure(style="display: inline-block;"):
                            img(style="border:0px;margin:0px;float:left;width:%dpx;" % width,
                                src=os.path.relpath(os.path.join(visuals['low_path'], visuals['image']), self.web_dir))
                            figcaption('Low-Light', style="text-align: center;")
                        with figure(style="display: inline-block;"):
                            img(style="border:0px;margin:0px;float:left;width:%dpx;" % width,
                                src=os.path.join('images', 'pred_' + visuals['image']))
                            figcaption('Prediction', style="text-align: center;")
                        with figure(style="display: inline-block;"):
                            img(style="border:0px;margin:0px;float:left;width:%dpx;" % width,
                                src=os.path.relpath(os.path.join(visuals['target_path'], visuals['image']), self.web_dir))
                            figcaption('Ground Truth', style="text-align: center;")
                        #### Add image and feature maps ####
                        br()
                        # Add a table for class probabilities
                        for k, v in visuals['metrics'].items():
                            p("{} = {}".format(k, float(v)))
                            br()

    def add_summary(self, dict_):
        """ Add a summary with key value pairs from the dictionary """

        self.doc += br()
        self.doc += hr()
        self.doc += br()
        self.add_header("Test Summary")
        self.doc += br()

        for k, v in dict_.items():
            self.doc += p(b(k), ' = ', str(v))
            self.doc += br()

    def save(self):
        """save the current content to the HMTL file"""
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()