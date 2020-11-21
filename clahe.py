""" Experiments with the CLAHE algorithm for low light image enhancement """

import os
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from skimage.metrics import structural_similarity as ssim, mean_squared_error


def apply_clahe(file: str, clip_limit: float = 40., grid_size: int = 8):
    """ Given an image, apply the CLAHE algorithm and return an enhanced image """

    assert file.endswith('.png')
    img = cv2.imread(file)
    if isinstance(img, type(None)):
        assert False
    # convert image to Lab color space
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    clahe_object = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))

    # apply CLAHE on the L channel
    out = clahe_object.apply(img_lab[:, :, 0])

    out_img = img_lab
    out_img[:, :, 0] = out
    # convert image from Lab to RGB space and return
    return cv2.cvtColor(out_img, cv2.COLOR_Lab2RGB)


def clahe():
    """ API for CLAHE """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='../datasets/lol/eval15/low', help='Directory path to locate images')
    parser.add_argument('--image_file', default='', help='Process a single image')
    parser.add_argument('--clip_limit', type=float, default=20.0, help='Grid size for CLAHE')
    parser.add_argument('--grid_size', type=int, default=2, help='Window/Grid size for CLAHE')
    parser.add_argument('--out_path', default='./results/clahe/lol/eval15', help='Path to store processed images')

    args = parser.parse_args()

    print("Clip Limit = {}, Grid Size = {}".format(args.clip_limit, args.grid_size))
    assert os.path.exists(args.out_path)

    if os.path.isfile(os.path.join(args.data_path, args.image_file)):
        cv2.imwrite(os.path.join(args.out_path, args.image_file),
                    apply_clahe(os.path.join(args.data_path, args.image_file),
                                clip_limit=args.clip_limit,
                                grid_size=args.grid_size))
    elif os.path.isdir(args.data_path):
        for file in os.listdir(args.data_path):
            if not isinstance(cv2.imread(os.path.join(args.data_path, file)), type(None)):
                cv2.imwrite(
                    os.path.join(args.out_path, file),
                    apply_clahe(os.path.join(args.data_path, file),
                                clip_limit=args.clip_limit,
                                grid_size=args.grid_size)
                )


def search_param(args, clip_limit: float = 40., grid_size: int = 8):
    """ Average metrics over given param values """

    rmse_value, ssim_value = 0., 0.
    for file in os.listdir(args.low_path):
        if file.endswith('.png'):
            pred = apply_clahe(os.path.join(args.low_path, file), clip_limit=clip_limit,
                               grid_size=grid_size)

            high = cv2.imread(os.path.join(args.high_path, file))
            high = cv2.cvtColor(high, cv2.COLOR_BGR2RGB)

            ssim_value += ssim(pred, high, data_range=high.max() - high.min(), multichannel=True, win_size=11)
            rmse_value += np.sqrt(mean_squared_error(pred, high))

    return ssim_value / len(os.listdir(args.low_path)), rmse_value / len(os.listdir(args.low_path))


def search_for_param(args, search_range=[], param: str = 'grid_size', grid_size=8, clip_limit=40.):
    """ Perform a range search for given parameter """

    rmse_list, ssim_list = [], []
    for p in search_range:

        if param == 'grid_size':
            grid_size = p
        else:
            clip_limit = p

        ssim_value, rmse_value = search_param(args, clip_limit=clip_limit, grid_size=grid_size)

        ssim_list.append(ssim_value)
        rmse_list.append(rmse_value)

    plt.plot(search_range, ssim_list, '-ro')
    plt.title('SSIM with fixed Grid Size = 2')
    plt.xlabel('Clip Limit')
    plt.ylabel('SSIM')
    plt.savefig(os.path.join(args.out_path, '{}_search_ssim.png'.format(param)))
    plt.clf()
    plt.plot(search_range, rmse_list, '-ro')
    plt.title('RMSE with fixed Grid Size = 2')
    plt.xlabel('Clip Limit')
    plt.ylabel('RMSE')
    plt.savefig(os.path.join(args.out_path, '{}_search_rmse.png'.format(param)))
    plt.clf()

def validate_params():
    """ Validate gridsize and/or contrast threshold for CLAHE based on SSIM/RMSE metrics """

    parser = argparse.ArgumentParser()

    parser.add_argument('--base_path', default='../datasets/lol/our485', help='Directory path to locate images')
    parser.add_argument('--image_file', default='', help='Process a single image')
    parser.add_argument('--clip_limit', type=float, default=40.0, help='Grid size for CLAHE')
    parser.add_argument('--grid_size', type=int, default=2, help='Window/Grid size for CLAHE')
    parser.add_argument('--out_path', default='./results/clahe/lol/our485', help='Path to metric plots')
    parser.add_argument('--grid_range', default='2, 4, 6, 8, 12', help='Range to validate grid_size')
    parser.add_argument('--thresh_range', default='10., 20., 30., 40., 50., 60., 80., 100.')
    args = parser.parse_args()

    args.low_path = os.path.join(args.base_path, 'low')
    args.high_path = os.path.join(args.base_path, 'high')
    args.grid_range = [int(_.replace(' ', '')) for _ in args.grid_range.split(',')]
    args.thresh_range = [float(_.replace(' ', '')) for _ in args.thresh_range.split(',')]

    assert os.path.exists(args.out_path)
    assert os.path.exists(args.base_path)
    assert os.path.exists(args.low_path)
    assert os.path.exists(args.high_path)

    # search_for_param(args, args.grid_range, param='grid_size', clip_limit=args.clip_limit)
    search_for_param(args, args.thresh_range, param='clip_limit', grid_size=args.grid_size)


def compute_ssim_rmse():
    """ Compute RMSE and SSIM for images in given dir """

    # results_dir = '../../enlighten_gan/EnlightenGAN/ablation/enlightengan_retrain/test_latest/images'
    true_img_dir = '../datasets/lol/eval15/high'
    # results_dir = '/raid/pkmandke/projects/cv_project/results/ex3tr5_bilinear_rmse/400/images'
    results_dir = './results/clahe/lol/eval15'

    f = open('./results/clahe_final.txt', 'w')
    ssim_sum, rmse_sum = 0., 0.
    for file in os.listdir(results_dir):
        # if file.endswith('.png') and 'fake_B' in file:
        if True:
            true = file
            # true = file.split('.')[0].split('_')[1] + '.png'
            # true = file.split('_')[0] + '.png'
            pred = cv2.cvtColor(cv2.imread(os.path.join(results_dir, file)), cv2.COLOR_BGR2RGB)
            high = cv2.cvtColor(cv2.imread(os.path.join(true_img_dir, true)), cv2.COLOR_BGR2RGB)

            normed = lambda x: (x - x.min()) / (x.max() - x.min())
            pred = normed(pred)
            high = normed(high)

            cur_rmse = mean_squared_error(pred, high)
            cur_ssim = ssim(pred, high, data_range=high.max() - high.min(), multichannel=True, win_size=11)
            ssim_sum += cur_ssim
            rmse_sum += cur_rmse
            f.write("Image: {}, SSIM = {}, RMSE = {}\n".format(true, cur_ssim, cur_rmse))

    f.write("Mean SSIM = {}, Mean RMSE = {}\n".format(ssim_sum / len(os.listdir(true_img_dir)),
                                                      rmse_sum / len(os.listdir(true_img_dir))))

    f.close()


if __name__ == '__main__':
    compute_ssim_rmse()
    # validate_params()
    # clahe()