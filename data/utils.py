""" Dataset utilities """

from torchvision import transforms


def MakeSquared(img, dim=320):
    img_H = img.size[0]
    img_W = img.size[1]

    # Resize
    smaller_dimension = 0 if img_H < img_W else 1
    larger_dimension = 1 if img_H < img_W else 0
    new_smaller_dimension = int(dim * img.size[smaller_dimension] / img.size[larger_dimension])
    if smaller_dimension == 1:
        img = transforms.functional.resize(img, (new_smaller_dimension, dim))
    else:
        img = transforms.functional.resize(img, (dim, new_smaller_dimension))

    # pad
    diff = dim - new_smaller_dimension
    pad_1 = int(diff / 2)
    pad_2 = diff - pad_1

    # RGBmean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    # fill = tuple([round(x) for x in RGBmean])

    if smaller_dimension == 0:
        img = transforms.functional.pad(img, (pad_1, 0, pad_2, 0), padding_mode='constant')
    else:
        img = transforms.functional.pad(img, (0, pad_1, 0, pad_2), padding_mode='constant')

    return img