""" Network architecture definitions """

import torch.nn as nn
import torchvision.models as tvms


# U-Net based auto-encoder with bilinear upsampling
class UNetBilinearUpsample(nn.Module):
    """ Input size assumed to be 320x320p """

    def __init__(self):
        super(UNetBilinearUpsample, self).__init__()

        self.ds1 = nn.Sequential(*[nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True)])
        self.ds2 = nn.Sequential(
            *[nn.ConstantPad2d((1, 0, 1, 0), 0), nn.Conv2d(16, 16, 3, stride=2), nn.BatchNorm2d(16),
              nn.ReLU(inplace=True)])  # 160x160

        self.ds3 = nn.Sequential(*[nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True)])
        self.ds4 = nn.Sequential(
            *[nn.ConstantPad2d((1, 0, 1, 0), 0), nn.Conv2d(32, 32, 3, stride=2), nn.BatchNorm2d(32),
              nn.ReLU(inplace=True)])  # 80x80

        self.ds5 = nn.Sequential(*[nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)])
        self.ds6 = nn.Sequential(
            *[nn.ConstantPad2d((1, 0, 1, 0), 0), nn.Conv2d(64, 64, 3, stride=2), nn.BatchNorm2d(64),
              nn.ReLU(inplace=True)])  # 40x40

        self.ds7 = nn.Sequential(*[nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)])
        self.ds8 = nn.Sequential(
            *[nn.ConstantPad2d((1, 0, 1, 0), 0), nn.Conv2d(128, 128, 3, stride=2), nn.BatchNorm2d(128),
              nn.ReLU(inplace=True)])  # 20x20

        self.us1 = nn.Sequential(*[nn.Upsample(mode='bilinear', scale_factor=2), nn.ConstantPad2d((1, 1, 1, 1), 0),
                                   nn.Conv2d(128, 64, 3, stride=1)])  # 40x40
        self.us2 = nn.Sequential(*[nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True)])  # 40x40

        self.us3 = nn.Sequential(*[nn.Upsample(mode='bilinear', scale_factor=2), nn.ConstantPad2d((1, 1, 1, 1), 0),
                                   nn.Conv2d(64, 32, 3, stride=1)])  # 80x80
        self.us4 = nn.Sequential(*[nn.ReLU(inplace=True), nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True)])  # 80x80

        self.us5 = nn.Sequential(*[nn.Upsample(mode='bilinear', scale_factor=2), nn.ConstantPad2d((1, 1, 1, 1), 0),
                                   nn.Conv2d(32, 16, 3, stride=1)])  # 160x160
        self.us6 = nn.Sequential(*[nn.ReLU(inplace=True), nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16),
                                   nn.ReLU(inplace=True)])  # 160x160

        self.us7 = nn.Sequential(*[nn.Upsample(mode='bilinear', scale_factor=2), nn.ConstantPad2d((1, 1, 1, 1), 0),
                                   nn.Conv2d(16, 8, 3, stride=1)])  # 320x320
        self.us8 = nn.Sequential(*[nn.ReLU(inplace=True), nn.Conv2d(8, 3, 3, padding=1)])  # 320x320

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ds_12 = self.ds2(self.ds1(x))
        ds_34 = self.ds4(self.ds3(ds_12))
        ds_56 = self.ds6(self.ds5(ds_34))
        ds_78 = self.ds8(self.ds7(ds_56))

        us_12 = self.us2(self.us1(ds_78) + ds_56)
        us_34 = self.us4(self.us3(us_12) + ds_34)
        us_56 = self.us6(self.us5(us_34) + ds_12)
        us_78 = self.us8(self.us7(us_56))

        return self.sigmoid(us_78) + x


# U-Net based auto-encoder with transpose convolutions
class UNetTransposeConv(nn.Module):
    """ Input size assumed to be 320x320p """

    def __init__(self):
        super(UNetTransposeConv, self).__init__()

        self.ds1 = nn.Sequential(*[nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True)])
        self.ds2 = nn.Sequential(
            *[nn.ConstantPad2d((1, 0, 1, 0), 0), nn.Conv2d(16, 16, 3, stride=2), nn.BatchNorm2d(16),
              nn.ReLU(inplace=True)])  # 160x160

        self.ds3 = nn.Sequential(*[nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True)])
        self.ds4 = nn.Sequential(
            *[nn.ConstantPad2d((1, 0, 1, 0), 0), nn.Conv2d(32, 32, 3, stride=2), nn.BatchNorm2d(32),
              nn.ReLU(inplace=True)])  # 80x80

        self.ds5 = nn.Sequential(*[nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)])
        self.ds6 = nn.Sequential(
            *[nn.ConstantPad2d((1, 0, 1, 0), 0), nn.Conv2d(64, 64, 3, stride=2), nn.BatchNorm2d(64),
              nn.ReLU(inplace=True)])  # 40x40

        self.ds7 = nn.Sequential(*[nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)])
        self.ds8 = nn.Sequential(
            *[nn.ConstantPad2d((1, 0, 1, 0), 0), nn.Conv2d(128, 128, 3, stride=2), nn.BatchNorm2d(128),
              nn.ReLU(inplace=True)])  # 20x20

        self.us1 = nn.Sequential(*[nn.ConvTranspose2d(128, 64, 2, stride=2)])  # 40x40
        self.us2 = nn.Sequential(*[nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True)])  # 40x40

        self.us3 = nn.Sequential(*[nn.ConvTranspose2d(64, 32, 2, stride=2)])  # 80x80
        self.us4 = nn.Sequential(*[nn.ReLU(inplace=True), nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True)])  # 80x80

        self.us5 = nn.Sequential(*[nn.ConvTranspose2d(32, 16, 2, stride=2)])  # 160x160
        self.us6 = nn.Sequential(*[nn.ReLU(inplace=True), nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16),
                                   nn.ReLU(inplace=True)])  # 160x160

        self.us7 = nn.Sequential(*[nn.ConvTranspose2d(16, 8, 2, stride=2)])  # 320x320
        self.us8 = nn.Sequential(*[nn.ReLU(inplace=True), nn.Conv2d(8, 3, 3, padding=1)])  # 320x320

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ds_12 = self.ds2(self.ds1(x))
        ds_34 = self.ds4(self.ds3(ds_12))
        ds_56 = self.ds6(self.ds5(ds_34))
        ds_78 = self.ds8(self.ds7(ds_56))

        us_12 = self.us2(self.us1(ds_78) + ds_56)
        us_34 = self.us4(self.us3(us_12) + ds_34)
        us_56 = self.us6(self.us5(us_34) + ds_12)
        us_78 = self.us8(self.us7(us_56))

        return self.sigmoid(us_78) + x
