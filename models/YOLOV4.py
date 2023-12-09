import torch
import torch.nn as nn

from models.CSPDarknet53 import BasicConv, CSPDarknet53


class Conv3Block(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(Conv3Block, self).__init__()
        self.conv1 = BasicConv(in_channels, mid_channels[0], 1, act='leaky')
        self.conv2 = BasicConv(mid_channels[0], mid_channels[1], 3, act='leaky')
        self.conv3 = BasicConv(mid_channels[1], mid_channels[0], 1, act='leaky')

    def forward(self, x):
        return self.conv3(self.conv2(self.conv1(x)))


class Conv5Block(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(Conv5Block, self).__init__()
        self.conv1 = BasicConv(in_channels, mid_channels[0], 1, act='leaky')
        self.conv2 = BasicConv(mid_channels[0], mid_channels[1], 3, act='leaky')
        self.conv3 = BasicConv(mid_channels[1], mid_channels[0], 1, act='leaky')
        self.conv4 = BasicConv(mid_channels[0], mid_channels[1], 3, act='leaky')
        self.conv5 = BasicConv(mid_channels[1], mid_channels[0], 1, act='leaky')

    def forward(self, x):
        return self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))


class SPP(nn.Module):
    def __init__(self, pool_sizes):
        super(SPP, self).__init__()
        self.pool1 = nn.MaxPool2d(pool_sizes[0], 1, pool_sizes[0] // 2)
        self.pool2 = nn.MaxPool2d(pool_sizes[1], 1, pool_sizes[1] // 2)
        self.pool3 = nn.MaxPool2d(pool_sizes[2], 1, pool_sizes[2] // 2)

    def forward(self, x):
        pool1 = self.pool1(x)
        pool2 = self.pool2(x)
        pool3 = self.pool3(x)
        out = torch.cat([pool3, pool2, pool1, x], dim=1)
        return out


class YOLOV4(nn.Module):
    def __init__(self, B=3, C=80):
        super(YOLOV4, self).__init__()
        in_channels = 3
        out_channels = 1024
        yolo_channels = (5 + C) * B

        self.backbone = CSPDarknet53(in_channels, out_channels)

        self.conv3block1 = Conv3Block(out_channels, mid_channels=[512, 1024])
        self.spp = SPP(pool_sizes=[5, 9, 13])
        self.conv3block2 = Conv3Block(2048, mid_channels=[512, 1024])

        self.conv1 = BasicConv(512, 256, 1, act='leaky')
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = BasicConv(512, 256, 1, act='leaky')

        self.conv5block1 = Conv5Block(512, mid_channels=[256, 512])

        self.conv3 = BasicConv(256, 128, 1, act='leaky')
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = BasicConv(256, 128, 1, act='leaky')

        # end pretrain layer
        self.conv5block2 = Conv5Block(256, mid_channels=[128, 256])

        # yolo1 layer
        self.yolo1_conv1 = BasicConv(128, 256, 3, act='leaky')
        self.yolo1_conv2 = nn.Conv2d(256, yolo_channels, 1)

        self.conv5 = BasicConv(128, 256, 3, stride=2, act='leaky')
        self.conv5block3 = Conv5Block(512, mid_channels=[256, 512])

        # yolo2 layer
        self.yolo2_conv1 = BasicConv(256, 512, 3, act='leaky')
        self.yolo2_conv2 = nn.Conv2d(512, yolo_channels, 1)

        self.conv6 = BasicConv(256, 512, 3, stride=2, act='leaky')
        self.conv5block4 = Conv5Block(1024, mid_channels=[512, 1024])

        # yolo3 layer
        self.yolo3_conv1 = BasicConv(512, 1024, 3, act='leaky')
        self.yolo3_conv2 = nn.Conv2d(1024, yolo_channels, 1)

    def forward(self, x):
        x1, x2, x3 = self.backbone(x)
        x3 = self.conv3block1(x3)
        x3 = self.spp(x3)
        x3 = self.conv3block2(x3)

        x4 = self.conv1(x3)
        x4 = self.up1(x4)
        x2 = self.conv2(x2)
        x4 = torch.cat([x2, x4], dim=1)

        x4 = self.conv5block1(x4)

        x5 = self.conv3(x4)
        x5 = self.up2(x5)
        x1 = self.conv4(x1)
        x5 = torch.cat([x1, x5], dim=1)

        x5 = self.conv5block2(x5)

        yolo1 = self.yolo1_conv1(x5)
        yolo1 = self.yolo1_conv2(yolo1)

        x5 = self.conv5(x5)
        x5 = torch.cat([x5, x4], dim=1)
        x5 = self.conv5block3(x5)

        yolo2 = self.yolo2_conv1(x5)
        yolo2 = self.yolo2_conv2(yolo2)

        x5 = self.conv6(x5)
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.conv5block4(x5)

        yolo3 = self.yolo3_conv1(x5)
        yolo3 = self.yolo3_conv2(yolo3)

        return [yolo1, yolo2, yolo3]