import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * F.softplus(x).tanh()


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, act='mish'):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Mish() if act == 'mish' else nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class BasicResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(BasicResBlock, self).__init__()
        self.conv1 = BasicConv(in_channels, mid_channels, 1)
        self.conv2 = BasicConv(mid_channels, in_channels, 3)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layer, is_first=True):
        super(ResBlock, self).__init__()
        self.conv1 = BasicConv(in_channels, out_channels, 3, stride=2)
        if is_first:
            self.conv2 = BasicConv(out_channels, out_channels, 1)
            self.conv3 = BasicConv(out_channels, out_channels, 1)
            self.basic_res_blocks = nn.Sequential(
                *[BasicResBlock(out_channels, out_channels // 2) for _ in range(num_layer)]
            )
            self.conv4 = BasicConv(out_channels, out_channels, 1)
            self.conv5 = BasicConv(out_channels * 2, out_channels, 1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels // 2, 1)
            self.conv3 = BasicConv(out_channels, out_channels // 2, 1)
            self.basic_res_blocks = nn.Sequential(
                *[BasicResBlock(out_channels // 2, out_channels // 2) for _ in range(num_layer)]
            )
            self.conv4 = BasicConv(out_channels // 2, out_channels // 2, 1)
            self.conv5 = BasicConv(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.conv2(x)
        x2 = self.conv3(x)
        x2 = self.basic_res_blocks(x2)
        x2 = self.conv4(x2)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv5(x)
        return x


class CSPDarknet53(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CSPDarknet53, self).__init__()
        inplanes = 32
        feature_channels = [64, 128, 256, 512]
        num_layers = [1, 2, 8, 8, 4]
        self.conv1 = BasicConv(in_channels, inplanes, kernel_size=3, stride=1)
        self.res_block1 = ResBlock(inplanes, feature_channels[0], num_layers[0], is_first=True)
        self.res_block2 = ResBlock(feature_channels[0], feature_channels[1], num_layers[1], is_first=False)
        self.res_block3 = ResBlock(feature_channels[1], feature_channels[2], num_layers[2], is_first=False)
        self.res_block4 = ResBlock(feature_channels[2], feature_channels[3], num_layers[3], is_first=False)
        self.res_block5 = ResBlock(feature_channels[3], out_channels, num_layers[4], is_first=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        out1 = self.res_block3(x)
        out2 = self.res_block4(out1)
        out3 = self.res_block5(out2)

        return [out1, out2, out3]