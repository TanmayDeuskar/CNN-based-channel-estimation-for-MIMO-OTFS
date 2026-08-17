import torch.nn as nn


class Residual2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ChannelEst2DNet(nn.Module):
    def __init__(self, in_channels, Nt):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv_mid = nn.Conv2d(32, 64, 3, padding=1)
        self.bn_mid = nn.BatchNorm2d(64)
        self.res1 = Residual2D(64)
        self.res2 = Residual2D(64)
        self.res3 = Residual2D(64)
        self.conv_red = nn.Conv2d(64, 32, 3, padding=1)
        self.bn_red = nn.BatchNorm2d(32)
        self.conv_out = nn.Conv2d(32, 2 * Nt, 3, padding=1)

    def forward(self, x):
        out = self.relu(self.bn_in(self.conv_in(x)))
        out = self.relu(self.bn_mid(self.conv_mid(out)))
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.relu(self.bn_red(self.conv_red(out)))
        return self.conv_out(out)
