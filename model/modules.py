# -*- coding: utf-8 -*-
from model.attentions import *


def conv3(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.Up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.Up(x)


class EDbranch(nn.Module):
    '只要fuse2的MSA,94.37 的不加ASPP的EDbranch'

    def __init__(self, in_ch):
        super(EDbranch, self).__init__()
        self.conv0 = nn.Conv2d(in_ch, 128, kernel_size=1)
        self.Down1 = conv3(128, 256)
        self.Down2 = conv3(256, 512)
        self.Down3 = conv3(512, 1024)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)

        self.conv1 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1)

    def forward(self, x):
        down0 = self.conv0(x)
        down1 = self.Down1(down0)  # 7 -> 5
        down2 = self.Down2(down1)  # 5 -> 3

        up1 = self.up1(down2)  # 3 -> 5
        fuse1 = self.conv1(torch.cat((up1, down1), dim=1))
        up2 = self.up2(fuse1)  # 5 -> 7
        fuse2 = self.conv2(torch.cat((up2, down0), dim=1))

        return down0, down1, down2, fuse1, fuse2  # 7, 5, 3, 5, 7


class DFEDSubnet(nn.Module):
    def __init__(self, in_ch):
        super(DFEDSubnet, self).__init__()
        self.edbranch1 = EDbranch(in_ch)
        self.edbranch2 = EDbranch(in_ch)

        self.Down1 = conv3(128, 256)
        self.Down2 = conv3(256, 512)
        self.transformerblock = GlobalTransformer(512)

        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)

        self.conv1 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=1)
        self.conv3 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=1)

    def forward(self, x1, x2):
        # two temporal branches
        d11, d12, d13, u11, u12 = self.edbranch1(x1)  # 7, 5, 3, 5, 7
        d21, d22, d23, u21, u22 = self.edbranch1(x2)
        diff1, diff2, diff3, diff4, diff5 = (d11 - d21), (d12 - d22), (d13 - d23), (u11 - u21), (u12 - u22)

        # differential branch
        diff_d1 = self.Down1(diff1)
        fuse1 = self.conv1(torch.cat((diff_d1, diff2), dim=1))
        diff_d2 = self.Down2(fuse1)
        fuse2 = self.conv2(torch.cat((diff_d2, diff3), dim=1))
        fuse2 = self.transformerblock(fuse2)

        up1 = self.up1(fuse2)
        fuse3 = self.conv3(torch.cat((up1, diff4), dim=1))
        up2 = self.up2(fuse3)
        fuse4 = self.conv4(torch.cat((up2, diff5), dim=1))

        return fuse4, fuse4


class DownEncoder(nn.Module):
    def __init__(self):
        super(DownEncoder, self).__init__()
        self.Down1 = conv3(128, 256)
        self.Down2 = conv3(256, 512)
        self.Down3 = conv3(512, 1024)

    def forward(self, x):
        down1 = self.Down1(x)
        down2 = self.Down2(down1)
        down3 = self.Down3(down2)

        return down1, down2, down3
