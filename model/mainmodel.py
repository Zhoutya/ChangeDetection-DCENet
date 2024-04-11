# -*- coding: utf-8 -*-
'''
    DCENet: Diff-Feature Contrast Enhancement Network for Semi-supervised Hyperspectral Change Detection
    Dataset: Farmland(yancheng), Hermiston
    Author: zhouty
    Time:2024.3.16
'''
from model.modules import *
from model.losses import *


class Gaussnoise3D(nn.Module):
    def __init__(self, mean=0, stddev=0.1):
        super(Gaussnoise3D, self).__init__()
        self.mean = mean
        self.stddev = stddev

    def forward(self, x):
        x_noisy = torch.empty_like(x)
        for i in range(x.shape[-1]):
            noise = torch.randn([x.shape[0], x.shape[1], x.shape[2]], device=x.device) * 0.1
            x_noisy[:, :, :, i] = x[:, :, :, i] + noise
        return x_noisy

    def __repr__(self):
        return self.__class__.__name__ + '(mean={},stddev={})'.format(self.mean, self.stddev)


class DCENet(nn.Module):
    # DCENet: DFA Subnet + msa + cl + kl
    def __init__(self, in_fea_num, num_classes=2):
        super(DCENet, self).__init__()
        self.getdiff = DFEDSubnet(in_fea_num)
        self.enhence = Gaussnoise3D()

        self.downa = DownEncoder()
        self.downb = DownEncoder()

        self.fc = nn.Linear(1024, 2)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Sequential(nn.Linear(1024, 256),
                                  nn.ReLU(inplace=True),  # first layer
                                  nn.Linear(256, 1024),
                                  nn.BatchNorm1d(1024, affine=False))  # output layer

        self.PCL = PCLoss(0.1)
        self.KL = KLloss()
        self.num_classes = num_classes

    def test(self, xl1, xl2):
        diff1, diff2 = self.getdiff(xl1, xl2)
        _, _, down = self.downa(diff2)
        down = down.squeeze(2).squeeze(2)
        out = self.fc(down)
        out = self.softmax(out)

        return out.detach()

    def forward(self, xl1, xl2, xu1, xu2):
        if self.training:
            # Differential Fusion Attention Subnetwork
            diffL1, diffL2 = self.getdiff(xl1, xl2)
            diffU1, diffU2 = self.getdiff(xu1, xu2)

            # Add noise
            diffL1_Noise = self.enhence(diffL1)
            diffU1_Noise = self.enhence(diffU1)

            # Multi-scale Kullback-Leibler Divergence
            downl1, downl2, downl = self.downa(diffL2)
            downl = downl.squeeze(2).squeeze(2)
            outl = self.softmax(self.fc(downl))
            downu1, downu2, downu = self.downa(diffU2)
            downu = downu.squeeze(2).squeeze(2)
            outu = self.softmax(self.fc(downu))

            downlN1, downlN2, Nosie_l = self.downb(diffL1_Noise)
            Nosie_l = Nosie_l.squeeze(2).squeeze(2)
            downuN1, downuN2, Nosie_u = self.downb(diffU1_Noise)
            Nosie_u = Nosie_u.squeeze(2).squeeze(2)

            klloss = self.KL(downl1, downlN1) + self.KL(downl2, downlN2) + self.KL(downl, Nosie_l) + \
                     self.KL(downu1, downuN1) + self.KL(downu2, downuN2) + self.KL(downu, Nosie_u)

            #  Feature-enhanced Probabilistic Contrast Loss
            z1 = self.proj(Nosie_l)
            z2 = self.proj(downl)
            z1 = self.softmax(z1)
            z2 = self.softmax(z2)

            p1 = self.proj(Nosie_u)
            p2 = self.proj(downu)
            p1 = self.softmax(p1)
            p2 = self.softmax(p2)

            CLloss = self.PCL(z1, z2) + self.PCL(p1, p2)

            return outl, outu, CLloss, klloss

        else:
            return self.test(xl1, xl2)
