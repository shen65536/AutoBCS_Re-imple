import torch.nn as nn


class oct_up(nn.Module):
    def __init__(self, num_in, alpha_x=0.5, bi_linear=False):
        super(oct_up, self).__init__()
        self.In_H = int(num_in * alpha_x)
        self.In_L = (num_in - self.In_H)
        if bi_linear:
            self.up_HH = nn.Upsample(scale_factor=2, mode='nearest')
            self.up_LH = nn.ConvTranspose2d(self.In_L, self.In_H, 4, stride=4)
            self.BN_H = nn.BatchNorm2d(self.In_H)
            self.ReLU_H = nn.ReLU()
            self.up_LL = nn.Upsample(scale_factor=2, mode='nearest')
            self.up_HL = nn.Conv2d(self.In_H, self.In_L, 3, padding=1)
            self.BN_L = nn.BatchNorm2d(self.In_L)
            self.ReLU_L = nn.ReLU()
        else:
            self.up_HH = nn.ConvTranspose2d(self.In_H, self.In_H, 2, stride=2)
            self.up_LH = nn.ConvTranspose2d(self.In_L, self.In_H, 4, stride=4)
            self.BN_H = nn.BatchNorm2d(self.In_H)
            self.ReLU_H = nn.ReLU()
            self.up_LL = nn.ConvTranspose2d(self.In_L, self.In_L, 2, stride=2)
            self.up_HL = nn.Conv2d(self.In_H, self.In_L, 3, padding=1)
            self.BN_L = nn.BatchNorm2d(self.In_L)
            self.ReLU_L = nn.ReLU()

    def forward(self, x_h, x_l):
        y_h1 = self.up_HH(x_h)
        y_h2 = self.up_LH(x_l)
        y_l1 = self.up_LL(x_l)
        y_l2 = self.up_HL(x_h)
        y_l = y_l1 + y_l2
        y_h = y_h1 + y_h2
        y_h = self.BN_H(y_h)
        y_h = self.ReLU_H(y_h)
        y_l = self.BN_L(y_l)
        y_l = self.ReLU_L(y_l)
        return y_h, y_l
