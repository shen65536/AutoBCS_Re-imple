import torch.nn as nn
import torch.nn.functional as F


class oct_conv(nn.Module):
    def __init__(self, num_in, num_out, alpha_x, alpha_y, ks=3, pd=1):
        super(oct_conv, self).__init__()

        self.In_H = int(num_in * alpha_x)
        self.In_L = (num_in - self.In_H)
        self.Out_H = int(num_out * alpha_y)
        self.Out_L = (num_out - self.Out_H)
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y

        if alpha_x == 1:
            # only for the input layer.
            self.convHH = nn.Conv2d(self.In_H, self.Out_H, ks, padding=pd)
            self.convHL = nn.Conv2d(self.In_H, self.Out_L, ks, padding=pd)
            self.BN_HH = nn.BatchNorm2d(self.Out_H)
            self.BN_HL = nn.BatchNorm2d(self.Out_L)
            self.ReLU_H = nn.ReLU()
            self.ReLU_L = nn.ReLU()
        elif alpha_y == 1:
            # only for the final output layer.
            self.convHH = nn.Conv2d(self.In_H, self.Out_H, ks, padding=pd)
            self.convLH = nn.Conv2d(self.In_L, self.Out_H, ks, padding=pd)
            self.up = nn.ConvTranspose2d(self.Out_H, self.Out_H, 2, stride=2)
            self.BN_HH = nn.BatchNorm2d(self.Out_H)
            self.BN_LH = nn.BatchNorm2d(self.Out_H)
            self.ReLU_H = nn.ReLU()
            self.FinalConv = nn.Conv2d(self.Out_H, 1, 1, stride=1, padding=0)
        else:
            # mid layers.
            self.convHH = nn.Conv2d(self.In_H, self.Out_H, ks, padding=pd)
            self.convLL = nn.Conv2d(self.In_L, self.Out_L, ks, padding=pd)
            self.convHL = nn.Conv2d(self.In_H, self.Out_L, ks, padding=pd)
            self.convLH = nn.Conv2d(self.In_L, self.Out_H, ks, padding=pd)
            self.up = nn.ConvTranspose2d(self.Out_H, self.Out_H, 2, stride=2)
            self.BN_HH = nn.BatchNorm2d(self.Out_H)
            self.BN_LH = nn.BatchNorm2d(self.Out_H)
            self.ReLU_H = nn.ReLU()
            self.BN_LL = nn.BatchNorm2d(self.Out_L)
            self.BN_HL = nn.BatchNorm2d(self.Out_L)
            self.ReLU_L = nn.ReLU()

    def forward(self, x_h, x_l):
        if self.alpha_x == 1:
            y_h = self.ReLU_H(self.BN_HH(self.convHH(x_h)))
            y_l = self.ReLU_L(self.BN_HL(self.convHL(F.avg_pool2d(x_h, 2))))

            return y_h, y_l
        elif self.alpha_y == 1:
            y_h1 = self.BN_HH(self.convHH(x_h))
            y_h2 = self.BN_LH(self.up(self.convLH(x_l)))
            y_h = self.FinalConv(self.ReLU_H(y_h1 + y_h2))

            return y_h
        else:
            y_h = self.ReLU_H(self.BN_HH(self.convHH(x_h)) + self.BN_LH(self.up(self.convLH(x_l))))
            y_l = self.ReLU_L(self.BN_LL(self.BN_HL(self.convHL(F.avg_pool2d(x_h, 2))) + self.convLL(x_l)))

            return y_h, y_l
