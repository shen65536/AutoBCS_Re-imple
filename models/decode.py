import torch
import torch.nn as nn

import models


class oct_decode(nn.Module):
    def __init__(self, num_in, num_out, alpha_x=0.5, alpha_y=0.5, bi_linear=False):
        super(oct_decode, self).__init__()

        if bi_linear:
            self.up = models.oct_up(num_in, alpha_x=0.5, bi_linear=True)
        else:
            self.up = models.oct_up(num_in, alpha_x=0.5, bi_linear=False)

        self.DecodeConv1 = models.oct_conv(2 * num_in, num_in, alpha_x, alpha_y)
        self.DecodeConv2 = models.oct_conv(num_in, num_out, alpha_x, alpha_y)

    def forward(self, x_h1, x_l1, x_h2, x_l2):
        x_h1, x_l1 = self.up(x_h1, x_l1)
        x_h = torch.cat([x_h1, x_h2], dim=1)
        x_l = torch.cat([x_l1, x_l2], dim=1)
        y_h, y_l = self.DecodeConv1(x_h, x_l)
        y_h, y_l = self.DecodeConv2(y_h, y_l)
        return y_h, y_l
