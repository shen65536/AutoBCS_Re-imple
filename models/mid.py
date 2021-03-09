import torch.nn as nn

import models


class oct_mid(nn.Module):
    def __init__(self, num_ch, alpha_x=0.5, alpha_y=0.5):
        super(oct_mid, self).__init__()
        self.MidConv1 = models.oct_conv(num_ch, 2 * num_ch, alpha_x, alpha_y)
        self.MidConv2 = models.oct_conv(2 * num_ch, num_ch, alpha_x, alpha_y)

    def forward(self, x_h, x_l):
        y_h, y_l = self.MidConv1(x_h, x_l)
        y_h, y_l = self.MidConv2(y_h, y_l)
        return y_h, y_l
