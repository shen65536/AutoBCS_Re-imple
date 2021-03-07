import torch.nn as nn

import model


class oct_encode(nn.Module):
    def __init__(self, num_in, num_out, alpha_x=0.5, alpha_y=0.5):
        super(oct_encode, self).__init__()
        self.EncodeConv1 = model.oct_conv(num_in, num_out, alpha_x, alpha_y)

        if alpha_x == 1:
            alpha_x = 0.5
        self.EncodeConv2 = model.oct_conv(num_out, num_out, alpha_x, alpha_y)

    def forward(self, x_h, x_l):
        y_h, y_l = self.EncodeConv1(x_h, x_l)
        y_h, y_l = self.EncodeConv2(y_h, y_l)
        return y_h, y_l
