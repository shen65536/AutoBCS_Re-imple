import torch.nn as nn
import torch.nn.functional as F

import models


class oct_net(nn.Module):
    def __init__(self, args):
        super(oct_net, self).__init__()
        self.num_inputs = 0
        self.num_outputs = 0
        self.encode_layers = []
        self.decode_layers = []
        self.initial_num_layers = 64
        self.encoding_depth = args.depth
        self.temp = list(range(1, self.encoding_depth + 1))

        self.input_oct = models.oct_encode(1, self.initial_num_layers, alpha_x=1, alpha_y=0.5)

        for encodingLayer in self.temp:
            if encodingLayer == 1:
                self.num_outputs = self.initial_num_layers * 2 ** (encodingLayer - 1)
                self.encode_layers.append(models.oct_encode(self.initial_num_layers, self.num_outputs))
            else:
                self.num_outputs = self.initial_num_layers * 2 ** (encodingLayer - 1)
                self.encode_layers.append(models.oct_encode(self.num_outputs // 2, self.num_outputs))

        self.encode_layers = nn.ModuleList(self.encode_layers)

        self.mid_layer = models.oct_mid(self.num_outputs)
        initial_decode_num_ch = self.num_outputs

        for decodingLayer in self.temp:
            if decodingLayer == self.encoding_depth:
                self.num_inputs = initial_decode_num_ch // 2 ** (decodingLayer - 1)
                self.decode_layers.append(models.oct_decode(self.num_inputs, self.num_inputs))
            else:
                self.num_inputs = initial_decode_num_ch // 2 ** (decodingLayer - 1)
                self.decode_layers.append(models.oct_decode(self.num_inputs, self.num_inputs // 2))

        self.decode_layers = nn.ModuleList(self.decode_layers)
        self.final_oct = models.oct_conv(self.num_inputs, self.num_inputs, alpha_x=0.5, alpha_y=1)

    def forward(self, x):
        input_x = x
        x_h = x
        x_l = 0
        names = self.__dict__
        x_h, x_l, = self.input_oct(x_h, x_l)
        temp = list(range(1, self.encoding_depth + 1))

        for encodingLayer in temp:
            temp_conv = self.encode_layers[encodingLayer - 1]
            x_h, x_l = temp_conv(x_h, x_l)
            names['EncodeX' + str(encodingLayer)] = x_h, x_l

            x_h = F.max_pool2d(x_h, 2)
            x_l = F.max_pool2d(x_l, 2)

        x_h, x_l = self.mid_layer(x_h, x_l)

        for decodingLayer in temp:
            temp_conv = self.decode_layers[decodingLayer - 1]
            x_h2, x_l2 = names['EncodeX' + str(self.encoding_depth - decodingLayer + 1)]
            x_h, x_l = temp_conv(x_h, x_l, x_h2, x_l2)

        x = self.final_oct(x_h, x_l)
        x = x + input_x
        return x
