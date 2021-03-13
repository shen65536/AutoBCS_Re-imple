import torch.nn as nn
import utils


class init_net(nn.Module):
    def __init__(self, args):
        super(init_net, self).__init__()
        self.args = args
        self.channels = args.channels
        self.block_size = args.block_size
        self.sample_points = int(args.ratio * args.block_size ** 2)

        self.sample = nn.Conv2d(self.channels, self.sample_points, kernel_size=self.block_size,
                                stride=self.block_size, padding=0, bias=False)
        nn.init.normal_(self.sample.weight, mean=0.0, std=0.028)
        self.init = nn.Conv2d(self.sample_points, self.block_size ** 2, kernel_size=1, bias=False)

    def forward(self, x):
        y = self.sample(x)
        y = self.init(y)
        y = utils.reshape(y, self.args)
        return y
