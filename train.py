import argparse
import torch.nn as nn
import torch.optim as optim

import model
import utils

arg = argparse.ArgumentParser(description="Options of AutoBCS.")
arg.add_argument("--train_path", default="./dataset/train")
arg.add_argument("--test_path", default="./dataset/test")
arg.add_argument("--block_size", default=32, type=int)
arg.add_argument("--batch_size", default=10, type=int)
arg.add_argument("--ratio", default=0.15, type=float)
arg.add_argument("--epochs", default=100, type=int)
arg.add_argument("--channels", default=3, type=int)
args = arg.parse_args()


def train(init_net, deep_net):
    print("Data loading")
    loader = utils.loader(args)
    dataset = loader.load()
    print("Data loaded.")

    criterion = nn.L1Loss()
    optimizer_init = optim.Adam(init_net.parameters())
    optimizer_deep = optim.Adam(deep_net.parameters())
    scheduler_init = optim.lr_scheduler.MultiStepLR(optimizer_init, milestones=[50, 80], gamma=0.1)
    scheduler_deep = optim.lr_scheduler.MultiStepLR(optimizer_deep, milestones=[50, 80], gamma=0.1)

    for epoch in range(args.epochs):
        for idx, item in enumerate(dataset):
            x, _ = item
            optimizer_init.zero_grad()
            optimizer_deep.zero_grad()

            init_x = init_net(x)
            deep_x = deep_net(init_x)

            loss_init = criterion(x, init_x)
            loss_deep = criterion(init_x, deep_x)

            loss_init.backward()
            loss_deep.backward()

            optimizer_init.step()
            optimizer_deep.step()

            print("=> ...")

    scheduler_init.step()
    scheduler_deep.step()


if __name__ == "__main__":
    train(model.init_net(args), model.oct_net(args))
