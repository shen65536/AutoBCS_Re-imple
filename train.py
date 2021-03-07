import torch
import torch.nn as nn
import torch.optim as optim

import model
import utils
import options


def train(args):
    init_net = model.init_net(args)
    deep_net = model.oct_net(args)

    print("Data loading.")
    loader = utils.loader(args)
    dataset = loader.load()
    print("Data loaded.")

    criterion = nn.L1Loss()
    optimizer_init = optim.Adam(init_net.parameters())
    optimizer_deep = optim.Adam(deep_net.parameters())
    scheduler_init = optim.lr_scheduler.MultiStepLR(optimizer_init, milestones=[50, 80], gamma=0.1)
    scheduler_deep = optim.lr_scheduler.MultiStepLR(optimizer_deep, milestones=[50, 80], gamma=0.1)

    print("Train start.")
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
    print("Train end.")
    torch.save(init_net.state_dict(), "./trained_models/init_net_ratio{}.pth".format(args.ratio))
    torch.save(deep_net.state_dict(), "./trained_models/deep_net_ratio{}.pth".format(args.ratio))
    print("Trained model saved.")


if __name__ == "__main__":
    my_args = options.args_set()
    train(my_args)
