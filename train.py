import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchsummary.torchsummary

import models
import utils


def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    init_net = nn.DataParallel(models.init_net(args)).to(device)
    deep_net = nn.DataParallel(models.oct_net(args)).to(device)

    print("Data loading.")
    dataset = utils.loader(args)
    print("Data loaded.")

    criterion = nn.L1Loss().to(device)
    optimizer_init = optim.Adam(init_net.parameters())
    optimizer_deep = optim.Adam(deep_net.parameters())
    scheduler_init = optim.lr_scheduler.MultiStepLR(optimizer_init, milestones=[50, 80], gamma=0.1)
    scheduler_deep = optim.lr_scheduler.MultiStepLR(optimizer_deep, milestones=[50, 80], gamma=0.1)

    print("Train start.")
    time_start = time.time()

    if os.path.exists(args.init_state_dict) and os.path.exists(args.deep_state_dict):
        if torch.cuda.is_available():
            checkpoint_init = torch.load(args.init_state_dict)
            checkpoint_deep = torch.load(args.deep_state_dict)
        else:
            checkpoint_init = torch.load(args.init_state_dict, map_location="cpu")
            checkpoint_deep = torch.load(args.deep_state_dict, map_location="cpu")

        init_net.load_state_dict(checkpoint_init["model"])
        optimizer_init.load_state_dict(checkpoint_init["optimizer"])

        deep_net.load_state_dict(checkpoint_deep["model"])
        optimizer_deep.load_state_dict(checkpoint_deep["optimizer"])

        start_epoch = checkpoint_deep["epoch"]
        print("Success loading epoch {}".format(start_epoch))
    else:
        start_epoch = 0
        print("No saved model, start epoch = 0.")

    for epoch in range(start_epoch, args.epochs):
        for idx, item in enumerate(dataset):
            x, _ = item
            x = x.to(device)

            optimizer_init.zero_grad()
            optimizer_deep.zero_grad()

            init_x = init_net(x)
            init_x = utils.reshape(init_x, args)
            deep_x = deep_net(init_x)

            loss_init = criterion(x, init_x)
            loss_deep = criterion(x, deep_x)

            loss_init.backward(retain_graph=True)
            loss_deep.backward()

            optimizer_init.step()
            optimizer_deep.step()

            use_time = time.time() - time_start
            if (idx + 1) % 20 == 0:
                print("=> epoch: {}, batch: {}, Loss1: {:.4f}, Loss2: {:.4f}, lr1: {}, lr2: {}, used time: {:.4f}"
                      .format(epoch + 1, idx + 1, loss_init.item(), loss_deep.item(),
                              optimizer_init.param_groups[0]['lr'], optimizer_deep.param_groups[0]['lr'], use_time))

        scheduler_init.step()
        scheduler_deep.step()
        state_init = {"model": init_net.state_dict(), "optimizer": optimizer_init.state_dict()}
        state_deep = {"model": deep_net.state_dict(), "optimizer": optimizer_deep.state_dict(), "epoch": epoch + 1}
        torch.save(state_init, args.init_state_dict)
        torch.save(state_deep, args.deep_state_dict)
        print("Check point of epoch {} saved.".format(epoch + 1))

    print("Train end.")
    torchsummary.summary(init_net, (1, 32, 32))
    torchsummary.summary(deep_net, (1, 32, 32))
    with open("./trained_models/init_net.txt", "w") as f1:
        f1.write(torchsummary.summary(init_net, (1, 32, 32)))
    with open("./trained_models/deep_net.txt", "w") as f2:
        f2.write(torchsummary.summary(deep_net, (1, 32, 32)))


if __name__ == "__main__":
    my_args = utils.args_set()
    train(my_args)
