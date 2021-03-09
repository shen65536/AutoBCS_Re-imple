import argparse


def args_set():
    ratio = 0.15
    epochs = 100
    channels = 1

    arg = argparse.ArgumentParser(description="Options of AutoBCS.")
    arg.add_argument("--init_state_dict", default="./trained_models/init_net_ratio{}.pth".format(ratio))
    arg.add_argument("--deep_state_dict", default="./trained_models/deep_net_ratio{}.pth".format(ratio))
    arg.add_argument("--images_path", default="./dataset/images")
    arg.add_argument("--train_path", default="./dataset/train")
    arg.add_argument("--test_path", default="./dataset/test")
    arg.add_argument("--block_size", default=32, type=int)
    arg.add_argument("--batch_size", default=10, type=int)
    arg.add_argument("--ratio", default=ratio, type=float)
    arg.add_argument("--epochs", default=epochs, type=int)
    arg.add_argument("--channels", default=channels, type=int)
    arg.add_argument("--depth", default=2, type=int)
    args = arg.parse_args()
    return args
