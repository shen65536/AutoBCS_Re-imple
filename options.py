import argparse


def args_set():
    arg = argparse.ArgumentParser(description="Options of AutoBCS.")
    arg.add_argument("--train_path", default="./dataset/train")
    arg.add_argument("--test_path", default="./dataset/test")
    arg.add_argument("--block_size", default=32, type=int)
    arg.add_argument("--batch_size", default=10, type=int)
    arg.add_argument("--ratio", default=0.15, type=float)
    arg.add_argument("--epochs", default=100, type=int)
    arg.add_argument("--channels", default=1, type=int)
    arg.add_argument("--depth", default=2, type=int)
    args = arg.parse_args()
    return args
