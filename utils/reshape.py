

def reshape(x, args):
    y = x.view(args.batch_size, args.channels, args.block_size, args.block_size)
    return y
