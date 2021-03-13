

def reshape(x, args):
    if x.size()[0] == args.batch_size:
        y = x.view(args.batch_size, args.channels, args.block_size, args.block_size)
    else:
        y = x.view(1, args.channels, args.block_size, args.block_size)
    return y
