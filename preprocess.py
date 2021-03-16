import os
from PIL import Image

import utils


def get_jpg_paths(args):
    path = args.images_path
    ps = []
    for root, dirs, files in os.walk(path, topdown=True):
        for name in files:
            _, extension = os.path.splitext(name)
            if extension == ".jpg":
                ps.append(os.path.join(root, name))
    return ps


def jpg_crop(path, args):
    count = 1
    for i, path in enumerate(path):
        image = Image.open(path)
        w = image.size[0]
        h = image.size[1]

        idx1 = range(0, w, args.block_size)
        idx2 = range(0, h, args.block_size)

        for a in idx1:
            for b in idx2:
                crop_box = (a, b, a + args.block_size, b + args.block_size)
                temp = image.crop(crop_box)
                temp.save("{}/preprocessed_images/({}).jpg".format(args.train_path, count))
                count += 1


if __name__ == "__main__":
    print("=> Now preprocessing.")
    my_args = utils.args_set()
    paths = get_jpg_paths(my_args)
    jpg_crop(paths, my_args)
    print("=> Done.")
