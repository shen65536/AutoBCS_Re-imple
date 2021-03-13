import time
import torch
import torchvision
import numpy as np
import torch.nn as nn
import scipy.io as scio
from skimage.metrics import structural_similarity as SSIM

import models
import utils

if __name__ == '__main__':
    args = utils.args_set()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tensor2image = torchvision.transforms.ToPILImage()
    batch_size = 1
    PSNR_total = 0
    SSIM_total = 0

    File_No = 100
    Folder_name = "{}/BSD100".format(args.test_path)

    with torch.no_grad():
        init_net = nn.DataParallel(models.init_net(args)).to(device).eval()
        init_net.load_state_dict(torch.load("./trained_models/init_net_ratio{}.pth".format(args.ratio),
                                            map_location='cpu')["model"])

        deep_net = nn.DataParallel(models.oct_net(args)).to(device).eval()
        deep_net.load_state_dict(torch.load("./trained_models/deep_net_ratio{}.pth".format(args.ratio),
                                            map_location='cpu')["model"])

        for i in range(1, File_No + 1):
            name = "{}/({}).mat".format(Folder_name, i)
            x = scio.loadmat(name)['temp3']
            x = torch.from_numpy(np.array(x)).to(device)
            x = x.float()
            ori_x = x

            h = x.size()[0]
            h_lack = 0
            w = x.size()[1]
            w_lack = 0

            if h % args.block_size != 0:
                h_lack = args.block_size - h % args.block_size
                temp_h = torch.zeros(h_lack, w)
                h = h + h_lack
                x = torch.cat((x, temp_h), 0)

            if w % args.block_size != 0:
                w_lack = args.block_size - w % args.block_size
                temp_w = torch.zeros(h, w_lack)
                w = w + w_lack
                x = torch.cat((x, temp_w), 1)

            x = torch.unsqueeze(x, 0)
            x = torch.unsqueeze(x, 0)

            idx_h = range(0, h, args.block_size)
            idx_w = range(0, w, args.block_size)

            num_patches = h * w // (args.block_size ** 2)
            temp = torch.zeros(num_patches, batch_size, args.channels, args.block_size, args.block_size)
            idx1 = range(0, h, args.block_size)
            idx2 = range(0, w, args.block_size)

            start_time = time.time()
            count = 0
            for a in idx1:
                for b in idx2:
                    input = x[:, :, a:a + args.block_size, b:b + args.block_size]
                    output = init_net(input)
                    output = deep_net(output)
                    temp[count, :, :, :, :, ] = output
                    count = count + 1
            end_time = time.time()

            y = torch.zeros(batch_size, args.channels, h, w)

            count = 0
            for a in idx1:
                for b in idx2:
                    y[:, :, a:a + args.block_size, b:b + args.block_size] = temp[count, :, :, :, :]
                    count = count + 1

            image = y[:, :, 0:h - h_lack, 0:w - w_lack]
            image = torch.squeeze(image)

            diff = image.numpy() - ori_x.numpy()
            mse = np.mean(np.square(diff))
            psnr = 10 * np.log10(1 / mse)
            PSNR_total = PSNR_total + psnr

            ssim = SSIM(image.numpy(), ori_x.numpy(), data_range=1)
            SSIM_total = SSIM_total + ssim

            image = tensor2image(image)
            image.save("./dataset/result/image/({}).jpg".format(i))
            print("=> process {} done! time: {:.3f}s, PSNR: {:.3f}, SSIM: {:.3f}."
                  .format(i, end_time - start_time, psnr, ssim))

        print("=> All the {} images done!, your AVG PSNR: {:.3f}, AVG SSIM: {:.3f}."
              .format(File_No, PSNR_total / File_No, SSIM_total / File_No))
