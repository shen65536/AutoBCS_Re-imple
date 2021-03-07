import time
import torch
import torchvision
import numpy as np
import torch.nn as nn
import scipy.io as scio

import models
import options

if __name__ == '__main__':
    args = options.args_set()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        init_net = models.init_net(args)
        init_net = nn.DataParallel(init_net)
        init_net.load_state_dict(torch.load("./trained_models/init_net_ratio{}.pth".format(args.ratio),
                                            map_location='cpu'))
        init_net.to(device)
        init_net.eval()

        deep_net = models.oct_net(args)
        deep_net = nn.DataParallel(deep_net)
        deep_net.load_state_dict(torch.load("./trained_models/init_net_ratio{}.pth".format(args.ratio),
                                            map_location='cpu'))
        deep_net.to(device)
        deep_net.eval()

        File_No = 100
        Folder_name = "{}/BSD100".format(args.test_path)

        for i in range(1, File_No + 1):
            name = "{}/({}).mat".format(Folder_name, i)
            data = scio.loadmat(name)
            image = data['temp3']
            image = np.array(image)
            image = torch.from_numpy(image)

            image = torch.unsqueeze(image, 0)
            image = torch.unsqueeze(image, 0)
            image = image.float()

            image = image.to(device)
            init_res = init_net(image)
            start_time = time.time()
            deep_res = deep_net(init_res)
            end_time = time.time()

            init_res = torch.squeeze(init_res, 0)
            init_res = torch.squeeze(init_res, 0)
            init_res = init_res.to('cpu')
            init_res = init_res.numpy()
            path = "{}/result/mat/({})_init.mat".format(Folder_name, i)
            scio.savemat(path, {'IR': init_res})

            tensor2image = torchvision.transforms.ToPILImage()

            tensor1 = torch.from_numpy(np.array(init_res))
            image1 = tensor2image(tensor1)
            image1.save("{}/result/image/({})_init.jpg".format(Folder_name, i))

            deep_res = torch.squeeze(deep_res, 0)
            deep_res = torch.squeeze(deep_res, 0)
            deep_res = deep_res.to('cpu')
            deep_res = deep_res.numpy()
            path = "{}/result/mat/({})_deep.mat".format(Folder_name, i)
            scio.savemat(path, {'FR': deep_res})

            tensor2 = torch.from_numpy(np.array(deep_res))
            image2 = tensor2image(tensor2)
            image2.save("{}/result/image/({})_deep.jpg".format(Folder_name, i))
