import torch
import torch.nn as nn

import model
import options

if __name__ == '__main__':
    args = options.args_set()

    with torch.no_grad():
        IniReconNet = model.init_net(args)  # 比例因子：4
        IniReconNet = nn.DataParallel(IniReconNet)
        IniReconNet.load_state_dict(
            torch.load('../Training/IniReconNet_100EPO_64BATCH_ScalingFactor_4.pth', map_location='cpu'))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        IniReconNet.to(device)
        IniReconNet.eval()

        DeepOctNet = OctNet(2)
        DeepOctNet = nn.DataParallel(DeepOctNet)
        DeepOctNet.load_state_dict(torch.load('../Training/DeepOctNet_100EPO_64BATCH_ScalingFactor_4.pth', map_location='cpu'))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        DeepOctNet.to(device)
        DeepOctNet.eval()

        File_No = 100
        Folder_name = '/Users/shen/PycharmProjects/BSD100'

        for i in range(1, File_No + 1):
            # name = ('../%s/(%d).mat' % (Folder_name, i))
            name = ('%s/(%d).mat' % (Folder_name, i))
            data = scio.loadmat(name)
            image = data['temp3']
            image = np.array(image)
            image = torch.from_numpy(image)

            image = torch.unsqueeze(image, 0)
            image = torch.unsqueeze(image, 0)
            image = image.float()

            # Evaluation
            image = image.to(device)
            pred_IR = IniReconNet(image)
            start_time = time.time()
            pred_FR = DeepOctNet(pred_IR)
            end_time = time.time()

            pred_IR = torch.squeeze(pred_IR, 0)
            pred_IR = torch.squeeze(pred_IR, 0)
            pred_IR = pred_IR.to('cpu')
            pred_IR = pred_IR.numpy()
            path = ('%s/(%d)_initRecon.mat' % (Folder_name, i))
            scio.savemat(path, {'PRED_IR': pred_IR})

            tensor2image = torchvision.transforms.ToPILImage()

            tensor1 = torch.from_numpy(np.array(pred_IR))
            image1 = tensor2image(tensor1)
            image1.save('%s/InitRecon(%d).jpg' % (Folder_name, i))

            pred_FR = torch.squeeze(pred_FR, 0)
            pred_FR = torch.squeeze(pred_FR, 0)
            pred_FR = pred_FR.to('cpu')
            pred_FR = pred_FR.numpy()
            path = ('%s/(%d)_finalRecon.mat' % (Folder_name, i))
            scio.savemat(path, {'PRED_FR': pred_FR})

            tensor2 = torch.from_numpy(np.array(pred_FR))
            image2 = tensor2image(tensor2)
            image2.save('%s/FinalRecon(%d).jpg' % (Folder_name, i))