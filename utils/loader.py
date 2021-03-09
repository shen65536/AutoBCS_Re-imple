import torchvision
import torch.utils.data as data


def loader(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Grayscale(num_output_channels=1)
    ])
    dst = torchvision.datasets.ImageFolder(args.train_path, transform=transforms)
    print("Dataset length: {}".format(len(dst)))
    dataset = data.DataLoader(dst, batch_size=args.batch_size, shuffle=False, drop_last=True)
    return dataset
