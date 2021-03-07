import torchvision
import torch.utils.data as data


class loader:
    def __init__(self, args):
        self.block_size = args.block_size
        self.train_path = args.train_path
        self.batch_size = args.batch_size

    def load(self):
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(self.block_size),
            torchvision.transforms.Grayscale(num_output_channels=1)
        ])

        dataset = torchvision.datasets.ImageFolder(self.train_path, transform=transforms)
        dataset = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataset
