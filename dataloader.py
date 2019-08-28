from enum import Enum
import torch.utils.data as utils
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class Cifar10Sampler(utils.Sampler):
    """ A sample which just return CIFAR10 images of a certain type"""
    def __init__(self, data_source, categories):
        self.categories = categories
        self.data_source = data_source
  
    def __iter__(self):
        indices = [idx for idx, data in enumerate(self.data_source) if data[1]==1]
        return iter(indices)

class Datasets(Enum):
    """ Enumerate available datasets """
    MNIST = 1
    CIFAR10 = 2

def get_dataloader(dataset: Datasets, batch_size: int) -> (utils.DataLoader, utils.DataLoader):
    """ Loads a dataset from the Datasets enumeration """
    if dataset==Datasets.CIFAR10:
        img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
            ])
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=img_transforms)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=img_transforms)
        trainsampler = Cifar10Sampler(trainset, 1)
        testsampler = Cifar10Sampler(testset, 1)
        trainloader = utils.DataLoader(trainset, sampler=trainsampler, shuffle=False, batch_size=batch_size)
        testloader = utils.DataLoader(testset, sampler=testsampler, shuffle=False, batch_size=batch_size)
    elif dataset==Datasets.MNIST:
        img_transforms = transforms.Compose([
            transforms.Pad(2), # 28x28 --> 32x32. Using 32x32, we don't need to pad when convolving/maxpooling
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,))
            ])
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=img_transforms)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=img_transforms)
        trainloader = utils.DataLoader(trainset, shuffle=False, batch_size=batch_size)
        testloader = utils.DataLoader(testset, shuffle=False, batch_size=batch_size)
    return trainloader, testloader