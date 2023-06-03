from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.datasets.utils import check_integrity
from typing import *
# from zipdata import ZipData

import bisect 
import numpy as np
import os
import pickle
import torch

# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
os.environ['IMAGENET_LOC_ENV'] = "/ssdscratch/hxue45/data/datasets/imagenet/" # imagenet
os.environ['PT_DATA_DIR'] = "/ssdscratch/hxue45/data/datasets/" # cifar-10


# list of all datasets
DATASETS = ["imagenet", "imagenet32", "cifar10", "cifar10_vit"]


def get_dataset(dataset: str, split: str) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "imagenet":
        return _imagenet(split)
    
    elif dataset == "imagenet32":
        return _imagenet32(split)

    elif dataset == "cifar10":
        return _cifar10(split)

    elif dataset == 'cifar10_vit':
        return _cifar10vit(split)


def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if 'imagenet' in dataset:
        return 1000
    elif "cifar10" in dataset:
        return 10


def get_normalize_layer(dataset: str, diff=None, vit=None) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if diff:
        return NormalizeLayer(_DIFF_MEAN, _DIFF_STD)
    if vit:
        # FOR VIT AND BEIT-L
        return NormalizeLayer(_CIFAR10_MEAN_VIT, _CIFAR10_STDDEV_VIT)

    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    elif dataset == 'cifar10_vit':
        return NormalizeLayer(_CIFAR10_MEAN_VIT, _CIFAR10_STDDEV_VIT)
        print("vit norm")
    elif dataset == "imagenet32":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    
 


def get_input_center_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's Input Centering layer"""
    if dataset == "imagenet":
        return InputCenterLayer(_IMAGENET_MEAN)
    elif dataset == "cifar10":
        return InputCenterLayer(_CIFAR10_MEAN)


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]



_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

_CIFAR10_MEAN_VIT = [0.5, 0.5, 0.5]
_CIFAR10_STDDEV_VIT = [0.5, 0.5, 0.5]

_DIFF_MEAN = [0, 0, 0]
_DIFF_STD = [1, 1, 1]





def _cifar10(split: str) -> Dataset:
    dataset_path = os.path.join(os.getenv('PT_DATA_DIR', 'datasets'))
    if split == "train":
        return datasets.CIFAR10(dataset_path, train=True, download=False, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "test":
        # print(dataset_path)
        return datasets.CIFAR10(dataset_path, train=False, download=False, transform=transforms.ToTensor())

    else:
        raise Exception("Unknown split name.")

def _cifar10vit(split: str) -> Dataset:
    dataset_path = os.path.join(os.getenv('PT_DATA_DIR', 'datasets'))
    if split == "train":
        return datasets.CIFAR10(dataset_path, train=True, download=False, transform=transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "test":
        # print(dataset_path)
        return datasets.CIFAR10(dataset_path, train=False, download=False, transform=transforms.Compose([
            # transforms.Resize(224),
            transforms.ToTensor()
            ]))

    else:
        raise Exception("Unknown split name.")


def _imagenet(split: str) -> Dataset:
    # print(os.environ['IMAGENET_LOC_ENV'])
    # print(os.environ)
    # if not IMAGENET_LOC_ENV in os.environ.keys():
    #     raise RuntimeError("environment variable for ImageNet directory not set")

    dir = os.environ['IMAGENET_LOC_ENV']
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            transforms.Resize(256),
            # transforms.Resize((384, 384)),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    return datasets.ImageFolder(subdir, transform)




def _imagenet32(split: str) -> Dataset:
    dataset_path = os.path.join(os.getenv('PT_DATA_DIR', 'datasets'), 'Imagenet32')
   
    if split == "train":
        return ImageNetDS(dataset_path, 32, train=True, transform=transforms.Compose([
                        transforms.RandomCrop(32, padding=4), 
                        transforms.RandomHorizontalFlip(), 
                        transforms.ToTensor()]))
    
    elif split == "test":
        return ImageNetDS(dataset_path, 32, train=False, transform=transforms.ToTensor())


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.register_buffer('means', torch.tensor(means))
        self.register_buffer('sds', torch.tensor(sds))

    def forward(self, input: torch.tensor, y=None):
        # print("norm layer input", input.max(), input.min())
        # print(self.means)
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).to(input.device)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).to(input.device)
        return (input - means)/sds



# from https://github.com/hendrycks/pre-training
class ImageNetDS(Dataset):
    """`Downsampled ImageNet <https://patrykchrabaszcz.github.io/Imagenet32/>`_ Datasets.

    Args:
        root (string): Root directory of dataset where directory
            ``ImagenetXX_train`` exists.
        img_size (int): Dimensions of the images: 64,32,16,8
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    """
    base_folder = 'Imagenet{}_train'
    train_list = [
        ['train_data_batch_1', ''],
        ['train_data_batch_2', ''],
        ['train_data_batch_3', ''],
        ['train_data_batch_4', ''],
        ['train_data_batch_5', ''],
        ['train_data_batch_6', ''],
        ['train_data_batch_7', ''],
        ['train_data_batch_8', ''],
        ['train_data_batch_9', ''],
        ['train_data_batch_10', '']
    ]

    test_list = [
        ['val_data', ''],
    ]

    def __init__(self, root, img_size, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.img_size = img_size

        self.base_folder = self.base_folder.format(img_size)

        # if not self._check_integrity():
        #    raise RuntimeError('Dataset not found or corrupted.') # TODO

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                with open(file, 'rb') as fo:
                    entry = pickle.load(fo)
                    self.train_data.append(entry['data'])
                    self.train_labels += [label - 1 for label in entry['labels']]
                    self.mean = entry['mean']

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((self.train_data.shape[0], 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, f)
            fo = open(file, 'rb')
            entry = pickle.load(fo)
            self.test_data = entry['data']
            self.test_labels = [label - 1 for label in entry['labels']]
            fo.close()
            self.test_data = self.test_data.reshape((self.test_data.shape[0], 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True



if __name__ == "__main__":
    dataset = get_dataset('imagenet32', 'train')
    embed()