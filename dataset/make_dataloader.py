import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

import math
import random

from .Dataset import Dataset
from .bases import ImageDataset
#from sampler.triplet_sampler import RandomIdentitySampler

from PIL import Image
import numpy as np

class GaussianMask(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img
        # print(img.size)
        width = img.size[0]
        height = img.size[1]
        mask = np.zeros((height, width))
        mask_h = np.zeros((height, width))
        mask_h += np.arange(0, width) - width / 2
        mask_v = np.zeros((width, height))
        mask_v += np.arange(0, height) - height / 2
        mask_v = mask_v.T

        numerator = np.power(mask_h, 2) + np.power(mask_v, 2)
        denominator = 2 * (height * height + width * width)
        mask = np.exp(-(numerator / denominator))

        img = np.asarray(img)
        new_img = np.zeros_like(img)
        new_img[:, :, 0] = np.multiply(mask, img[:, :, 0])
        new_img[:, :, 1] = np.multiply(mask, img[:, :, 1])
        new_img[:, :, 2] = np.multiply(mask, img[:, :, 2])

        return Image.fromarray(new_img)


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

def train_collate_fn(batch):
    imgs, pids, _, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids
#collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果

def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids


def make_dataloader(Cfg):
    train_transforms = T.Compose([
        T.Resize(Cfg.MODEL.INPUT_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(8),
        T.RandomCrop(Cfg.MODEL.INPUT_SIZE),
        T.RandomRotation(10, resample=Image.BICUBIC, expand=False, center=None),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        RandomErasing(probability=0.5, sh=0.1, mean=(0.4914, 0.4822, 0.4465))
    ])

    val_transforms = T.Compose([
        T.Resize(Cfg.MODEL.INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    num_workers = Cfg.DATALOADER.DATALOADER_NUM_WORKERS
    dataset = Dataset(data_dir = Cfg.DATALOADER.DATA_DIR, verbose = True)
    num_classes = dataset.num_train_pids

    train_set = ImageDataset(dataset.train, train_transforms)

    if Cfg.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(train_set,
            batch_size = Cfg.SOLVER.BATCHSIZE,
            shuffle = True,
            num_workers = num_workers,
            sampler = None,
            collate_fn = train_collate_fn, #customized batch sampler
            drop_last = True
        )
    else:
        print('unsupported sampler! expected softmax but got {}'.format(Cfg.DATALOADER.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(val_set,
        batch_size=Cfg.TEST.IMS_PER_BATCH,
        shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, val_loader, len(dataset.query), num_classes