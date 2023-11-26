import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

import cv2
from onehot import onehot
import math

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class BagTrainDataset(Dataset):

    def __init__(self, transform=None):
        self.transform = transform
        
    def __len__(self):
        return len(os.listdir('../datasets/All/images'))

    def __getitem__(self, idx):
        img_name = os.listdir('../datasets/All/images')[idx]
        imgA = cv2.imread('../datasets/All/images/'+img_name)
        imgA = cv2.resize(imgA, (512, 512))
        imgB = cv2.imread('../datasets/All/labels/'+img_name, 0)
        imgB = cv2.resize(imgB, (512, 512), interpolation = cv2.INTER_NEAREST)
        imgB = imgB/255
        imgB = imgB.astype('uint8')
        imgB = imgB.reshape(1, 512, 512)
        imgB = torch.FloatTensor(imgB)
        if self.transform:
            imgA = self.transform(imgA)

        return imgA, imgB
        
class BagTestDataset(Dataset):

    def __init__(self, transform=None):
        self.transform = transform
        
    def __len__(self):
        return len(os.listdir('../datasets/semi_test/images'))

    def __getitem__(self, idx):
        img_name = os.listdir('../datasets/semi_test/images')[idx]
        imgA = cv2.imread('../datasets/semi_test/images/'+img_name)
        imgA = cv2.resize(imgA, (512, 512))
        imgB = cv2.imread('../datasets/semi_test/labels/'+img_name, 0)
        imgB = cv2.resize(imgB, (512, 512), interpolation = cv2.INTER_NEAREST)
        imgB = imgB/255
        imgB = imgB.astype('uint8')
        imgB = imgB.reshape(1, 512, 512)
        imgB = torch.FloatTensor(imgB)
        if self.transform:
            imgA = self.transform(imgA)

        return imgA, imgB

bag1 = BagTrainDataset(transform)
bag2 = BagTestDataset(transform)

test_size = 15
temp_size = len(bag2) - 15

test_dataset, temp = random_split(bag2, [test_size, temp_size],
                                                               generator=torch.Generator().manual_seed(3568))

train_dataloader = DataLoader(bag1, batch_size=8, shuffle=True, num_workers=16)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16)


# if __name__ =='__main__':

    # for train_batch in train_dataloader:
    #     print(train_batch)
    #
    # for test_batch in test_dataloader:
    #     print(test_batch)
