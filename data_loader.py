import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
import glob


class AF(data.Dataset):
    def __init__(self, data_folder, ref_folder, type = '*.png', transform=None):
        self.file_list = glob.glob(os.path.join(data_folder,'*', type))
        self.ref_folder = ref_folder
        self.transform = transform

    def __getitem__(self, index):
        file_name = self.file_list[index]
        len_index = file_name.split('/')[-1]
        label_file_name = os.path.join(self.ref_folder, len_index)
        trans_back = transforms.ToTensor()
        image = Image.open(file_name).convert('L')
        label_image = Image.open(file_name).convert('L')
        label_image = trans_back(label_image)
        if self.transform is not None:
            image =  self.transform(image)
        else:
            image = trans_back(image)
        return image, label_image

    def __len__(self):
        return (len(self.file_list))


def get_loader(data_folder, ref_folder, transform, batch_size, shuffle, num_workers, type = '*.png'):
    af = AF(data_folder = data_folder, ref_folder = ref_folder, transform = transform, type = type)

    data_loader = torch.utils.data.DataLoader(dataset=af,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              )
    return data_loader
