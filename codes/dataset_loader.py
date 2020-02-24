# author: Mohammad Minhazul Haq
# created on: February 23, 2020

import numpy as np
from torch.utils import data
from skimage.io import imread
import os
import random

class WSI_Dataset(data.Dataset):
    def __init__(self, dir, id_list_file, batch_size, transform=None, mean=0.0, std=1.0):
        self.wsi_folders = []
        wsi_ids = np.load(id_list_file)

        for wsi_id in wsi_ids:
            wsi_name = wsi_id.split('_')[0]
            wsi_label = wsi_id.split('_')[-1]

            wsi_path = os.path.join(dir, wsi_name)

            if os.access(wsi_path, os.R_OK):
                wsi_patch_filenames = os.listdir(wsi_path)

                if len(wsi_patch_filenames) >= batch_size:
                    self.wsi_folders.append({"path": wsi_path,
                                             "label": int(wsi_label) - 1,
                                             "name": wsi_name})
                else:
                    print("WSI with less than " + str(batch_size) + " patches: " + wsi_name + " - " + str(len(wsi_patch_filenames)) + " patches")
            else:
                print("Permission denied for WSI: " + wsi_name)

        self.batch_size = batch_size  # M
        self.transform = transform
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.wsi_folders)

    def __getitem__(self, index):
        wsi = self.wsi_folders[index]
        wsi_path = wsi["path"]
        print(str(index) + ", " + wsi_path + ", " + str(wsi["label"]) + ", " + wsi["name"])

        wsi_patch_filenames = os.listdir(wsi_path)
        random.shuffle(wsi_patch_filenames)

        wsi_patches = np.zeros((self.batch_size, 3, 224, 224))

        for i in range(self.batch_size):
            wsi_patch_filename = wsi_patch_filenames[i]

            try:
                wsi_patch = imread(os.path.join(wsi_path, wsi_patch_filename))

                if self.transform:
                    wsi_patch = self.transform(wsi_patch)

                wsi_patches[i] = wsi_patch
            except (ValueError):
                print("Bad file with ValueError")
                i -= 1

        return wsi_patches, wsi["label"], wsi["name"]
