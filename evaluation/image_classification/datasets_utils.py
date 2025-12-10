import csv
import os
import config # Assuming config is in sys.path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# A lot of the approaches here are inspired from the wonderful paper from O'Connor and Andreas 2021.
# https://github.com/lingo-mit/context-ablations
#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------


# noinspection PyUnusedLocal
class CheXpertDataSet(Dataset):
    # noinspection PyUnusedLocal
    def __init__(self, data_PATH, transform=None, policy="ones"):
        """
        data_PATH: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        Upolicy: name the policy with regard to the uncertain labels.
        """
        image_names = []
        labels = []
        self.dict = [
            {"1.0": "1", "": "0", "0.0": "0", "-1.0": "0"},
            {"1.0": "1", "": "0", "0.0": "0", "-1.0": "1"},
        ]
        self.strange_list = [
            "CheXpert-v1.0-small/train/patient00773/study2/view1_frontal.jpg",
            "CheXpert-v1.0-small/train/patient00770/study1/view1_frontal.jpg",
            "CheXpert-v1.0-small/train/patient34662/study18/view1_frontal.jpg",
        ]

        with open(data_PATH, "r") as f:
            header = f.readline().strip("\n").split(",")
            self._label_header = [
                header[7],
                header[10],
                header[11],
                header[13],
                header[15],
            ]
            # print(self._label_header)
            csvReader = csv.reader(f)
            # next(csvReader, None) # skip the header
            for line in csvReader:
                label_list = []
                image_name = line[0]

                label = line[5:]

                # for i in range(2):
                #   if label[i]:
                #       #print(label[i])
                #       a = float(label[i])
                #       if a == 1:
                #           label[i] = 1
                #       else:
                #           label[i] = 0
                #   else:
                #       label[i] = 0
                for index, value in enumerate(label):
                    if index == 5 or index == 8:
                        label_list.append(self.dict[1].get(value))
                    elif index == 2 or index == 6 or index == 10:
                        label_list.append(self.dict[0].get(value))
                label_list = list(map(int, label_list))

                if image_name in self.strange_list:
                    pass
                else:
                    image_names.append(image_name)
                    labels.append(label_list)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        image_name = self.image_names[index]
        image_name = os.path.join(config.IMAGES_ROOT, image_name)
        image = Image.open(image_name).convert("RGB")
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label), image_name

    def __len__(self):
        return len(self.image_names)


class ExpandChannels:
    """
    Transforms an image with one channel to an image with three channels by copying
    pixel intensities of the image along the 1st dimension.
    """

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        :param data: Tensor of shape [1, H, W].
        :return: Tensor with channel copied three times, shape [3, H, W].
        """
        if data.shape[0] != 1:
            raise ValueError(f"Expected input of shape [1, H, W], found {data.shape}")
        return torch.repeat_interleave(data, 3, dim=0)
