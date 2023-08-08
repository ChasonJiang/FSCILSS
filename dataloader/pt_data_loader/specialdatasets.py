# MIT License
#
# Copyright (c) 2020 Marvin Klingner
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import, division, print_function

import sys
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataloader.pt_data_loader.basedataset import BaseDataset
import dataloader.pt_data_loader.mytransforms as mytransforms
import dataloader.definitions.labels_file as lf
from torch.utils.data import Dataset
from torchvision import transforms
import os

import json
import cv2
import numpy as np

import dataloader.pt_data_loader.mytransforms as mytransforms
import dataloader.pt_data_loader.dataset_parameterset as dps
import dataloader.file_io.get_path as gp
import dataloader.file_io.dir_lister as dl


class StandardDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super(StandardDataset, self).__init__(*args, **kwargs)

        if self.disable_const_items is False:
            assert self.parameters.K is not None and self.parameters.stereo_T is not None, '''There are no K matrix and
            stereo_T parameter available for this dataset.'''

    def add_const_dataset_items(self, sample):
        K = self.parameters.K.copy()

        native_key = ('color', 0, -1) if (('color', 0, -1) in sample) else ('color_right', 0, -1)
        native_im_shape = sample[native_key].shape

        K[0, :] *= native_im_shape[1]
        K[1, :] *= native_im_shape[0]

        sample["K", -1] = K
        sample["stereo_T"] = self.parameters.stereo_T

        return sample


class KITTIDataset(StandardDataset):
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)


class MapillaryDataset(StandardDataset):
    def __init__(self, *args, **kwargs):
        super(MapillaryDataset, self).__init__(*args, **kwargs)


class CityscapesDataset(StandardDataset):
    def __init__(self, *args, **kwargs):
        super(CityscapesDataset, self).__init__(*args, **kwargs)


class Gta5Dataset(StandardDataset):
    def __init__(self, *args, **kwargs):
        super(Gta5Dataset, self).__init__(*args, **kwargs)


class SimpleDataset(BaseDataset):
    '''
        Dataset that uses the Simple Mode. keys_to_load must be specified.
    '''
    def __init__(self, *args, **kwargs):
        super(SimpleDataset, self).__init__(*args, **kwargs)

    def add_const_dataset_items(self, sample):
        return sample


class cityscapes_simple_Dataset():
    # def __init__(self, *args, **kwargs):
    #     super(StandardDataset, self).__init__(*args, **kwargs)
    #     self.folders_to_load=folders_to_load
    def __init__(self,
                 data_dict,
                 labels=None,
                 data_transforms=None,
                 loading_pseudo_label=False,
                 dataset="cityscapes"
                 ):


        keys_to_load=['color', 'segmentation']
        self.loading_pseudo_label = loading_pseudo_label
        path_getter = gp.GetPath()  
        dataset_folder = path_getter.get_data_path()
        self.datasetpath = os.path.join(dataset_folder, dataset)
        assert labels is not None
        self.labels = labels
        self.labels_mode = "fromid"


        self.data = self.read_from_dict(data_dict, self.loading_pseudo_label)

        self.data_transforms = list(data_transforms)

        # Error if CreateColorAug and CreateScaledImage not in transforms.
        if mytransforms.CreateScaledImage not in data_transforms:
            raise Exception('The transform CreateScaledImage() has to be part of the data_transforms list')
        if mytransforms.CreateColoraug not in data_transforms:
            raise Exception('The transform CreateColoraug() has to be part of the data_transforms list')

        # Error if depth, segmentation or flow keys are given but not the corresponding Convert-Transform
        if any([key.startswith('segmentation') for key in keys_to_load]) and \
                mytransforms.ConvertSegmentation not in self.data_transforms:
            raise Exception('When loading segmentation images, please add mytransforms.ConvertSegmentation() to '
                            'the data_transforms')


        # Set the correct parameters to the ConvertDepth and ConvertSegmentation transforms
        for i, transform in zip(range(len(self.data_transforms)), self.data_transforms):
            if isinstance(transform, mytransforms.ConvertSegmentation):
                transform.set_mode(self.labels, self.labels_mode)

        self.data_transforms = transforms.Compose(self.data_transforms)

        
        self.load_transforms = transforms.Compose(
            [mytransforms.LoadRGB(),
             mytransforms.LoadSegmentation(),
             ])

    def read_from_dict(self, data_dict,loading_pseudo_label):
        data_files = {}
        frame_index = 0
        resolution = -1

        # gt_list = []
        # img_list = []
        # for key in data_dict['gt'].keys():
        #     gt_list.append(data_dict['gt'][key])
        #     img_list.append(data_dict['img'][key])
        data_files.update({('color', frame_index, resolution): data_dict['img']})
        data_files.update({('segmentation', frame_index, resolution): data_dict['gt']})

        return data_files

    def __len__(self):
        """Return the number of elements inside the dataset"""
        dict_keys = list(self.data.keys())
        return len(self.data[dict_keys[0]])

    def __getitem__(self, number):
        """Dataset element with index number 'number' is loaded"""
        sample = {}
        for item in list(self.data.keys()):
            if isinstance(self.data[item][number], str):
                element = self.read_image_file(self.data[item][number])
                # print(self.data[item][number])
            else:
                element = self.data[item][number]
            sample.update({item: element})
        sample = self.load_transforms(sample)
        sample = self.data_transforms(sample)
        return sample

    def read_image_file(self, filepath):
        """Returns an image as a numpy array"""
        filepath = os.path.join(self.datasetpath, filepath)
        filepath = filepath.replace('/', os.sep)
        filepath = filepath.replace('\\', os.sep)
        image = cv2.imread(filepath, -1)
        return image


# if __name__ == '__main__':
#     """
#     The following code is an example of how a dataloader object can be created for a specific dataset. In this case,
#     the cityscapes dataset is used. 
    
#     Every dataset should be created using the StandardDataset class. Necessary arguments for its constructor are the
#     dataset name and the information whether to load the train, validation or test split. In this standard setting,
#     for every dataset only the color images are loaded. The user can pass a list of keys_to_load in order to also use
#     other data categories, depending on what is available in the dataset. It is also possible to define a list of
#     transforms that are performed every time an image is loaded from the dataset.
#     """
#     def print_dataset(dataloader, num_elements=3):
#         """
#         This little function prints the size of every element in a certain amount of dataloader samples.

#         :param dataloader: dataloader object that yields the samples
#         :param num_elements: number of samples of which the sizes are to be printed
#         """
#         for element, i in zip(dataloader, range(num_elements)):
#             print('+++ Image {} +++'.format(i))
#             for key in element.keys():
#                 print(key, element[key].shape)
#             plt.imshow(np.array(element[('color', 0, 0)])[0, :, :, :].transpose(1, 2, 0))

#     # Simple example of how to load a dataset. Every supported dataset can be loaded that way.
#     dataset = 'cityscapes'
#     trainvaltest_split = 'train'
#     keys_to_load = ['color', 'depth', 'segmentation', 'camera_intrinsics']   # Optional; standard is just 'color'

#     # The following parametes and the data_transforms list are optional. Standard is just the transform ToTensor()
#     width = 640
#     height = 192
#     scales = [0, 1, 2, 3]
#     data_transforms = [#mytransforms.RandomExchangeStereo(),  # (color, 0, -1)
#                        mytransforms.RandomHorizontalFlip(),
#                        mytransforms.RandomVerticalFlip(),
#                        mytransforms.CreateScaledImage(),  # (color, 0, 0)
#                        mytransforms.RandomRotate(0.0),
#                        mytransforms.RandomTranslate(0),
#                        mytransforms.RandomRescale(scale=1.1, fraction=0.5),
#                        mytransforms.RandomCrop((320, 1088)),
#                        mytransforms.Resize((height, width)),
#                        mytransforms.MultiResize(scales),
#                        mytransforms.CreateColoraug(new_element=True, scales=scales),  # (color_aug, 0, 0)
#                        mytransforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
#                                                 hue=0.1, gamma=0.0),
#                        mytransforms.GaussianBlurr(fraction=0.5),
#                        mytransforms.RemoveOriginals(),
#                        mytransforms.ToTensor(),
#                        mytransforms.NormalizeZeroMean(),
#                        ]

#     print('Loading {} dataset, {} split'.format(dataset, trainvaltest_split))
#     traindataset = StandardDataset(dataset,
#                                    trainvaltest_split,
#                                    keys_to_load=keys_to_load,
#                                    stereo_mode='mono',
#                                    keys_to_stereo=['color', 'depth', 'segmentation'],
#                                    data_transforms=data_transforms
#                                    )
#     trainloader = DataLoader(traindataset, batch_size=1,
#                              shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
#     print(traindataset.stereo_mode)
#     print_dataset(trainloader)