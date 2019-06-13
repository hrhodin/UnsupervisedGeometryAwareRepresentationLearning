import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import csv
import numpy as np
import torch
import torchvision
import torch.utils.data as data

import h5py
import imageio

from random import shuffle

import IPython

import numpy.linalg as la

from utils import datasets as utils_data
from tqdm import tqdm
import pickle


class CollectedDataset(data.Dataset):
    def __init__(self, data_folder, 
                 input_types, label_types,
                 mean=(0.485, 0.456, 0.406),
                 stdDev= (0.229, 0.224, 0.225),
                 useSequentialFrames=0,
                 ):
        for arg,val in list(locals().items()):
            setattr(self, arg, val)

        class Image256toTensor(object):
            def __call__(self, pic):
                img = torch.from_numpy(pic.transpose((2, 0, 1))).float()
                img = img.div(255)
                return img

            def __repr__(self):
                return self.__class__.__name__ + '()'

        self.transform_in = torchvision.transforms.Compose([
            Image256toTensor(), #torchvision.transforms.ToTensor() the torchvision one behaved differently for different pytorch versions, hence the custom one..
            torchvision.transforms.Normalize(self.mean, self.stdDev)
        ])

        h5_label_file = h5py.File(data_folder + '/labels.h5', 'r')
        print('Loading h5 label file to memory')
        self.label_dict = {key: np.array(value) for key, value in h5_label_file.items()}

    def __len__(self):
        return len(self.label_dict['frame'])
               
    def getLocalIndices(self, index):
        input_dict = {}
        cam = int(self.label_dict['cam'][index].item())
        seq = int(self.label_dict['seq'][index].item())
        frame = int(self.label_dict['frame'][index].item())
        return cam, seq, frame

    def __getitem__(self, index):
        cam, seq, frame = self.getLocalIndices(index)
        def getImageName(key):
            return self.data_folder + '/seq_{:03d}/cam_{:02d}/{}_{:06d}.png'.format(seq, cam, key, frame)
        def loadImage(name):
            #             if not os.path.exists(name):
            #                 raise Exception('Image not available ({})'.format(name))
            return np.array(self.transform_in(imageio.imread(name)), dtype='float32')
        def loadData(types):
            new_dict = {}
            for key in types:
                if key in ['img_crop','bg_crop']:
                    new_dict[key] = loadImage(getImageName(key)) #np.array(self.transform_in(imageio.imread(getImageName(key))), dtype='float32')
                else:
                    new_dict[key] = np.array(self.label_dict[key][index], dtype='float32')
            return new_dict
        return loadData(self.input_types), loadData(self.label_types)

class CollectedDatasetSampler(data.sampler.Sampler):
    def __init__(self, data_folder, batch_size,
                 actor_subset=None,
                 useSubjectBatches=0, useCamBatches=0,
                 randomize=True,
                 useSequentialFrames=0,
                 every_nth_frame=1):
        # save function arguments
        for arg,val in list(locals().items()):
            setattr(self, arg, val)

        # build cam/subject datastructure
        h5_label_file = h5py.File(data_folder + '/labels.h5', 'r')
        print('Loading h5 label file to memory')
        label_dict = {key: np.array(value) for key, value in h5_label_file.items()}
        self.label_dict = label_dict
        print('Establishing sequence association. Available labels:', list(label_dict.keys()))
        all_keys = set()
        camsets = {}
        sequence_keys = {}
        data_length = len(label_dict['frame'])
        with tqdm(total=data_length) as pbar:
            for index in range(data_length):
                pbar.update(1)
                sub_i = int(label_dict['subj'][index].item())
                cam_i = int(label_dict['cam'][index].item())
                seq_i = int(label_dict['seq'][index].item())
                frame_i = int(label_dict['frame'][index].item())

                if actor_subset is not None and sub_i not in actor_subset:
                    continue

                key = (sub_i, seq_i, frame_i)
                if key not in camsets:
                    camsets[key] = {}
                camsets[key][cam_i] = index

                # only add if accumulated enough cameras
                if len(camsets[key]) >= self.useCamBatches:
                    all_keys.add(key)

                    if seq_i not in sequence_keys:
                        sequence_keys[seq_i] = set()
                    sequence_keys[seq_i].add(key)

        self.all_keys = list(all_keys)
        self.camsets = camsets
        self.sequence_keys = {seq: list(keyset) for seq, keyset in sequence_keys.items()}
        print("DictDataset: Done initializing, listed {} camsets ({} frames) and {} sequences".format(
                                            len(self.camsets), len(self.all_keys), len(sequence_keys)))

    def __iter__(self):
        index_list = []
        print("Randomizing dataset (CollectedDatasetSampler.__iter__)")
        with tqdm(total=len(self.all_keys)//self.every_nth_frame) as pbar:
            for index in range(0,len(self.all_keys), self.every_nth_frame):
                pbar.update(1)
                key = self.all_keys[index]
                def getCamSubbatch(key):
                    camset = self.camsets[key]
                    cam_keys = list(camset.keys())
                    assert self.useCamBatches <= len(cam_keys)
                    if self.randomize:
                        shuffle(cam_keys)
                    if self.useCamBatches == 0:
                        cam_subset_size = 99
                    else:
                        cam_subset_size = self.useCamBatches
                    cam_indices = [camset[k] for k in cam_keys[:cam_subset_size]]
                    return cam_indices

                index_list = index_list + getCamSubbatch(key)
                if self.useSubjectBatches:
                    seqi = key[1]
                    potential_keys = self.sequence_keys[seqi]
                    key_other = potential_keys[np.random.randint(len(potential_keys))]
                    index_list = index_list + getCamSubbatch(key_other)

        subject_batch_factor = 1+int(self.useSubjectBatches > 0) # either 1 or 2
        cam_batch_factor = max(1,self.useCamBatches)
        sub_batch_size = cam_batch_factor*subject_batch_factor
        assert len(index_list) % sub_batch_size == 0
        indices_batched = np.array(index_list).reshape([-1,sub_batch_size])
        if self.randomize:
            indices_batched = np.random.permutation(indices_batched)
        indices_batched = indices_batched.reshape([-1])[:(indices_batched.size//self.batch_size)*self.batch_size] # drop last frames
        return iter(indices_batched.reshape([-1,self.batch_size]))


if __name__ == '__main__':
    dataset = CollectedDataset(
                 data_folder='/Users/rhodin/H36M-MultiView-test',
                 input_types=['img_crop','bg_crop'], label_types=['3D'])

    batch_sampler = CollectedDatasetSampler(
                 data_folder='/Users/rhodin/H36M-MultiView-test',
                 useSubjectBatches=1, useCamBatches=2,
                 batch_size=8,
                 randomize=True)

    trainloader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler,
                                              num_workers=0, pin_memory=False,
                                              collate_fn=utils_data.default_collate_with_string)

    # iterate over batches
    for input, labels in iter(trainloader):
        IPython.embed()

