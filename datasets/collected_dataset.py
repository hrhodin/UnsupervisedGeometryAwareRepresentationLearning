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
                 useSubjectBatches=0, useCamBatches=0,
                 randomize=True,
                 mean=(0.485, 0.456, 0.406),
                 stdDev= (0.229, 0.224, 0.225),
                 useSequentialFrames=0,
                 ):
        args = list(locals().items())
        # save function arguments
        for arg,val in args:
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

        # build cam/subject datastructure
        h5_label_file = h5py.File(data_folder+'/labels.h5', 'r')
        print('Loading h5 label file to memory')
        self.label_dict = {key: np.array(value) for key,value in h5_label_file.items()}
        all_keys_name = data_folder+'/all_keys.pickl'
        sequence_keys_name = data_folder+'/sequence_keys.pickl'
        camsets_name = data_folder+'/camsets.pickl'
        print('Done loading h5 label file')
        if os.path.exists(sequence_keys_name):
            print('Loading sequence-subject-cam association from pickle files {}.'.format(sequence_keys_name))
            self.all_keys = pickle.load( open(all_keys_name, "rb" ) )
            self.sequence_keys = pickle.load( open(sequence_keys_name, "rb" ) )
            self.camsets = pickle.load( open(camsets_name, "rb" ) )
            print('Done loading sequence association.')
        else:
            print('Establishing sequence association. Available labels:',list(h5_label_file.keys()))
            all_keys = set()
            camsets = {}
            sequence_keys = {}
            data_length = len(h5_label_file['frame'])
            with tqdm(total=data_length) as pbar:
                for index in range(data_length):
                    pbar.update(1)
                    sub_i = int(h5_label_file['subj'][index].item())
                    cam_i = int(h5_label_file['cam'][index].item())
                    seq_i = int(h5_label_file['seq'][index].item())
                    frame_i = int(h5_label_file['frame'][index].item())
                    
                    key = (sub_i,seq_i,frame_i)
                    if key not in camsets:
                        camsets[key] = {}
                    camsets[key][cam_i] = index
                    
                    # only add if accumulated enough cameras
                    if len(camsets[key])>=useCamBatches:
                        all_keys.add(key)
        
                        if seq_i not in sequence_keys:
                            sequence_keys[seq_i] = set()
                        sequence_keys[seq_i].add(key)
                
            self.all_keys = list(all_keys)
            self.camsets = camsets
            self.sequence_keys = {seq: list(keyset) for seq,keyset in sequence_keys.items()}
            pickle.dump(self.all_keys, open(all_keys_name, "wb" ) )
            pickle.dump(self.sequence_keys, open(sequence_keys_name, "wb" ) )
            pickle.dump(self.camsets, open(camsets_name, "wb" ) )
            print("DictDataset: Done initializing, listed {} camsets ({} frames) and {} sequences".format(self.__len__(), self.__len__()*useCamBatches, len(sequence_keys)))
                   
    def __len__(self):
        if self.useCamBatches > 0:
            return len(self.all_keys)
        else:
            return len(self.label_dict['frame'])
               
    def getLocalIndices(self, index):
        input_dict = {}
        cam = int(self.label_dict['cam'][index].item())
        seq = int(self.label_dict['seq'][index].item())
        frame = int(self.label_dict['frame'][index].item())
        return cam, seq, frame, index

    def getItemIntern(self, cam, seq, frame, index):
        def getImageName(key):
            return self.data_folder+'/seq_{:03d}/cam_{:02d}/{}_{:06d}.png'.format(seq,cam,key,frame)
        def loadImage(name):
#             if not os.path.exists(name):
#                 print('Image not available ({})'.format(name))
#                 raise Exception('Image not available')
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

    def __getitem__(self, index):
        if self.useSequentialFrames > 1:
            frame_skip = 1  # 6:1 sec, since anyways subsampled at 5 frames
            #cam, seq, frame = getLocalIndices(index)

            # ensure that a sequence is not exceeded
            for i in range(self.useSequentialFrames):
                cam_skip = 4
                index_range = list(range(index+i, index+i + self.useSequentialFrames * frame_skip*cam_skip, frame_skip*cam_skip))
                #print('index_range',index_range,'seq',[int(self.label_dict['seq'][i]) for i in index_range])
                #print('index_range',index_range,'frame',[int(self.label_dict['frame'][i]) for i in index_range])
                #print('index_range',index_range,'cam',[int(self.label_dict['cam'][i]) for i in index_range])
                if len(self.label_dict['seq'])>index_range[-1] and self.label_dict['seq'][index_range[0]] == self.label_dict['seq'][index_range[-1]]:
                    break

            # collect single results
            single_examples = [self.getItemIntern(*self.getLocalIndices(i)) for i in index_range]
            collated_examples = utils_data.default_collate_with_string(single_examples) #accumulate list of single frame results
            return collated_examples
        if self.useCamBatches > 0:
            key = self.all_keys[index]           
            def getCamSubbatch(key):
                camset = self.camsets[key]
                cam_keys = list(camset.keys())
                assert self.useCamBatches <= len(cam_keys)
                if self.randomize:
                    shuffle(cam_keys)
                cam_keys_shuffled = cam_keys[:self.useCamBatches]
                return [self.getItemIntern(*self.getLocalIndices(camset[cami])) for cami in cam_keys_shuffled]
            
            single_examples = getCamSubbatch(key)
            if self.useSubjectBatches > 0:
                #subj = key[0]
                seqi = key[1]
                potential_keys = self.sequence_keys[seqi]
                key_other = potential_keys[np.random.randint(len(potential_keys))]
                single_examples = single_examples + getCamSubbatch(key_other)
            
            collated_examples = utils_data.default_collate_with_string(single_examples) #accumulate list of single frame results
            return collated_examples
        else:
            return self.getItemIntern(*self.getLocalIndices(index))

if __name__ == '__main__':
    dataset = CollectedDataset(
                 data_folder='/cvlabdata1/home/rhodin/code/humanposeannotation/python/pytorch_human_reconstruction/TMP/H36M-MultiView-test',
                 input_types=['img_crop','bg_crop'], label_types=['3D'],
                 useSubjectBatches=2, useCamBatches=4,
                 randomize=True)

    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        IPython.embed()

