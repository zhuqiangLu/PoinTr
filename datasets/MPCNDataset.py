import torch.utils.data as data
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io import IO
import random
import os
import json
from .build import DATASETS
from einops import rearrange
from utils.logger import *

import torch

# References:
# - https://github.com/hzxie/GRNet/blob/master/utils/data_loaders.py

@DATASETS.register_module()
class MPCN(data.Dataset):
    # def __init__(self, data_root, subset, class_choice = None):
    def __init__(self, config):
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset
        self.cars = config.CARS
        # self.complete_only = config.COMPLETE_ONLY
        self.split = config.SPLIT
        # self.keep_gt = config.KEEP_GT
        self.normalize = config.NORMALIZE if hasattr(config, 'NORMALIZE') else False
        self.noise_scale = config.NOISE_SCALE if hasattr(config, 'NOISE_SCALE') else 0
        self.shift = config.SHIFT if hasattr(config, 'SHIFT') else 0

        self.atom_dist = config.atom_dist if hasattr(config, 'atom_dist') else None
        # self.mask_ratio = config.MASK_RATIO
        self.min_gt = config.min_gt if hasattr(config, 'min_gt') else -1
        self.max_gt = config.max_gt if hasattr(config, 'max_gt') else -1
        self.min_input = config.min_input if hasattr(config, 'min_input') else -1
        self.max_input = config.max_input if hasattr(config, 'max_input') else -1

        

        self.n_renderings = 1 if self.subset == 'train' else 1
        self.file_list = self._get_file_list(self.subset, self.n_renderings)
        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self, subset):
        if subset == 'train':
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            }, 
            # {
            #     'callback': 'RandomMirrorPoints',
            #     'objects': ['partial', 'gt']
            # },
            {
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])
        else:
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])

    def _get_file_list(self, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []
        # base_dir = '/home/curry/pointcloud/PoinTr/data/0701n/' + subset + '/'
        base_dir = os.path.join(self.complete_points_path, subset)
        # samples = os.listdir(base_dir + 'partial')
        samples = os.listdir(base_dir)
        #for dc in self.dataset_categories:
        #    print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='PCNDATASET')
        #    samples = dc[subset]

        for s in samples:
            file_list.append({
                'taxonomy_id':'NA',

                'model_id': s,

                'file_path': os.path.join(base_dir, s),
               
            })


        print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='PCNDATASET')
        return file_list



    def normalize_pc(self, points):
        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
        points /= furthest_distance
        return points

    def get_noise(self, points):
        if isinstance(self.noise_scale, float):
            noise = np.random.normal(loc=0., scale=self.noise_scale, size=points.shape).astype(points.dtype)
        elif isinstance(self.noise_scale, list):
            noise = list()
            for channel_scale in self.noise_scale:
                channel_noise = np.random.normal(loc=0., scale=channel_scale, size=(len(points), 1)).astype(points.dtype)
                noise.append(channel_noise)
            noise = np.concatenate(noise, axis=-1)
        else:
            raise 

        return noise

    def add_shift(self, points):
        shift = np.random.uniform(0, self.shift, size=(1, 3)).astype(points.dtype) 
        return shift


    def set_gt_ratio(self, ratio):
        self.gt_ratio = ratio



    def get_label(self, num_points, scheme=None, shuffle=True):

        # label = np.random.choice(len(self.atom_dist), num_points, self.atom_dist)
        label = np.asarray(random.choices(range(len(self.atom_dist)), k=num_points, weights=self.atom_dist))
        label_tp = torch.from_numpy(label)
        # print(len(self.atom_dist), num_points, self.atom_dist, label_tp.unique(return_counts=True))
        
        return label 







    def __getitem__(self, idx):
        sample = self.file_list[idx]
        gt = IO.get(sample['file_path']).astype(np.float32)
        

        
        np.random.shuffle(gt) 
        
        # there are missing points in real gt 
        n_gt = 1. 
        n_points = gt.shape[0]
        if self.min_gt > 0 and self.max_gt > 0:
            n_gt = random.uniform(self.min_gt, self.max_gt)
            n_points = int(gt.shape[0]*n_gt)



        # get noise
        noise = self.get_noise(gt[:n_points])
        noised_data = noise + gt[:n_points]

        if self.normalize:
            gt = self.normalize_pc(gt)
            noised_data = self.normalize_pc(noised_data)

        input_data = noised_data
        if self.min_input > 0 and self.max_input > 0:
            input_ratio = random.uniform(self.min_input, self.max_input)
            n_input = int(gt.shape[0] * input_ratio)
            input_data = input_data[:n_input].astype(np.float32)
            
        gt = gt.astype(np.float32)
        gt_label = np.zeros_like(input_data)[:, 0]
        if self.atom_dist is not None:
            gt_label = self.get_label(n_points)

        # return sample['taxonomy_id'], sample['model_id'], (input_data, gt)

            
        return sample['taxonomy_id'], sample['model_id'], (input_data, gt, n_points), gt_label

    def __len__(self):
        return len(self.file_list)
        # return 64
