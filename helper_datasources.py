from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct, open, pow, range, round, str, super, zip

import sys
import numpy as np
import json
import scipy.io
import os

import config
import lib
import data

################################################################
class DataSources(object):
    def __init__(self, dataset_name):
        raw_data_dir = config.raw_data_dir(dataset_name)
        
        with open(raw_data_dir+'/dataset.json', 'r', encoding='utf-8') as captions_f:
            captions_data = json.load(captions_f)['images']
        features = scipy.io.loadmat(raw_data_dir+'/vgg_feats.mat')['feats'].T #image features matrix are transposed

        raw_dataset = {
                'train': { 'filenames': list(), 'images': list(), 'captions': list() },
                'val':   { 'filenames': list(), 'images': list(), 'captions': list() },
                'test':  { 'filenames': list(), 'images': list(), 'captions': list() },
            }
        for (image_id, (caption_data, image)) in enumerate(zip(captions_data, features)):
            split = caption_data['split']
            if split == 'restval':
                continue
            filename = caption_data['filename']
            caption_group = [ caption['tokens'] for caption in caption_data['sentences'] ]
            #image = image/np.linalg.norm(image)

            raw_dataset[split]['filenames'].append(filename)
            raw_dataset[split]['images'].append(image)
            raw_dataset[split]['captions'].append(caption_group)

        if config.debug:
            for split in raw_dataset:
                for column in raw_dataset[split]:
                    raw_dataset[split][column] = raw_dataset[split][column][:500]

        self.train = data.DataSource(caption_groups=raw_dataset['train']['captions'], images=np.array(raw_dataset['train']['images']), image_filenames=raw_dataset['train']['filenames'])
        self.val   = data.DataSource(caption_groups=raw_dataset['val']  ['captions'], images=np.array(raw_dataset['val']  ['images']), image_filenames=raw_dataset['val']['filenames'])
        self.test  = data.DataSource(caption_groups=raw_dataset['test'] ['captions'], images=np.array(raw_dataset['test'] ['images']), image_filenames=raw_dataset['test']['filenames'])
