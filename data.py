from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct, open, pow, range, round, str, super, zip

import collections
import numpy as np
import json

EDGE_INDEX = 0
UNKNOWN_INDEX = 1

################################################################################
class DataSource(object):

    def __init__(self, caption_groups, images=None, image_filenames=None, indexes=None):
        self.indexes         = list(range(len(caption_groups))) if indexes is None else indexes
        self.caption_groups  = caption_groups
        self.first_captions  = [ group[0] for group in caption_groups ]
        self.images          = images
        self.image_filenames = image_filenames
        self.size            = sum(len(group) for group in caption_groups)

    def shuffle(self):
        seed = np.random.randint(0, 0xFFFFFFFF, dtype=np.uint32)
        rand = np.random.RandomState()
        
        rand.seed(seed)
        rand.shuffle(self.indexes)
        rand.seed(seed)
        rand.shuffle(self.caption_groups)
        if self.images is not None:
            rand.seed(seed)
            rand.shuffle(self.images)

    def sublist(self, groups_prefix_size):
        return DataSource(
                caption_groups = self.caption_groups[:groups_prefix_size],
                images         = self.images[:groups_prefix_size] if self.images is not None else None,
                indexes        = self.indexes[:groups_prefix_size],
            )
    
    def first_caption_only(self):
        return DataSource(
                caption_groups = [ [ group[0] ] for group in self.caption_groups ],
                images         = self.images if self.images is not None else None,
                indexes        = self.indexes,
            )
    
    def text_only(self):
        return DataSource(
                caption_groups = self.caption_groups,
                indexes        = self.indexes,
            )
    
    def get_vocab(self, min_token_freq):
        all_tokens = (token for cap_group in self.caption_groups for cap in cap_group for token in cap)
        token_freqs = collections.Counter(all_tokens)
        vocab = sorted(token_freqs.keys(), key=lambda token:(-token_freqs[token], token))
        while token_freqs[vocab[-1]] < min_token_freq:
            vocab.pop()
        vocab = [ '<EDG>', '<UNK>' ] + sorted(vocab)
        
        assert vocab[EDGE_INDEX] == '<EDG>'
        assert vocab[UNKNOWN_INDEX] == '<UNK>'
        
        return vocab

################################################################################
class ProcessedCaptions(object):

    def __init__(self, prefixes_indexes, prefixes_lens, targets_indexes):
        self.prefixes_indexes = prefixes_indexes
        self.prefixes_lens    = prefixes_lens
        self.targets_indexes  = targets_indexes

########################################################################################
class Dataset(object):

    def __init__(self, min_token_freq=None, training_datasource=None, validation_datasource=None, testing_datasource=None):
        self.min_token_freq        = min_token_freq
        self.training_datasource   = training_datasource
        self.validation_datasource = validation_datasource
        self.testing_datasource    = testing_datasource
        self.loaded                = False
        
        self.vocab          = None
        self.vocab_size     = None
        self.token_to_index = None
        self.index_to_token = None

        self.training_images     = None
        self.training_proccaps   = None
        self.validation_images   = None
        self.validation_proccaps = None
        self.testing_images      = None
        self.testing_proccaps    = None
        
    ############################################
    def minimal_load(self, data_save_dir):
        with open(data_save_dir+'/vocab.json', 'r', encoding='utf-8') as f:
            vocab = json.loads(f.read())
        assert vocab[EDGE_INDEX] == '<EDG>'
        assert vocab[UNKNOWN_INDEX] == '<UNK>'
        
        self.vocab          = vocab
        self.vocab_size     = len(self.vocab)
        self.token_to_index = { token: i for (i, token) in enumerate(self.vocab) }
        self.index_to_token = { i: token for (i, token) in enumerate(self.vocab) }
        self.loaded         = True
    
    ############################################
    def minimal_save(self, data_save_dir):
        with open(data_save_dir+'/vocab.json', 'w', encoding='utf-8') as f:
            print(str(json.dumps(self.vocab)), file=f)
    
    ############################################
    def process(self, vocab=None):
        if self.training_datasource is None:
            raise ValueError('Cannot process a dataset without a training data source')
        if self.min_token_freq is None == vocab is None:
            raise ValueError('Cannot set or leave out both min_token_freq and vocab')

        if vocab is None:
            self.vocab = self.training_datasource.get_vocab(self.min_token_freq)
        else:
            self.vocab = vocab
        self.vocab_size     = len(self.vocab)
        self.token_to_index = { token: i for (i, token) in enumerate(self.vocab) }
        self.index_to_token = { i: token for (i, token) in enumerate(self.vocab) }

        (self.training_proccaps, self.training_images) = self._process_captions(self.training_datasource)
        if self.validation_datasource is not None:
            (self.validation_proccaps, self.validation_images) = self._process_captions(self.validation_datasource)
        if self.testing_datasource is not None:
            (self.testing_proccaps, self.testing_images) = self._process_captions(self.testing_datasource)

    ############################################
    def _process_captions(self, datasource):
        raw_indexes = list()
        raw_lens = list()
        images = list()
        for (i, cap_group) in enumerate(datasource.caption_groups):
            for cap in cap_group:
                if datasource.images is not None:
                    images.append(datasource.images[i])
                cap_indexes = [ self.token_to_index.get(token, UNKNOWN_INDEX) for token in cap ]
                raw_indexes.append(cap_indexes)
                raw_lens.append(len(cap)+1) #add 1 due to edge token

        max_len = max(raw_lens)

        prefixes_indexes = np.zeros([datasource.size, max_len], np.int32)
        prefixes_lens    = np.array(raw_lens, np.int32)
        targets_indexes  = np.zeros([datasource.size, max_len], np.int32)
        for (i, cap_indexes) in enumerate(raw_indexes):
            prefixes_indexes[i,:len(cap_indexes)+1] = [EDGE_INDEX]+cap_indexes
            targets_indexes [i,:len(cap_indexes)+1] = cap_indexes+[EDGE_INDEX]

        return (ProcessedCaptions(prefixes_indexes, prefixes_lens, targets_indexes), np.array(images) if datasource.images is not None else None)
