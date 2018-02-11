from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct, open, pow, range, round, str, super, zip

import json
import sys
import numpy as np
import scipy.stats
import scipy.spatial.distance

import config
import lib
import data
import model_normal
import helper_datasources

print('#'*100)
print('Measuring multimodal vector differences')

num_trials = 100

for target_caplen in [ 5+1, 10+1, 15+1, 20+1 ]: #including start token
    with open(config.base_dir+'/multimodal_diffs_results_{}.txt'.format(target_caplen), 'w', encoding='utf-8') as f:
        print(*[
                    'dataset_name',
                    'architecture',
                    'run',
                    'num_caps',
                ] + [
                    'cos_word{}'.format(i) for i in range(target_caplen)
                ] + [
                    'euc_word{}'.format(i) for i in range(target_caplen)
                ] + [
                    'meanerr_word{}'.format(i) for i in range(target_caplen)
                ], sep='\t', file=f
            )

    for dataset_name in [ 'mscoco', 'flickr30k', 'flickr8k' ]:
        datasources = helper_datasources.DataSources(dataset_name)
        train_imgs = datasources.train.images
        dataset = data.Dataset(
                min_token_freq        = config.min_token_freq,
                training_datasource   = datasources.train,
                testing_datasource    = datasources.test,
            )
        dataset.process()
        
        test_imgs    = list()
        test_caps    = list()
        test_caplens = list()
        for (img, cap, caplen) in zip(dataset.testing_images, dataset.testing_proccaps.prefixes_indexes, dataset.testing_proccaps.prefixes_lens):
            if caplen == target_caplen:
                test_imgs.append(img)
                test_caps.append(cap)
                test_caplens.append(caplen)
        num_caps = len(test_caps)

        for run in range(1, config.num_runs+1):
            for architecture in [ 'init', 'pre', 'par', 'merge' ]:
                full_name = '_'.join(str(x) if x is not None else ''
                        for x in [
                                dataset_name,
                                architecture,
                                run,
                            ]
                    )
                print('Starting', target_caplen, full_name)
                
                with model_normal.NormalModel(
                        dataset                 = dataset,
                        init_method             = config.hyperparams[architecture]['init_method'],
                        min_init_weight         = config.hyperparams[architecture]['min_init_weight'],
                        max_init_weight         = config.hyperparams[architecture]['max_init_weight'],
                        embed_size              = config.hyperparams[architecture]['embed_size'],
                        rnn_size                = config.hyperparams[architecture]['rnn_size'],
                        post_image_size         = config.hyperparams[architecture]['post_image_size'],
                        post_image_activation   = config.hyperparams[architecture]['post_image_activation'],
                        rnn_type                = config.hyperparams[architecture]['rnn_type'],
                        learnable_init_state    = config.hyperparams[architecture]['learnable_init_state'],
                        multimodal_method       = architecture,
                        optimizer               = config.hyperparams[architecture]['optimizer'],
                        learning_rate           = config.hyperparams[architecture]['learning_rate'],
                        normalize_image         = config.hyperparams[architecture]['normalize_image'],
                        weights_reg_weight      = config.hyperparams[architecture]['weights_reg_weight'],
                        image_dropout_prob      = config.hyperparams[architecture]['image_dropout_prob'],
                        post_image_dropout_prob = config.hyperparams[architecture]['post_image_dropout_prob'],
                        embedding_dropout_prob  = config.hyperparams[architecture]['embedding_dropout_prob'],
                        rnn_dropout_prob        = config.hyperparams[architecture]['rnn_dropout_prob'],
                        max_epochs              = config.hyperparams[architecture]['max_epochs'] if not config.debug else 2,
                        val_minibatch_size      = config.val_minibatch_size,
                        train_minibatch_size    = config.hyperparams[architecture]['train_minibatch_size'],
                    ) as model:
                    model.compile_model()
                    model.load_params(config.base_dir+'/'+full_name)
                    
                    prog = lib.ProgressBar(num_trials, 5)
                    for trial in range(num_trials):
                        other_imgs = list(train_imgs) #make sure that none of the 'other images' match the captions
                        np.random.seed(trial)
                        np.random.shuffle(other_imgs)
                        other_imgs = other_imgs[:len(test_caps)]
                        
                        dists_cos     = [ [] for _ in range(target_caplen) ]
                        dists_euc     = [ [] for _ in range(target_caplen) ]
                        dists_meanerr = [ [] for _ in range(target_caplen) ]
                        num_minibatches = int(np.ceil(len(test_imgs)/config.val_minibatch_size))
                        for i in range(num_minibatches):
                            orig_multimodal_vectors = model.raw_run(
                                    model.multimodal_vectors,
                                    images=test_imgs[i*config.val_minibatch_size:(i+1)*config.val_minibatch_size],
                                    prefixes=test_caps[i*config.val_minibatch_size:(i+1)*config.val_minibatch_size],
                                    prefixes_lens=test_caplens[i*config.val_minibatch_size:(i+1)*config.val_minibatch_size],
                                    temperature=config.beamsearch_temperature
                                )
                            new_multimodal_vectors = model.raw_run(
                                    model.multimodal_vectors,
                                    images=other_imgs[i*config.val_minibatch_size:(i+1)*config.val_minibatch_size],
                                    prefixes=test_caps[i*config.val_minibatch_size:(i+1)*config.val_minibatch_size],
                                    prefixes_lens=test_caplens[i*config.val_minibatch_size:(i+1)*config.val_minibatch_size],
                                    temperature=config.beamsearch_temperature
                                )
                            
                            for (orig_vec_seq, new_vec_seq) in zip(orig_multimodal_vectors, new_multimodal_vectors):
                                for i in range(target_caplen):
                                    dists_cos[i].append(scipy.spatial.distance.cosine(orig_vec_seq[i], new_vec_seq[i]))
                                    dists_euc[i].append(scipy.spatial.distance.euclidean(orig_vec_seq[i], new_vec_seq[i]))
                                    dists_meanerr[i].append(np.abs(orig_vec_seq[i] - new_vec_seq[i])/orig_vec_seq.shape[-1])
                        
                        prog.inc_value()
                    print()
                        
                    with open(config.base_dir+'/multimodal_diffs_results_{}.txt'.format(target_caplen), 'a', encoding='utf-8') as f:
                        print(*[
                                    str(x) if x is not None else ''
                                    for x in [
                                            dataset_name,
                                            architecture,
                                            run,
                                            num_caps,
                                        ] + [
                                            np.mean(d) for d in dists_cos
                                        ] + [
                                            np.mean(d) for d in dists_euc
                                        ] + [
                                            np.mean(d) for d in dists_meanerr
                                        ]
                                ],
                                sep='\t', file=f
                            )
                            
print('='*100)
print(lib.formatted_clock())