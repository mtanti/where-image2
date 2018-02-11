from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct, open, pow, range, round, str, super, zip

import json
import sys
import numpy as np

import config
import lib
import data
import model_normal
import helper_datasources

sys.path.append(config.mscoco_dir)
from pycocoevalcap.cider.cider_scorer import CiderScorer

print('#'*100)
print('Exploring hyperparameters (2) on flickr8k')
datasources = helper_datasources.DataSources('flickr8k')

lib.create_dir(config.base_dir_hyperpar)

dataset = data.Dataset(
        min_token_freq        = config.min_token_freq,
        training_datasource   = datasources.train,
        validation_datasource = datasources.val,
    )
dataset.process()

val_caps = [ [ ' '.join(cap) for cap in cap_group ] for cap_group in datasources.val.caption_groups ]
val_imgs  = datasources.val.images

if not lib.file_exists(config.base_dir_hyperpar+'/completed2.txt'):
    with open(config.base_dir_hyperpar+'/completed2.txt', 'w', encoding='utf-8') as f:
        pass
    
with open(config.base_dir_hyperpar+'/completed2.txt', 'r', encoding='utf-8') as f:
    completed_multimodal_methods = f.read().strip().split('\n')

for multimodal_method in [ 'merge', 'par', 'pre', 'init' ]:
    if multimodal_method in completed_multimodal_methods:
        continue
    
    timer = lib.Timer()
    
    print('='*100)
    print(lib.formatted_clock())
    print('Multimodal method:', multimodal_method)
    print()
    
    print(*([
                'init_method',
                'min_init_weight',
                'max_init_weight',
                'embed_size',
                'rnn_size',
                'post_image_size',
                'post_image_activation',
                'rnn_type',
                'learnable_init_state',
                'optimizer',
                'learning_rate',
                'normalize_image',
                'weights_reg_weight',
                'image_dropout_prob',
                'post_image_dropout_prob',
                'embedding_dropout_prob',
                'rnn_dropout_prob',
                'train_minibatch_size',
                'max_epochs',
                'beam_width'
            ]+[ 'cost_{}'.format(i+1) for i in range(config.num_runs) ]+[
                'cost',
                'duration'
            ]),
            sep='\t'
        )
    with open(config.base_dir_hyperpar+'/evaluations2_{}.txt'.format(multimodal_method), 'w', encoding='utf-8') as f:
        print(
                'init_method',
                'min_init_weight',
                'max_init_weight',
                'embed_size',
                'rnn_size',
                'post_image_size',
                'post_image_activation',
                'rnn_type',
                'learnable_init_state',
                'optimizer',
                'learning_rate',
                'normalize_image',
                'weights_reg_weight',
                'image_dropout_prob',
                'post_image_dropout_prob',
                'embedding_dropout_prob',
                'rnn_dropout_prob',
                'train_minibatch_size',
                'max_epochs',
                'beam_width',
                'cost',
                'duration',
                sep='\t', file=f
            )
    
    with open(config.base_dir_hyperpar+'/evaluations_{}.txt'.format(multimodal_method), 'r', encoding='utf-8') as f:
        hyperparam_costs = dict()
        for line in f.read().strip().split('\n')[1:]:
            (
                    init_method,
                    min_init_weight,
                    max_init_weight,
                    embed_size,
                    rnn_size,
                    post_image_size,
                    post_image_activation,
                    rnn_type,
                    learnable_init_state,
                    optimizer,
                    learning_rate,
                    normalize_image,
                    weights_reg_weight,
                    image_dropout_prob,
                    post_image_dropout_prob,
                    embedding_dropout_prob,
                    rnn_dropout_prob,
                    train_minibatch_size,
                    cost,
                ) = line.split('\t')[2:-1]
            hyperparams = (
                    init_method,
                    float(min_init_weight),
                    float(max_init_weight),
                    int(embed_size),
                    int(rnn_size),
                    int(post_image_size),
                    post_image_activation,
                    rnn_type,
                    learnable_init_state == 'True',
                    optimizer,
                    float(learning_rate),
                    normalize_image == 'True',
                    float(weights_reg_weight),
                    float(image_dropout_prob),
                    float(post_image_dropout_prob),
                    float(embedding_dropout_prob),
                    float(rnn_dropout_prob),
                    int(train_minibatch_size),
                )
            cost = float(cost)
            
            if hyperparams not in hyperparam_costs:
                hyperparam_costs[hyperparams] = [ cost ]
            else:
                hyperparam_costs[hyperparams].append(cost)
        
    hyperparam_candidates = sorted(hyperparam_costs.keys(), key=lambda hyp:np.mean(hyperparam_costs[hyp]))[:config.hyperpar_num_candidate_hyps]
    del hyperparam_costs
    
    def hamming_dist(hyp1, hyp2):
        return sum(a != b for (a,b) in zip(hyp1, hyp2))
    def min_hamming_dist(hyp, group):
        return min(hamming_dist(hyp, other) for other in group)
    best_hyperparams = [hyperparam_candidates.pop(0)]
    for _ in range(config.hyperpar_num_best_hyps-1):
        next_best_i = max(range(len(hyperparam_candidates)), key=lambda i:(min_hamming_dist(hyperparam_candidates[i], best_hyperparams),-i))
        best_hyperparams.append(hyperparam_candidates.pop(next_best_i))
    
    best_cost = None
    for (
            init_method,
            min_init_weight,
            max_init_weight,
            embed_size,
            rnn_size,
            post_image_size,
            post_image_activation,
            rnn_type,
            learnable_init_state,
            optimizer,
            learning_rate,
            normalize_image,
            weights_reg_weight,
            image_dropout_prob,
            post_image_dropout_prob,
            embedding_dropout_prob,
            rnn_dropout_prob,
            train_minibatch_size,
        ) in best_hyperparams:
        for max_epochs in [ 10, 100 ]:
            with model_normal.NormalModel(
                    dataset                 = dataset,
                    init_method             = init_method,
                    min_init_weight         = min_init_weight,
                    max_init_weight         = max_init_weight,
                    embed_size              = embed_size,
                    rnn_size                = rnn_size,
                    post_image_size         = post_image_size,
                    post_image_activation   = post_image_activation,
                    rnn_type                = rnn_type,
                    learnable_init_state    = learnable_init_state,
                    multimodal_method       = multimodal_method,
                    optimizer               = optimizer,
                    learning_rate           = learning_rate,
                    normalize_image         = normalize_image,
                    weights_reg_weight      = weights_reg_weight,
                    image_dropout_prob      = image_dropout_prob,
                    post_image_dropout_prob = post_image_dropout_prob,
                    embedding_dropout_prob  = embedding_dropout_prob,
                    rnn_dropout_prob        = rnn_dropout_prob,
                    max_epochs              = max_epochs,
                    val_minibatch_size      = config.val_minibatch_size,
                    train_minibatch_size    = train_minibatch_size,
                ) as model:
                
                model.compile_model()
                for beam_width in range(1, config.hyperpar_max_beam_width+1):
                    inner_timer = lib.Timer()
                    print(
                            init_method+(' '*5 if len(init_method) < 10 else ''),
                            min_init_weight,
                            max_init_weight,
                            embed_size,
                            rnn_size,
                            post_image_size,
                            post_image_activation,
                            rnn_type,
                            learnable_init_state,
                            optimizer,
                            learning_rate,
                            normalize_image,
                            weights_reg_weight,
                            image_dropout_prob,
                            post_image_dropout_prob,
                            embedding_dropout_prob,
                            rnn_dropout_prob,
                            train_minibatch_size,
                            max_epochs,
                            beam_width,
                            end='\t', sep='\t'
                        )
                    costs = []
                    for run in range(config.num_runs):
                        model.init_params()
                        model.fit(config.base_dir_hyperpar)
                        
                        captions_tokens = model.generate_captions_beamsearch(val_imgs, beam_width=beam_width, lower_bound_len=config.beamsearch_lower_bound_len, upper_bound_len=config.beamsearch_upper_bound_len, temperature=config.beamsearch_temperature)
                        hyp_caps = [ ' '.join(caption_tokens) for (caption_tokens, prob) in captions_tokens ]
                        cider_scorer = CiderScorer(n=4, sigma=6) #Values from MSCOCO evaluation toolkit
                        for (hyp, refs) in zip(hyp_caps, val_caps):
                            cider_scorer += (hyp, refs)
                        cost = -cider_scorer.compute_score()[0]
                        print(cost, end='\t')
                        costs.append(cost)
                    duration = inner_timer.get_duration()
                    
                    cost = np.mean(costs)
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        with open(config.base_dir_hyperpar+'/result_{}.txt'.format(multimodal_method), 'w', encoding='utf-8') as f:
                            print('Best cost:', str(best_cost), file=f)
                            print('', file=f)
                            print('init_method             = \'{}\','.format(init_method), file=f)
                            print('min_init_weight         = {},'.format(min_init_weight), file=f)
                            print('max_init_weight         = {},'.format(max_init_weight), file=f)
                            print('embed_size              = {},'.format(embed_size), file=f)
                            print('rnn_size                = {},'.format(rnn_size), file=f)
                            print('post_image_size         = {},'.format(post_image_size), file=f)
                            print('post_image_activation   = \'{}\','.format(post_image_activation), file=f)
                            print('rnn_type                = \'{}\','.format(rnn_type), file=f)
                            print('learnable_init_state    = {},'.format(learnable_init_state), file=f)
                            print('optimizer               = \'{}\','.format(optimizer), file=f)
                            print('learning_rate           = {},'.format(learning_rate), file=f)
                            print('normalize_image         = {},'.format(normalize_image), file=f)
                            print('weights_reg_weight      = {},'.format(weights_reg_weight), file=f)
                            print('image_dropout_prob      = {},'.format(image_dropout_prob), file=f)
                            print('post_image_dropout_prob = {},'.format(post_image_dropout_prob), file=f)
                            print('embedding_dropout_prob  = {},'.format(embedding_dropout_prob), file=f)
                            print('rnn_dropout_prob        = {},'.format(rnn_dropout_prob), file=f)
                            print('train_minibatch_size    = {},'.format(train_minibatch_size), file=f)
                            print('max_epochs              = {},'.format(max_epochs), file=f)
                            print('beam_width              = {},'.format(beam_width), file=f)
                            
                    print(str(cost)+('*' if best_cost == cost else ''), lib.format_duration(duration), sep='\t')
                    with open(config.base_dir_hyperpar+'/evaluations2_{}.txt'.format(multimodal_method), 'a', encoding='utf-8') as f:
                        print(*[ str(x) for x in [
                                init_method,
                                min_init_weight,
                                max_init_weight,
                                embed_size,
                                rnn_size,
                                post_image_size,
                                post_image_activation,
                                rnn_type,
                                learnable_init_state,
                                optimizer,
                                learning_rate,
                                normalize_image,
                                weights_reg_weight,
                                image_dropout_prob,
                                post_image_dropout_prob,
                                embedding_dropout_prob,
                                rnn_dropout_prob,
                                train_minibatch_size,
                                max_epochs,
                                beam_width,
                                cost,
                                duration
                            ] ], sep='\t', file=f)
                
    with open(config.base_dir_hyperpar+'/completed2.txt', 'a', encoding='utf-8') as f:
        print(multimodal_method, file=f)
        
    print(lib.format_duration(timer.get_duration()))
    print()
            
print('='*100)
print(lib.formatted_clock())
