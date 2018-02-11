from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct, open, pow, range, round, str, super, zip

import GPyOpt
import hyperopt
import copy
import sys
import random
import numpy as np

import config
import lib
import data
import model_normal
import helper_datasources

sys.path.append(config.mscoco_dir)
from pycocoevalcap.cider.cider_scorer import CiderScorer

domains = [
        ('init_method',             [ 'normal', 'xavier_normal' ]), #[ 'uniform', 'normal', 'xavier_uniform', 'xavier_normal' ]
        ('max_init_weight',         [ 0.1, 0.01 ]), #[ 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1 ]
        ('embed_size',              [ 64, 128, 256, 512 ]),
        ('post_image_activation',   [ 'none', 'relu' ]), #[ 'none', 'relu', 'tanh', 'swish' ]
        ('rnn_type',                [ 'gru' ]),
        ('learnable_init_state',    [ False, True ]),
        ('optimizer',               [ 'adam' ]), #[ 'rmsprop', 'adam', 'adagrad' ]
        ('learning_rate',           [ 0.001 ]), #[ 1e-4, 1e-3, 1e-2, 1e-1 ]
        ('normalize_image',         [ False, True ]),
        ('weights_reg_weight',      [ 0.0, 1e-8 ]), #[ 0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1 ]
        ('image_dropout_prob',      [ 0.0, 0.5 ]),
        ('post_image_dropout_prob', [ 0.0, 0.5 ]),
        ('embedding_dropout_prob',  [ 0.0, 0.5 ]),
        ('rnn_dropout_prob',        [ 0.0, 0.5 ]),
        ('train_minibatch_size',    [ 32, 64, 128 ]),
    ]

print('#'*100)
print('Exploring hyperparameters on flickr8k')
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

def objective(hyperparams):
    global i
    global best_cost
    global best_hyperparams
    global phase
    
    [
            init_method,
            max_init_weight,
            embed_size,
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
        ] = [
                values[int(hyperparam)]
                for ((name, values), hyperparam) in zip(domains, hyperparams[0])
            ]
    
    min_init_weight = -max_init_weight
    rnn_size = embed_size
    post_image_size = embed_size
    
    i += 1
    print(
            i,
            phase+(' '*5 if len(phase) < 7 else ''),
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
            sep='\t', end='\t'
        )
    
    timer = lib.Timer()
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
                max_epochs              = config.hyperpar_max_epochs,
                val_minibatch_size      = config.val_minibatch_size,
                train_minibatch_size    = train_minibatch_size,
            ) as model:
        model.compile_model()
        model.init_params()
        model.fit(config.base_dir_hyperpar)
        captions_tokens = model.generate_captions_beamsearch(val_imgs, beam_width=config.hyperpar_beam_width, lower_bound_len=config.beamsearch_lower_bound_len, upper_bound_len=config.beamsearch_upper_bound_len, temperature=config.beamsearch_temperature)
        hyp_caps = [ ' '.join(caption_tokens) for (caption_tokens, prob) in captions_tokens ]
        cider_scorer = CiderScorer(n=4, sigma=6) #Values from MSCOCO evaluation toolkit
        for (hyp, refs) in zip(hyp_caps, val_caps):
            cider_scorer += (hyp, refs)
        cost = -cider_scorer.compute_score()[0]
    duration = timer.get_duration()
    
    if best_cost is None or cost < best_cost:
        best_cost = cost
        best_hyperparams = list(hyperparams[0])
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
            print('max_epochs              = {},'.format(config.hyperpar_max_epochs), file=f)
            print('beam_width              = {},'.format(config.hyperpar_beam_width), file=f)
            
    print(str(cost)+('*' if best_cost == cost else ''), lib.format_duration(duration), sep='\t')
    with open(config.base_dir_hyperpar+'/evaluations_{}.txt'.format(multimodal_method), 'a', encoding='utf-8') as f:
        print(*[ str(x) for x in [
                i,
                phase,
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
                duration
            ] ], sep='\t', file=f)
    
    return cost

if not lib.file_exists(config.base_dir_hyperpar+'/completed.txt'):
    with open(config.base_dir_hyperpar+'/completed.txt', 'w', encoding='utf-8') as f:
        pass
    
with open(config.base_dir_hyperpar+'/completed.txt', 'r', encoding='utf-8') as f:
    completed_multimodal_methods = f.read().strip().split('\n')

for multimodal_method in [ 'merge', 'par', 'pre', 'init' ]:
    if multimodal_method in completed_multimodal_methods:
        continue
        
    timer = lib.Timer()
    
    i = 0
    best_cost = None
    best_hyperparams = None
    phase = None
    
    print('='*100)
    print(lib.formatted_clock())
    print('Multimodal method:', multimodal_method)
    print()
    print(
            '#',
            'phase',
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
            'cost',
            'duration',
            sep='\t'
        )
    
    with open(config.base_dir_hyperpar+'/evaluations_{}.txt'.format(multimodal_method), 'w', encoding='utf-8') as f:
        print(
                '#',
                'phase',
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
                'cost',
                'duration',
                sep='\t', file=f
            )
    
    phase = 'random'
    known_random_hyperpars = dict()
    for _ in range(config.hyperpar_max_evals):
        while True:
            hyperpars = tuple([ random.randrange(len(values)) for (name, values) in domains ])
            if hyperpars not in known_random_hyperpars:
                break
        cost = objective([hyperpars]) #added an extra nested list to make structure compatible with what GPyOpt generates
        known_random_hyperpars[hyperpars] = cost
        
    #http://nbviewer.jupyter.org/github/SheffieldML/GPyOpt/blob/devel/manual/index.ipynb
    phase = 'bayes'
    (X, Y) = zip(*known_random_hyperpars.items())
    X = np.array([ list(x) for x in X ])
    Y = np.array([ [y] for y in Y ])
    bo = GPyOpt.methods.BayesianOptimization(objective, [
                {
                        'name':   name,
                        'type':   'categorical',
                        'domain': tuple(range(len(values)))
                    }
                for (name, values) in domains
            ],
            model_type='GP',
            initial_design_numdata=0,
            X=X,
            Y=Y,
            acquisition_type='EI',
            normalize_Y=False,
            exact_feval=False,
            num_cores=1,
            maximize=False,
            verbosity=False,
            verbosity_model=False,
        )
    bo.run_optimization(config.hyperpar_max_evals, eps=-1, verbosity=False) #number of combinations produced by Baysian optimization (eps=-1 is important to avoid stopping before making hyperpar_max_evals evaluations)
    
    #https://jaberg.github.io/hyperopt/
    phase = 'parzen'
    hyperopt.fmin(
            lambda hyperparams:{ 'loss': objective(hyperparams), 'status': hyperopt.STATUS_OK },
            space=[
                    [ #added an extra nested list to make structure compatible with what GPyOpt generates
                        hyperopt.hp.choice(name, tuple(range(len(values))))
                        for (name, values) in domains
                    ]
                ],
            algo=hyperopt.tpe.suggest,
            max_evals=config.hyperpar_max_evals, #number of combinations produced by Tree of Parzen Estimators
        )
    
    #Fine-tune using greedy hill climbing
    phase = 'finetune'
    previous_best_cost = best_cost
    previous_updated_hyperparam_index = -1
    new_updated_hyperparam_index = -1
    for _ in range(config.hyperpar_finetune_iters):
        hyperparams = copy.deepcopy(best_hyperparams)
        for (hyperparam_index, ((name, values), curr_value_index)) in enumerate(zip(domains, list(hyperparams))):
            for (new_value_index, new_value) in enumerate(values):
                if hyperparam_index != previous_updated_hyperparam_index and new_value_index != curr_value_index:
                    hyperparams[hyperparam_index] = new_value_index
                    
                    objective([hyperparams]) #added an extra nested list to make structure compatible with what GPyOpt generates
                    if best_hyperparams == hyperparams:
                        new_updated_hyperparam_index = hyperparam_index
                    
                    hyperparams[hyperparam_index] = curr_value_index
        
        previous_updated_hyperparam_index = new_updated_hyperparam_index
        if previous_best_cost == best_cost:
            break
        previous_best_cost = best_cost
    
    with open(config.base_dir_hyperpar+'/completed.txt', 'a', encoding='utf-8') as f:
        print(multimodal_method, file=f)
    
    print(lib.format_duration(timer.get_duration()))
    print()
    
print('='*100)
print(lib.formatted_clock())