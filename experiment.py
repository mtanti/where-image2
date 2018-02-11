from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct, open, pow, range, round, str, super, zip

import sys
import numpy as np
import json
import scipy.io
import os
import collections

import config
import lib
import data
import helper_datasources
import model_base
import model_normal
import model_idealmock

sys.path.append(config.mscoco_dir)
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.cider.cider_scorer import CiderScorer

########################################################################################
class _FitListener(model_base.FitListener):
    
    def __init__(self):
        super(_FitListener, self).__init__()
        self.run_timer = lib.Timer()
        self.epoch_timer = lib.Timer()
        self.last_epoch = 0
        self.duration = 0
        self.training_prog = None
    
    def fit_started(self, model):
        print('epoch', 'progress', 'val loss', 'duration', sep='\t')
        self.run_timer.restart()
    
    def epoch_started(self, model, epoch_num):
        self.epoch_timer.restart()
        print(epoch_num, end='\t')
        if epoch_num == 0:
            print(' '*lib.ProgressBar.width(5), end=' | \t')
    
    def training_started(self, model):
        pass
    
    def minibatch_started(self, model, minibatch_num, num_minibatches):
        if minibatch_num == 1:
            self.training_prog = lib.ProgressBar(num_minibatches, 5)
    
    def minibatch_ended(self, model, minibatch_num, num_minibatches):
        self.training_prog.inc_value()
    
    def training_ended(self, model):
        print(' | ', end='\t')
        
    def validation_started(self, model):
        pass
    
    def validation_ended(self, model, validation_loss):
        print(round(validation_loss, 3), end='\t')
    
    def epoch_ended(self, model, epoch_num):
        self.last_epoch = epoch_num
        epoch_time = self.epoch_timer.get_duration()
        print(lib.format_duration(epoch_time))
    
    def fit_ended(self, model):
        print()
        self.duration = self.run_timer.get_duration()

########################################################################################
class ExperimentRunner(object):

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        
        print()
        print('='*100)
        
        print('Starting dataset:', dataset_name)
        datasources = helper_datasources.DataSources(dataset_name)

        lib.create_dir(config.base_dir)
        self.completed = set()
        
        self.dataset = data.Dataset(
                min_token_freq        = config.min_token_freq,
                training_datasource   = datasources.train,
                validation_datasource = datasources.val,
                testing_datasource    = datasources.test,
            )
        self.dataset.process()
        
        print('Num training captions:      ', datasources.train.size)
        print('Max training caption length:', self.dataset.training_proccaps.prefixes_indexes.shape[1]-1)
        print('Vocab size:                 ', self.dataset.vocab_size)
        print()
        
        self.mean_training_caps_len = np.mean([ len(cap) for caption_group in datasources.train.caption_groups for cap in caption_group ])
        
        self.known_train_caps = { ' '.join(cap) for caption_group in datasources.train.caption_groups for cap in caption_group }
        
        self.all_str_test_caps = [ [ ' '.join(cap) for cap in caption_group ] for caption_group in datasources.test.caption_groups ]
        
        self.test_caps = datasources.test.first_captions
        self.test_imgs = datasources.test.images
        
        self.test_caps_ret = datasources.test.first_captions[:1000]
        self.test_imgs_ret = datasources.test.images[:1000]
        
        with open(config.base_dir+'/imgs_'+dataset_name+'.txt', 'w', encoding='utf-8') as f:
            for filename in datasources.test.image_filenames:
                print(str(filename), file=f)
        
        with open(config.base_dir+'/caps_'+dataset_name+'.txt', 'w', encoding='utf-8') as f:
            for cap in datasources.test.first_captions:
                print(str(' '.join(cap)), file=f)
        
        #Prepare MSCOCO evaluation toolkit
        with open(config.mscoco_dir+'/annotations/captions.json', 'w', encoding='utf-8') as f:
            print(str(json.dumps({
                    'info':        {
                            'description':  None,
                            'url':          None,
                            'version':      None,
                            'year':         None,
                            'contributor':  None,
                            'date_created': None,
                        },
                    'images':      [
                            {
                                'license':       None,
                                'url':           None,
                                'file_name':     None,
                                'id':            image_id,
                                'width':         None,
                                'date_captured': None,
                                'height':        None
                            }
                            for image_id in range(len(datasources.test.caption_groups))
                        ],
                    'licenses':    [
                        ],
                    'type':        'captions',
                    'annotations': [
                            {
                                'image_id': image_id,
                                'id':       caption_id,
                                'caption':  ' '.join(caption)
                            }
                            for (caption_id, (image_id, caption)) in enumerate((image_id, caption)
                            for (image_id, caption_group) in enumerate(datasources.test.caption_groups)
                            for caption in caption_group)
                        ]
                })), file=f)

        if not lib.file_exists(config.base_dir+'/results.txt'):
            with open(config.base_dir+'/results.txt', 'w', encoding='utf-8') as f:
                print(*[
                            'dataset_name',
                            'architecture',
                            'run',
                            'vocab_size',
                            'num_training_caps',
                            'mean_training_caps_len',
                            'num_params',
                            'geomean_pplx',
                            'num_inf_pplx',
                            'vocab_used',
                            'vocab_used_frac',
                            'mean_cap_len',
                            'num_existing_caps',
                            'num_existing_caps_frac',
                            'existing_caps_CIDEr',
                            'unigram_entropy',
                            'bigram_entropy',
                            'CIDEr',
                            'METEOR',
                            'ROUGE_L',
                            'Bleu_1',
                            'Bleu_2',
                            'Bleu_3',
                            'Bleu_4',
                            'R@1',
                            'R@5',
                            'R@10',
                            'median_rank',
                            'R@1_frac',
                            'R@5_frac',
                            'R@10_frac',
                            'median_rank_frac',
                            'num_epochs',
                            'training_time',
                            'total_time',
                        ], sep='\t', file=f
                    )
        else:
            with open(config.base_dir+'/results.txt', 'r', encoding='utf-8') as f:
                for line in f.readlines()[1:]:
                    [
                            dataset_name,
                            architecture,
                            run,
                        ] = line.split('\t')[:3]
                    full_name = '_'.join([
                            dataset_name,
                            architecture,
                            run
                        ])
                    self.completed.add(full_name)

    ############################################

    def run(self, architecture, run):
        full_name = '_'.join(str(x) if x is not None else ''
                for x in [
                        self.dataset_name,
                        architecture,
                        run,
                    ]
            )
        if full_name not in self.completed:
            if architecture == 'human':
                with model_idealmock.IdealMockModel(self.test_imgs, self.test_caps) as model:
                    self._run(full_name, model, architecture, run)
            else:
                with model_normal.NormalModel(
                        dataset                 = self.dataset,
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
                    self._run(full_name, model, architecture, run)
        else:
            print(full_name, 'found completed already')
        
        print()

        
    def _run(self, full_name, model, architecture, run):
        timer = lib.Timer()
            
        print('-'*100)
        print(lib.formatted_clock())
        print('Starting', full_name)
        print()
        
        #Create data directory
        lib.create_dir(config.base_dir+'/'+full_name)
        
        fit_listener = _FitListener()
        
        #Model preparation
        model.compile_model()
        model.init_params()
        
        print('Training model...')
        
        self.dataset.minimal_save(config.base_dir+'/'+full_name)
        sub_timer = lib.Timer()
        model.fit(config.base_dir+'/'+full_name, listener=fit_listener)
        print(lib.format_duration(sub_timer.get_duration()))
        print()
        
        print('Evaluating perplexity...')
        
        sub_timer = lib.Timer()
        perplexity_prog = lib.ProgressBar(len(self.test_imgs), 5)
        (loggeomean_pplx, num_inf_pplx) = model.image_caption_loggeomean_perplexities(self.test_imgs, self.test_caps, listener=lambda curr_pos, final_pos:perplexity_prog.inc_value())
        print()
        print(lib.format_duration(sub_timer.get_duration()))
        print()
        
        print('Evaluating generation...')
        
        sub_timer = lib.Timer()
        generation_prog = lib.ProgressBar(len(self.test_imgs), 5)
        if architecture == 'human':
            captions_tokens = model.generate_captions_beamsearch(self.test_imgs, listener=lambda curr_pos, final_pos:generation_prog.inc_value())
        else:
            captions_tokens = model.generate_captions_beamsearch(self.test_imgs, beam_width=config.hyperparams[architecture]['beam_width'], lower_bound_len=config.beamsearch_lower_bound_len, upper_bound_len=config.beamsearch_upper_bound_len, temperature=config.beamsearch_temperature, listener=lambda curr_pos, final_pos:generation_prog.inc_value())
        
        unigram_freqs = collections.defaultdict(lambda:0)
        bigram_freqs  = collections.defaultdict(lambda:0)
        for (caption_tokens, logprob) in captions_tokens:
            caption_tokens = [ token for token in caption_tokens if token in self.dataset.token_to_index ]
            for i in range(len(caption_tokens)):
                unigram = caption_tokens[i]
                unigram_freqs[unigram] += 1
                if i < len(caption_tokens)-1:
                    bigram = (caption_tokens[i], caption_tokens[i+1])
                    bigram_freqs[bigram] += 1
        
        vocab_used = len(unigram_freqs)
        
        unigram_freqs = np.array(list(unigram_freqs.values()))
        unigram_probs = unigram_freqs/unigram_freqs.sum()
        unigram_entropy = -(unigram_probs*np.log2(unigram_probs)).sum()
        
        bigram_freqs = np.array(list(bigram_freqs.values()))
        bigram_probs = bigram_freqs/bigram_freqs.sum()
        bigram_entropy = -(bigram_probs*np.log2(bigram_probs)).sum()
        
        num_existing_caps = 0
        existing_caps = list()
        existing_caps_refs = list()
        cap_lens = list()
        with open(config.base_dir+'/'+full_name+'/generated_captions.txt', 'w', encoding='utf-8') as f:
            for (i, (caption_tokens, logprob)) in enumerate(captions_tokens):
                cap_lens.append(len(caption_tokens))
                cap = ' '.join(caption_tokens)
                if cap in self.known_train_caps:
                    num_existing_caps += 1
                    existing_caps.append(cap)
                    existing_caps_refs.append(self.all_str_test_caps[i])
                print(str(cap), file=f)
                
        if num_existing_caps > 0:
            cider_scorer = CiderScorer(n=4, sigma=6) #Values from MSCOCO evaluation toolkit
            for (hyp, refs) in zip(existing_caps, existing_caps_refs):
                cider_scorer += (hyp, refs)
            existing_caps_cider = cider_scorer.compute_score()[0]
        else:
            existing_caps_cider = ''
        
        #Evaluate with MSCOCO evaluation toolkit
        with open(config.mscoco_dir+'/results/generated_captions.json', 'w', encoding='utf-8') as f_out:
            print(str(json.dumps([
                    {
                            'image_id': image_id,
                            'caption':  ' '.join(caption_tokens)
                        }
                    for (image_id, (caption_tokens, logprob)) in enumerate(captions_tokens)
                ])), file=f_out)
        print()
        coco = COCO(config.mscoco_dir+'/annotations/captions.json')
        cocoRes = coco.loadRes(config.mscoco_dir+'/results/generated_captions.json')
        cocoEval = COCOEvalCap(coco, cocoRes)
        cocoEval.evaluate()
        print(lib.format_duration(sub_timer.get_duration()))
        print()
        
        print('Evaluating retrieval...')
        
        sub_timer = lib.Timer()
        r1  = 0
        r5  = 0
        r10 = 0
        ranks = list()
        retrieval_prog = lib.ProgressBar(len(self.test_imgs_ret)**2, 5)
        with open(config.base_dir+'/'+full_name+'/retrieved_images.txt', 'w', encoding='utf-8') as f:
            for (correct_index, cap) in enumerate(self.test_caps_ret):
                logprobs = model.image_caption_logprobs(self.test_imgs_ret, [cap]*len(self.test_imgs_ret), listener=lambda curr_pos, final_pos:retrieval_prog.inc_value())
                retrieved_indexes = sorted(range(len(self.test_imgs_ret)), key=lambda i:logprobs[i], reverse=True)
                correct_index_pos = retrieved_indexes.index(correct_index)
                if correct_index_pos == 0:
                    r1 += 1
                if correct_index_pos < 5:
                    r5 += 1
                if correct_index_pos < 10:
                    r10 += 1
                ranks.append(correct_index_pos+1)
                print(str('\t'.join(str(p) for p in logprobs)), file=f)
        median_rank = np.median(ranks)
        print()
        print(lib.format_duration(sub_timer.get_duration()))
                
        print()

        #Save result
        duration = timer.get_duration()
        with open(config.base_dir+'/results.txt', 'a', encoding='utf-8') as f:
            print(*[
                    str(x) if x is not None else ''
                    for x in [
                            self.dataset_name,
                            architecture,
                            run,
                            self.dataset.vocab_size,
                            self.dataset.training_datasource.size,
                            self.mean_training_caps_len,
                            model.get_num_params(),
                            2**loggeomean_pplx,
                            num_inf_pplx,
                            vocab_used,
                            vocab_used/self.dataset.vocab_size,
                            np.mean(cap_lens),
                            num_existing_caps,
                            num_existing_caps/len(self.test_imgs),
                            existing_caps_cider,
                            unigram_entropy,
                            bigram_entropy,
                            cocoEval.eval['CIDEr'],
                            cocoEval.eval['METEOR'],
                            cocoEval.eval['ROUGE_L'],
                            cocoEval.eval['Bleu_1'],
                            cocoEval.eval['Bleu_2'],
                            cocoEval.eval['Bleu_3'],
                            cocoEval.eval['Bleu_4'],
                            r1,
                            r5,
                            r10,
                            median_rank,
                            r1/len(self.test_imgs_ret),
                            r5/len(self.test_imgs_ret),
                            r10/len(self.test_imgs_ret),
                            median_rank/len(self.test_imgs_ret),
                            fit_listener.last_epoch,
                            fit_listener.duration,
                            duration,
                        ]
                ],
                sep='\t', file=f
            )
        
        print(lib.format_duration(duration))
        print()

print('#'*100)
print(lib.formatted_clock())

for dataset_name in [ 'flickr8k', 'flickr30k', 'mscoco' ]:
    exp = ExperimentRunner(dataset_name)
    for run in range(1, config.num_runs+1):
        if run == 1:
            exp.run('human', run)
        for multimodal_method in [ 'init', 'pre', 'par', 'merge' ]:
            exp.run(multimodal_method, run)

print('='*100)
print(lib.formatted_clock())
