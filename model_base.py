from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct, open, pow, range, round, str, super, zip

import numpy as np
import collections
import heapq

import lib
import data

########################################################################################
class FitListener(object):

    def __init__(self):
        pass
    
    def fit_started(self, model):
        pass
    
    def epoch_started(self, model, epoch_num):
        pass
    
    def training_started(self, model):
        pass
    
    def minibatch_started(self, model, minibatch_num, num_minibatches):
        pass
    
    def minibatch_ended(self, model, minibatch_num, num_minibatches):
        pass
    
    def training_ended(self, model):
        pass
        
    def validation_started(self, model):
        pass
    
    def validation_ended(self, model, validation_loss):
        pass
    
    def epoch_ended(self, model, epoch_num):
        pass
    
    def fit_ended(self, model):
        pass
        
#################################################################
class _Beam(object):
#For use by beam search.
#For comparison of prefixes, the tuple (prefix_logprobability, complete_caption) is used.
#This is so that if two prefixes have equal probabilities then a complete sentence is preferred over an incomplete one since (0.5, False) < (0.5, True)

    #################################################################
    def __init__(self, beam_width):
        self.heap = list()
        self.beam_width = beam_width

    #################################################################
    def add(self, logprob, complete, prefix, prefix_len):
        heapq.heappush(self.heap, (logprob, complete, prefix, prefix_len))
        if len(self.heap) > self.beam_width:
            heapq.heappop(self.heap)

    #################################################################
    def __iter__(self):
        return iter(self.heap)
        
########################################################################################
class Model(object):

    def __init__(self, dataset, val_minibatch_size):
        self.dataset            = dataset
        self.val_minibatch_size = val_minibatch_size
        
    ############################################

    def compile_model(self):
        pass

    ############################################

    def init_params(self):
        pass
    
    ############################################
    
    def save_params(self, param_save_dir):
        pass
    
    ############################################

    def load_params(self, param_save_dir):
        pass
    
    ############################################

    def fit(self, param_save_dir, listener=FitListener()):
        listener.fit_started(self)
        listener.fit_ended(self)
        return None
    
    ############################################
    
    def get_num_params(self):
        return 0
    
    ############################################
    
    def get_raw_probs(self, images, prefixes, prefixes_lens, temperature):
        pass
    
    ############################################

    def image_caption_logprobs(self, images, captions, listener=lambda curr_pos, final_pos:None):
        num_batched_caps = self.val_minibatch_size
        
        cap_logprobs = []
        
        unpadded_batch_caps = []
        batch_caps_lens = []
        batch_images = []
        num_ready = 0
        amount = len(captions)
        for (i, (cap, img)) in enumerate(zip(captions, images)):
            unpadded_batch_caps.append([ self.dataset.token_to_index.get(token, data.UNKNOWN_INDEX) for token in cap ])
            batch_caps_lens.append(len(cap)+1) #include edge
            batch_images.append(img)
            if len(unpadded_batch_caps) == num_batched_caps or i == len(captions)-1: #if batch is full or all captions have been processed
                max_len = max(batch_caps_lens)
                batch_caps = np.zeros([len(unpadded_batch_caps), max_len], np.int32)
                batch_targets = np.zeros([len(unpadded_batch_caps), max_len], np.int32)
                for (j, (indexes, cap_len)) in enumerate(zip(unpadded_batch_caps, batch_caps_lens)):
                    batch_caps[j,:cap_len] = [data.EDGE_INDEX]+indexes
                    batch_targets[j,:cap_len] = indexes+[data.EDGE_INDEX]
                    
                batch_distributions = self.get_raw_probs(batch_images, batch_caps, batch_caps_lens, 1.0)
                    
                for (distribution, targets, cap_len) in zip(batch_distributions, batch_targets, batch_caps_lens):
                    probs = distribution[np.arange(distribution.shape[0]), targets][:cap_len]
                    if 0.0 in probs:
                        cap_logprobs.append(np.inf)
                    else:
                        cap_logprobs.append(np.sum(np.log2(probs)))
                    num_ready += 1
                    listener(num_ready, amount)
                
                del unpadded_batch_caps[:]
                del batch_caps_lens[:]
                del batch_images[:]
        
        return cap_logprobs
    
    ############################################
    
    def image_caption_logperplexities(self, images, captions, listener=lambda curr_pos, final_pos:None):
        # Let P = probability of a caption with L words
        # Let pi = probability of word i in caption
        # P = p1*...*pL
        # log P = (log p1) + ... + (log pL)
        # pplx = 2^(-1/L log P) = 2^(-1/L (log p1 + ... + log pL))
        # log pplx = -1/L (log p1 + ... + log pL) = -1/L logprob = -logprob/L
        logprobs = self.image_caption_logprobs(images, captions, listener)
        return [ -logprob/(len(cap)+1) for (logprob, cap) in zip(logprobs, captions) ]
    
    ############################################
    
    def image_caption_loggeomean_perplexities(self, images, captions, listener=lambda curr_pos, final_pos:None):
        # Let pplxi = perplexity of caption i out of N captions
        # geomean = (pplx1*...*pplxN)**(1/N)
        # log geomean = (1/N) log (pplx1*...*pplxN) = (1/N) (log pplx1 + ... + log pplxN) = (logpplx1 + ... + logpplxN)/N
        logpplxs = self.image_caption_logperplexities(images, captions, listener)
        return (
                np.sum(logpplx for logpplx in logpplxs if not np.isinf(logpplx))/len(captions),
                sum(np.isinf(logpplx) for logpplx in logpplxs)
            )
    
    ############################################

    def generate_captions_beamsearch(self, images, beam_width=3, lower_bound_len=3, upper_bound_len=50, temperature=1.0, listener=lambda curr_pos, final_pos:None):
        num_batched_beams = self.val_minibatch_size//beam_width
        
        num_ready = 0
        amount = len(images)
        is_cap_complete = [ False ]*amount
        complete_caps = [ None ]*amount
        complete_caps_logprobs = [ None ]*amount
        prev_beams = []
        for _ in range(amount):
            beam = _Beam(beam_width)
            beam.add(0.0, False, [data.EDGE_INDEX], 1)
            prev_beams.append(beam)
        while True:
            curr_beams = [ _Beam(beam_width) for _ in range(amount) ]
            batch_orig_indexes = []
            unpadded_batch_prefixes = []
            batch_prefixes_logprobs = []
            batch_prefixes_lens = []
            batch_images = []
            beams_batched = 0
            for i in range(amount):
                if not is_cap_complete[i]:
                    for (prefix_logprob, complete, prefix, prefix_len) in prev_beams[i]:
                        if complete == True:
                            curr_beams[i].add(prefix_logprob, True, prefix, prefix_len)
                        else:
                            batch_orig_indexes.append(i)
                            unpadded_batch_prefixes.append(prefix)
                            batch_prefixes_logprobs.append(prefix_logprob)
                            batch_prefixes_lens.append(prefix_len)
                            batch_images.append(images[i])
                    beams_batched += 1
                    if beams_batched == num_batched_beams:
                        break
            if len(batch_orig_indexes) == 0:
                break
                
            max_len = max(batch_prefixes_lens)
            batch_prefixes = np.zeros([len(unpadded_batch_prefixes), max_len], np.int32)
            for (i, (indexes, prefix_len)) in enumerate(zip(unpadded_batch_prefixes, batch_prefixes_lens)):
                batch_prefixes[i,:prefix_len] = indexes
                
            batch_distributions = self.get_raw_probs(batch_images, batch_prefixes, batch_prefixes_lens, temperature)
            
            grouped_beam_batches = collections.defaultdict(list)
            for (orig_index, prefix, prefix_logprob, prefix_len, distribution_series) in zip(batch_orig_indexes, unpadded_batch_prefixes, batch_prefixes_logprobs, batch_prefixes_lens, batch_distributions):
                distribution = distribution_series[prefix_len-1]
                grouped_beam_batches[orig_index].append((prefix, prefix_logprob, prefix_len, distribution))
                
            for (orig_index, beam_group) in grouped_beam_batches.items():
                for (prefix, prefix_logprob, prefix_len, distribution) in beam_group:
                    for (next_index, next_prob) in enumerate(distribution):
                        if next_index == data.UNKNOWN_INDEX: #skip unknown
                            pass
                        elif next_index == data.EDGE_INDEX: #if next item is the end token then mark prefix as complete and leave out the end token
                            if prefix_len >= lower_bound_len: #only consider terminating the prefix if it has sufficient length
                                curr_beams[orig_index].add(prefix_logprob+np.log2(next_prob), True, prefix, prefix_len)
                        else: #if next item is a non-end token then mark prefix as incomplete (if its length does not exceed the clip length, ignoring start token)
                            curr_beams[orig_index].add(prefix_logprob+np.log2(next_prob), prefix_len == upper_bound_len, prefix+[next_index], prefix_len+1)
                    
                (best_logprob, best_complete, best_prefix, best_prefix_len) = max(curr_beams[orig_index])
                if best_complete == True:
                    is_cap_complete[orig_index] = True
                    curr_beams[orig_index] = None
                    complete_caps[orig_index] = best_prefix[1:best_prefix_len]
                    complete_caps_logprobs[orig_index] = best_logprob
                    num_ready += 1
                    listener(num_ready, amount)

                prev_beams[orig_index] = curr_beams[orig_index]
        
        return [ ([ self.dataset.index_to_token[index] for index in cap ], cap_logprob) for (cap, cap_logprob) in zip(complete_caps, complete_caps_logprobs) ]
    
    ############################################

    def close(self):
        pass
        
    ############################################
    
    def __enter__(self):
        return self
        
    ############################################
    
    def __exit__(self, type, value, traceback):
        self.close()