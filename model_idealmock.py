from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct, open, pow, range, round, str, super, zip

import numpy as np

import lib
import model_base

########################################################################################
class IdealMockModel(model_base.Model):

    def __init__(self, test_imgs, test_caps):
        super(IdealMockModel, self).__init__(None, None)
        
        self.oracle = { tuple(img): cap for (img, cap) in zip(test_imgs, test_caps) }
    
    ############################################

    def image_caption_logprobs(self, images, captions, listener=lambda curr_pos, final_pos:None):
        num_ready = 0
        amount = len(captions)
        cap_logprobs = list()
        for (cap, img) in zip(captions, images):
            if self.oracle[tuple(img)] == cap:
                cap_logprobs.append(0.0)
            else:
                cap_logprobs.append(-np.inf)
            
            num_ready += 1
            listener(num_ready, amount)
        
        return cap_logprobs
    
    ############################################
    
    def generate_captions_beamsearch(self, images, listener=lambda curr_pos, final_pos:None):
        num_ready = 0
        amount = len(images)
        complete_caps = list()
        complete_caps_logprobs = list()
        for img in images:
            complete_caps.append(self.oracle[tuple(img)])
            complete_caps_logprobs.append(0.0)
            
            num_ready += 1
            listener(num_ready, amount)

        return list(zip(complete_caps, complete_caps_logprobs))