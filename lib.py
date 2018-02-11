from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct, open, pow, range, round, str, super, zip

import os
import os.path
import errno
import sys
import timeit
import time
import heapq
import numpy as np

################################################################################
class ProgressBar(object):

    def __init__(self, max_value, perc_inc):
        if perc_inc <= 0 or 100%perc_inc != 0:
            raise ValueError('{} is not a factor of 100.'.format(perc_inc))
        self.curr_value      = 0
        self.max_value       = max_value
        self.perc_inc        = perc_inc
        self.last_perc_shown = 0

    def reset(self):
        self.curr_value      = 0
        self.last_perc_shown = 0

    def inc_value(self, amount=1):
        self.curr_value += amount
        if self.curr_value > self.max_value:
            raise ValueError('Current value {} has been increased beyond maximum value {} after being incremented by {}.'.format(self.curr_value, self.max_value, amount))
        curr_perc = int(self.curr_value/self.max_value*100)
        if curr_perc >= self.last_perc_shown+self.perc_inc:
            for p in range(self.last_perc_shown+self.perc_inc, curr_perc+1, self.perc_inc):
                if self.last_perc_shown != 0:
                    print(' {}%'.format(p), end='')
                else:
                    print('{}%'.format(p), end='')
                self.last_perc_shown = p

    @staticmethod
    def width(perc_inc):
        all_percs = ' '.join('{}%'.format(p) for p in range(0+perc_inc, 100+1, perc_inc))
        return len(all_percs)

################################################################################
def create_dir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

################################################################
def file_exists(path):
    return os.path.isfile(path)

################################################################################
def format_duration(seconds):
    remainder = seconds
    (hours, remainder) = divmod(remainder, 60*60)
    (minutes, remainder) = divmod(remainder, 60)

    if hours > 0:
        return '{:>2}h:{:>2}m:{:>2}s'.format(hours,minutes,remainder)
    elif minutes > 0:
        return '    {:>2}m:{:>2}s'.format(minutes,remainder)
    else:
        return '        {:>2}s'.format(remainder)

################################################################################
class Timer(object):

    def __init__(self):
        self.start_time = timeit.default_timer()
    
    def restart(self):
        self.start_time = timeit.default_timer()
    
    def get_duration(self):
        return round(timeit.default_timer() - self.start_time)

#################################################################
def formatted_clock():
    return time.strftime('%Y/%m/%d %H:%M:%S')