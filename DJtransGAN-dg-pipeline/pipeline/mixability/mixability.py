import warnings
warnings.filterwarnings('ignore')

import os
import copy
import torch
import random
import librosa
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
# tf_slim (used by the vendored music_puzzle_games/model.py - see
# script/patch_music_puzzle_games.py) still references this pre-2.16
# internal location for the Tensor class.
from tensorflow.python.framework import ops as _tf_ops
if not hasattr(_tf_ops, 'Tensor'):
    _tf_ops.Tensor = tf.Tensor
import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()

from pipeline.config          import settings
from pipeline.mixability      import read_audio
from music_puzzle_games.model import SEN


net = SEN(is_train=False)


class Batch():
    def __init__(self):
        self.x1 = []
        self.x2 = []
        self.y = []
        
    
def key_compatibility(src_key, cand_key):
    if src_key == -1 or cand_key == -1:
        return 1
    diff = abs(src_key-cand_key)
    return 1 if min(diff, 13 - diff) <= settings.KEY_DIFF else 0
    
def bpm_compatibility(src_bpm, cand_bpm):
    diff = abs(src_bpm - cand_bpm)
    return 1 if diff <= settings.BPM_DIFF else 0


def estimiate_compatibility_by_rule(src_meta, cand_meta):
    return key_compatibility(src_meta['key'], cand_meta['key']) and bpm_compatibility(src_meta['bpm'], cand_meta['bpm'])
    

def estimate_mixability_by_nn(src_audio, cand_audio):
    with tf1.Session() as sess:
        sess.run(tf1.global_variables_initializer())
        net.saver.restore(sess, os.path.join(settings.MIXABLE_DIR, 'model', 'model'))
        batch     = Batch()
        batch.x1  = [read_audio(src_audio)]
        batch.x2  = [read_audio(cand_audio)]
        batch.y   = [[0, 0]]
        score     = net.calculate(sess, batch)[0][1]
        return score
        
    

        
        