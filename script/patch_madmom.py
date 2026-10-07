"""Patch a known bug in the installed madmom package so it works with modern numpy.

madmom 0.16.1 (the latest PyPI release, from 2018) builds a numpy array from a
list of (path, log_prob) results with mismatched shapes. Older numpy silently
built an object array; numpy>=1.24 raises instead. Run this once after
installing requirements on a new machine:

    python script/patch_madmom.py
"""
import re
import numpy as np
import collections, collections.abc

for _alias, _builtin in (('int', int), ('float', float), ('bool', bool), ('object', object), ('str', str)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _builtin)

if not hasattr(collections, 'MutableSequence'):
    collections.MutableSequence = collections.abc.MutableSequence

import madmom.features.downbeats as downbeats

path = downbeats.__file__
src = open(path, encoding='utf-8').read()

old = 'best = np.argmax(np.asarray(results)[:, 1])'
new = 'best = np.argmax(np.asarray(results, dtype=object)[:, 1])'

if new in src:
    print(f'Already patched: {path}')
elif old in src:
    open(path, 'w', encoding='utf-8').write(src.replace(old, new))
    print(f'Patched: {path}')
else:
    raise SystemExit(
        f'Could not find the expected line in {path} - madmom internals '
        'may have changed; patch it manually or check djtransgan/process/beat.py.'
    )
