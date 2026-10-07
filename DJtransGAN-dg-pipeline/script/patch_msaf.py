"""Patch a known bug in the installed msaf package so it works with modern scipy.

msaf 0.1.80's bundled pymf module does `from scipy import inf`, a re-export
scipy dropped years ago. Run this once after installing requirements:

    python script/patch_msaf.py
"""
import importlib.util
import os

# Can't `import msaf` here - that's exactly the broken import chain this
# script is meant to fix. Locate the package on disk instead.
msaf_spec = importlib.util.find_spec('msaf')
path = os.path.join(os.path.dirname(msaf_spec.origin), 'pymf', 'sivm_search.py')
src = open(path, encoding='utf-8').read()

old = 'from scipy import inf'
new = 'from numpy import inf'

if new in src:
    print(f'Already patched: {path}')
elif old in src:
    open(path, 'w', encoding='utf-8').write(src.replace(old, new))
    print(f'Patched: {path}')
else:
    raise SystemExit(
        f'Could not find the expected line in {path} - msaf internals '
        'may have changed; patch it manually.'
    )
