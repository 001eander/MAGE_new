#!/usr/bin/env python3
"""Write frozen nested Tiny-ImageNet lists into splits/.

n=1 is image A. n=2 is A then B (half-A-half-B pair). Larger n is a prefix
of the same ordered list. Remaining images after A,B are shuffled with
SPLIT_SEED so the selection is reproducible.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from util.exp_data import (  # noqa: E402
    IMAGE_A,
    IMAGE_B,
    N_VALUES,
    SPLIT_SEED,
    default_num_images,
    default_source_root,
    default_splits_dir,
    generation_defaults,
    source_info,
)

IMAGE_EXTS = ('.jpeg', '.jpg', '.png')


def collect_train_relpaths(source_root):
    train = os.path.join(source_root, 'train')
    paths = []
    for dirpath, dirnames, filenames in os.walk(train):
        dirnames.sort()
        for name in sorted(filenames):
            ext = os.path.splitext(name)[1].lower()
            if ext in IMAGE_EXTS:
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, source_root).replace('\\', '/')
                paths.append(rel)
    return paths


def build_nested_order(all_rel):
    have = set(all_rel)
    if IMAGE_A not in have:
        raise FileNotFoundError('A missing: {}'.format(IMAGE_A))
    if IMAGE_B not in have:
        raise FileNotFoundError('B missing: {}'.format(IMAGE_B))
    rest = [p for p in all_rel if p not in (IMAGE_A, IMAGE_B)]
    rest.sort()
    rng = np.random.RandomState(SPLIT_SEED)
    rng.shuffle(rest)
    return [IMAGE_A, IMAGE_B] + rest


def write_lines(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for line in lines:
            f.write(line + '\n')


def main():
    source_root = default_source_root()
    splits_dir = default_splits_dir()
    all_rel = collect_train_relpaths(source_root)
    if len(all_rel) < max(N_VALUES):
        raise RuntimeError(
            'need at least {} train images, found {} in {}'.format(
                max(N_VALUES), len(all_rel), source_root))
    ordered = build_nested_order(all_rel)
    os.makedirs(splits_dir, exist_ok=True)
    for n in N_VALUES:
        # n=1 is A only; n>=2 starts with A,B
        chosen = ordered[:n] if n >= 2 else [IMAGE_A]
        write_lines(os.path.join(splits_dir, 'n{}.txt'.format(n)), chosen)
    probe = ordered[:8]
    write_lines(os.path.join(splits_dir, 'probe.txt'), probe)
    words = {}
    words_path = os.path.join(source_root, 'words.txt')
    if os.path.isfile(words_path):
        with open(words_path) as f:
            for line in f:
                wid, name = line.rstrip('\n').split('\t', 1)
                words[wid] = name
    manifest = {
        'source': source_info(),
        'source_root': os.path.relpath(source_root, ROOT),
        'splits_dir': os.path.relpath(splits_dir, ROOT),
        'seed': SPLIT_SEED,
        'n_values': list(N_VALUES),
        'nested': True,
        'A': {
            'id': 'A',
            'path': IMAGE_A,
            'wnid': IMAGE_A.split('/')[1],
            'name': words.get(IMAGE_A.split('/')[1], ''),
            'used_for': ['n=1', 'n>=2 prefix', 'half A half B image A'],
        },
        'B': {
            'id': 'B',
            'path': IMAGE_B,
            'wnid': IMAGE_B.split('/')[1],
            'name': words.get(IMAGE_B.split('/')[1], ''),
            'used_for': ['n=2 second image', 'n>=2 prefix', 'half A half B image B'],
        },
        'probe': {
            'path': 'probe.txt',
            'note': 'A, B, plus the next 6 of n=10. Large-n oneshot-vs-self uses this set.',
        },
        'generation_defaults': generation_defaults(2),
        'num_images_by_n': {str(n): default_num_images(n) for n in N_VALUES},
        'imagenet256': (
            'Not prepared. ImageNet-256 is required only when comparing FID '
            'to the paper (especially n=10000 and *50000). Tiny FID is not that comparison.'
        ),
    }
    with open(os.path.join(splits_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
        f.write('\n')
    print('wrote', splits_dir)
    for n in N_VALUES:
        print('n={}: {} images, first={}'.format(
            n, n, (ordered[:n] if n >= 2 else [IMAGE_A])[0]))


if __name__ == '__main__':
    main()
