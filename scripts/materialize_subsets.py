#!/usr/bin/env python3
"""Symlink frozen lists into ImageFolder trees at data/subsets/n<N>/."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from util.exp_data import (  # noqa: E402
    N_VALUES,
    default_source_root,
    default_splits_dir,
    load_n_list,
    resolve_paths,
)


def materialize_one(n, splits_dir, source_root, dest_root):
    rels = load_n_list(n, splits_dir)
    dest = os.path.join(dest_root, 'n{}'.format(n), 'train')
    os.makedirs(dest, exist_ok=True)
    for i, (rel, src) in enumerate(zip(rels, resolve_paths(rels, source_root))):
        if not os.path.isfile(src):
            raise FileNotFoundError(src)
        class_id = 'id{}'.format(i)
        class_dir = os.path.join(dest, class_id)
        os.makedirs(class_dir, exist_ok=True)
        ext = os.path.splitext(src)[1] or '.JPEG'
        link = os.path.join(class_dir, 'img{}'.format(ext))
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(os.path.abspath(src), link)
    return dest


def main():
    splits_dir = default_splits_dir()
    source_root = default_source_root()
    dest_root = os.path.join(ROOT, 'data', 'subsets')
    for n in N_VALUES:
        dest = materialize_one(n, splits_dir, source_root, dest_root)
        print(dest, 'n={}'.format(n))


if __name__ == '__main__':
    main()
