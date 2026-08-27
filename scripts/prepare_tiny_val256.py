#!/usr/bin/env python3
"""Upsample Tiny-ImageNet val 64x64 -> 256x256 for the shared Tiny FID reference.

FID on this folder is not comparable to paper ImageNet-256 numbers.
"""
import os
import sys

from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from util.exp_data import TRAIN_EVAL_SIZE, default_source_root  # noqa: E402

IMAGE_EXTS = ('.jpeg', '.jpg', '.png')


def collect_val_images(source_root):
    val_images = os.path.join(source_root, 'val', 'images')
    if not os.path.isdir(val_images):
        raise FileNotFoundError(val_images)
    paths = []
    for name in sorted(os.listdir(val_images)):
        if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
            paths.append(os.path.join(val_images, name))
    return paths


def main():
    source_root = default_source_root()
    out_dir = os.path.join(ROOT, 'data', 'tiny-imagenet-val256')
    os.makedirs(out_dir, exist_ok=True)
    paths = collect_val_images(source_root)
    size = TRAIN_EVAL_SIZE
    for i, src in enumerate(paths):
        img = Image.open(src).convert('RGB')
        img = img.resize((size, size), Image.BICUBIC)
        dest = os.path.join(out_dir, '{}.png'.format(str(i).zfill(5)))
        img.save(dest)
        if (i + 1) % 1000 == 0:
            print('resized', i + 1, '/', len(paths), flush=True)
    print('wrote', len(paths), 'images to', out_dir, flush=True)


if __name__ == '__main__':
    main()
