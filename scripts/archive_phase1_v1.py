#!/usr/bin/env python3
"""Move phase-1 v1 outputs out of the way before the v2 rerun."""
from __future__ import print_function

import json
import os
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from util.phase1 import COMBOS, PHASE1_NS  # noqa: E402

DEST = os.path.join(ROOT, 'outputs', '_archive', 'phase1_v1')
NAMES = [name for name, _, _ in COMBOS] + ['phase1']


def main():
    os.chdir(ROOT)
    os.makedirs(DEST, exist_ok=True)
    note = os.path.join(DEST, 'NOTE.txt')
    with open(note, 'w') as f:
        f.write(
            'Phase 1 v1 (before the 8000-epoch rerun).\n'
            'n=1/2/10: 2000 epochs. n=100: 400 epochs. warmup=40, min_lr=0.\n'
            'n>=10 train loss had not reached the floor; cosine LR was ~0.\n'
        )
    recipe_src = os.path.join(DEST, 'phase1_recipe.v1.json')
    # Snapshot whatever is still in splits if we have not overwritten yet.
    live = os.path.join(ROOT, 'splits', 'phase1_recipe.json')
    if os.path.isfile(live) and not os.path.isfile(recipe_src):
        shutil.copy2(live, recipe_src)
    moved = []
    for name in NAMES:
        src = os.path.join(ROOT, 'outputs', name)
        dst = os.path.join(DEST, name)
        if not os.path.isdir(src):
            continue
        if os.path.exists(dst):
            print('already archived', dst)
            continue
        shutil.move(src, dst)
        moved.append(name)
        print('moved', src, '->', dst)
    # Leave shared VQGAN recon in outputs/_ref.
    print('archived', moved, 'to', DEST)
    return 0


if __name__ == '__main__':
    sys.exit(main())
