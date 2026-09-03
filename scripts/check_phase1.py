#!/usr/bin/env python3
"""Login-node checks for phase 1 (four-cell slice on each n)."""
import ast
import json
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from util.exp_metrics import wrong_token_embed_stats  # noqa: E402
from util.phase1 import (  # noqa: E402
    COMBOS,
    PHASE1_NS,
    RECIPE,
    job_from_index,
    last_logged_epoch,
    metrics_path,
    train_finished,
    write_recipe_json,
)


def assert_true(cond, msg):
    if not cond:
        raise AssertionError(msg)


def source_has(path, needle):
    with open(os.path.join(ROOT, path)) as f:
        return needle in f.read()


def parse_has_arg(path, dest):
    with open(os.path.join(ROOT, path)) as f:
        tree = ast.parse(f.read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in getattr(node, 'keywords', []):
                if kw.arg == 'dest' and isinstance(kw.value, ast.Str) and kw.value.s == dest:
                    return True
            for arg in node.args:
                if isinstance(arg, ast.Str) and arg.s == '--' + dest:
                    return True
                if isinstance(arg, ast.Constant) and arg.value == '--' + dest:
                    return True
    return False


def check_code():
    assert_true(source_has('models_mage.py', 'linear_head'), 'models_mage linear_head')
    assert_true(source_has('models_mage.py', 'token_classifier'), 'token_classifier')
    assert_true(parse_has_arg('main_pretrain.py', 'linear_head'), 'main_pretrain --linear_head')
    assert_true(source_has('main_pretrain.py', 'linear_head=args.linear_head'),
                'main_pretrain passes linear_head')
    assert_true(source_has('main_pretrain.py', 'reused cached tokens'),
                'main_pretrain reuses cached_train_tokens.pt')
    assert_true(source_has('eval_exp.py', 'wrong_token_embed_stats'), 'eval wrong-token distances')
    assert_true(source_has('eval_exp.py', 'infer_linear_head'), 'eval infers linear_head')
    assert_true(source_has('eval_exp.py', 'generate_batch_uncover1'), 'uncover1 sampler')
    assert_true(parse_has_arg('eval_exp.py', 'schedule'), 'eval --schedule')
    assert_true(os.path.isfile(os.path.join(ROOT, 'scripts', 'phase1_uncover1_one.sh')),
                'uncover1 eval script')
    script = os.path.join(ROOT, 'scripts', 'phase1_train_eval.sh')
    assert_true(os.path.isfile(script), script)
    with open(script) as f:
        text = f.read()
    assert_true('#BSUB -J phase1[1-16]' in text, 'job array 1-16')
    assert_true('gpu11||hname==gpu12' in text, 'V100 host select')
    assert_true('A100' not in text and 'gpu13' not in text, 'no A100/L20')
    pack = os.path.join(ROOT, 'scripts', 'phase1_pack.sh')
    assert_true(os.path.isfile(pack), pack)
    with open(pack) as f:
        pack_text = f.read()
    assert_true('CUDA_VISIBLE_DEVICES' in pack_text, 'pack per-gpu workers')
    assert_true('GPU_IDS' in pack_text, 'pack uses allocated physical GPU ids')
    assert_true('gpu13' not in pack_text, 'pack stays on V100')
    cell = os.path.join(ROOT, 'scripts', 'phase1_cell.sh')
    with open(cell) as f:
        cell_text = f.read()
    assert_true('--cache_tokens' in cell_text, 'cell caches tokens')
    assert_true('--min_lr' in cell_text, 'cell passes min_lr')
    assert_true('--profile' in cell_text, 'cell profiles')
    assert_true('--resume' in cell_text, 'cell resumes checkpoint-last if present')


def check_recipe():
    path = write_recipe_json(os.path.join(ROOT, 'splits', 'phase1_recipe.json'))
    with open(path) as f:
        payload = json.load(f)
    assert_true(payload['n'] == list(PHASE1_NS), payload['n'])
    assert_true(len(payload['combos']) == 4, payload['combos'])
    heads = {(c['linear_head'], c['label_smoothing']) for c in payload['combos']}
    assert_true(heads == {(False, 0.1), (False, 0.0), (True, 0.1), (True, 0.0)}, heads)
    # Only those two knobs differ; shared fields stay put.
    assert_true(RECIPE['mask_ratio_min'] == 0.5, 'paper mask min')
    assert_true(RECIPE['no_aug'] is True, 'no_aug')
    assert_true(RECIPE['model'] == 'mage_vit_base_patch16', RECIPE['model'])
    assert_true(RECIPE['epochs_by_n']['1'] == 8000, RECIPE['epochs_by_n'])
    assert_true(RECIPE['epochs_by_n']['100'] == 8000, 'equal visits/image')
    assert_true(RECIPE['epochs_by_n']['1000'] == 8000, 'n=1000 epochs')
    assert_true(RECIPE['epochs_by_n']['10000'] == 8000, 'n=10000 epochs')
    assert_true(RECIPE['batch_size_by_n']['1000'] == 50, 'n=1000 batch')
    assert_true(RECIPE['batch_size_by_n']['10000'] == 100, 'n=10000 batch')
    assert_true(RECIPE['min_lr'] == 1.0e-6, RECIPE['min_lr'])
    assert_true(RECIPE['warmup_epochs'] == 200, RECIPE['warmup_epochs'])
    assert_true(RECIPE['cache_tokens'] is True, 'cache_tokens')
    n, name, lh, ls = job_from_index(1)
    assert_true((n, name, lh, ls) == (1, 'dot_smooth', False, 0.1), (n, name, lh, ls))
    n, name, lh, ls = job_from_index(16)
    assert_true((n, name, lh, ls) == (100, 'lin_nosmooth', True, 0.0), (n, name, lh, ls))
    n, name, lh, ls = job_from_index(17)
    assert_true((n, name, lh, ls) == (1000, 'dot_smooth', False, 0.1), (n, name, lh, ls))
    n, name, lh, ls = job_from_index(24)
    assert_true((n, name, lh, ls) == (10000, 'lin_nosmooth', True, 0.0), (n, name, lh, ls))
    tmp = tempfile.mkdtemp()
    try:
        cell = os.path.join(tmp, 'outputs', 'lin_nosmooth', 'n1000')
        os.makedirs(cell)
        with open(os.path.join(cell, 'log.txt'), 'w') as f:
            f.write('{"epoch": 199, "train_loss": 6.0}\n')
        assert_true(last_logged_epoch('lin_nosmooth', 1000, root=tmp) == 199, 'mid log')
        assert_true(train_finished(20, root=tmp) is False, 'mid-run is not finished')
        with open(os.path.join(cell, 'log.txt'), 'a') as f:
            f.write('{"epoch": 7999, "train_loss": 1.0}\n')
        assert_true(train_finished(20, root=tmp) is True, 'last epoch is finished')
    finally:
        shutil.rmtree(tmp)
    note = os.path.join(ROOT, 'splits', 'phase1_star50000.txt')
    if not os.path.isfile(note):
        with open(note, 'w') as f:
            f.write(RECIPE['star50000'] + '\n')
    with open(note) as f:
        text = f.read().strip()
    assert_true('not run' in text, text)


def check_wrong_token_math():
    rng = np.random.RandomState(0)
    emb = rng.randn(1024, 8).astype(np.float64)
    gt = np.array([[3, 3, 3, 3]], dtype=np.int64)
    pred = np.array([[3, 3, 3, 3]], dtype=np.int64)
    stats = wrong_token_embed_stats(pred, gt, emb, seed=0)
    assert_true(stats['n_wrong'] == 0, stats)
    assert_true(stats['wrong_cosine_distance'] is None, stats)
    pred = np.array([[3, 7, 3, 7]], dtype=np.int64)
    stats = wrong_token_embed_stats(pred, gt, emb, seed=0)
    assert_true(stats['n_wrong'] == 2, stats)
    assert_true(stats['wrong_cosine_distance'] is not None, stats)
    assert_true(stats['random_wrong_cosine_distance'] is not None, stats)
    # Distance to the correct row must be 0 if we "mispredict" the same id.
    same = wrong_token_embed_stats(gt, gt, emb)
    assert_true(same['n_wrong'] == 0, same)


def check_artifacts(require_complete):
    missing = []
    incomplete = []
    for n in PHASE1_NS:
        for name, _, _ in COMBOS:
            path = os.path.join(ROOT, metrics_path(name, n))
            if not os.path.isfile(path):
                missing.append(path)
                continue
            with open(path) as f:
                m = json.load(f)
            if m.get('oneshot_vs_self') is None:
                incomplete.append(path + ':oneshot')
            if m.get('fid', {}).get('generated_vs_ref') is None:
                incomplete.append(path + ':fid')
            if m.get('coverage') is None or m.get('match_error') is None:
                incomplete.append(path + ':coverage/match')
            if not m.get('grid') or not os.path.isfile(os.path.join(ROOT, m['grid'])):
                incomplete.append(path + ':grid')
            if m.get('wrong_token_embed') is None:
                incomplete.append(path + ':wrong_token_embed')
            table = os.path.join(ROOT, 'outputs', 'phase1', 'n{}'.format(n), 'table.md')
            if not os.path.isfile(table):
                missing.append(table)
    print('metrics missing', len(missing))
    print('metrics incomplete', len(incomplete))
    if require_complete:
        assert_true(not missing and not incomplete, 'missing={} incomplete={}'.format(
            missing[:5], incomplete[:5]))
    return missing, incomplete


def main():
    os.chdir(ROOT)
    print('check code')
    check_code()
    print('check recipe')
    check_recipe()
    print('check wrong-token math')
    check_wrong_token_math()
    require = '--require-artifacts' in sys.argv
    print('check artifacts (require={})'.format(require))
    missing, incomplete = check_artifacts(require)
    for path in missing[:8]:
        print('  missing', path)
    for path in incomplete[:8]:
        print('  incomplete', path)
    print('phase1 code/recipe checks passed')
    if require:
        print('phase1 artifacts passed for n={}'.format(list(PHASE1_NS)))
    elif missing or incomplete:
        print('artifacts not complete yet (expected until jobs finish)')


if __name__ == '__main__':
    main()
