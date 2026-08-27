#!/usr/bin/env python3
"""Login-node checks that phase 0 files and metric definitions are complete."""
import ast
import json
import os
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from util.exp_data import (  # noqa: E402
    IMAGE_A,
    IMAGE_B,
    N_VALUES,
    default_num_images,
    default_source_root,
    default_splits_dir,
    generation_defaults,
    load_n_list,
    resolve_paths,
)
from util.exp_fid import fid_two_folders, frechet_distance  # noqa: E402
from util.exp_metrics import (  # noqa: E402
    REQUIRED_METRIC_KEYS,
    build_metrics,
    coverage_from_errors,
    nearest_token,
    write_grid,
    write_metrics_json,
)


def assert_true(cond, msg):
    if not cond:
        raise AssertionError(msg)


def check_lists():
    splits = default_splits_dir()
    source = default_source_root()
    lists = {}
    for n in N_VALUES:
        path = os.path.join(splits, 'n{}.txt'.format(n))
        assert_true(os.path.isfile(path), 'missing {}'.format(path))
        rels = load_n_list(n, splits)
        assert_true(len(rels) == n, '{} has {} lines, want {}'.format(path, len(rels), n))
        assert_true(len(set(rels)) == n, '{} has duplicates'.format(path))
        lists[n] = rels
        missing = [p for p in resolve_paths(rels, source) if not os.path.isfile(p)]
        assert_true(not missing, 'missing files for n={}: {}'.format(n, missing[:3]))
    assert_true(lists[1] == [IMAGE_A], 'n=1 must be A only')
    assert_true(lists[2] == [IMAGE_A, IMAGE_B], 'n=2 must be A then B')
    for bigger, smaller in ((2, 1), (10, 2), (100, 10), (1000, 100), (10000, 1000)):
        assert_true(
            lists[bigger][:smaller] == lists[smaller],
            'n={} is not a prefix of n={}'.format(smaller, bigger))
    manifest = os.path.join(splits, 'manifest.json')
    assert_true(os.path.isfile(manifest), 'missing manifest')
    with open(manifest) as f:
        man = json.load(f)
    assert_true(man['A']['path'] == IMAGE_A, 'manifest A')
    assert_true(man['B']['path'] == IMAGE_B, 'manifest B')
    assert_true(man.get('nested') is True, 'manifest nested')
    probe = os.path.join(splits, 'probe.txt')
    assert_true(os.path.isfile(probe), 'missing probe')
    return lists


def check_defaults():
    g = generation_defaults(2)
    assert_true(g['temperature'] == 6.0, 'default temperature')
    assert_true(g['num_iter'] == 20, 'default steps')
    assert_true(g['argmax'] is False, 'default is sample, not argmax')
    assert_true(default_num_images(1) == 1000, 'n=1 gens')
    assert_true(default_num_images(2) == 1000, 'n=2 gens')
    assert_true(default_num_images(10) == 1000, 'n=10 gens')
    assert_true(default_num_images(100) == 5000, 'n=100 gens')
    assert_true(default_num_images(1000) == 50000, 'n=1000 gens')
    assert_true(default_num_images(10000) == 50000, 'n=10000 gens')


def check_metrics_math():
    train = np.array([
        [0] * 256,
        [1] * 256,
    ], dtype=np.int32)
    gen = np.array([
        [0] * 256,
        [1] * 240 + [2] * 16,
        [3] * 256,
    ], dtype=np.int32)
    t2g, _ = nearest_token(train, gen)
    g2t, nn = nearest_token(gen, train)
    cov = coverage_from_errors(t2g)
    assert_true(t2g.tolist() == [0, 16], t2g)
    assert_true(cov['exact'] == 0.5, cov)
    assert_true(cov['approx'] == 1.0, cov)
    assert_true(nn.tolist() == [0, 1, 0] or nn[2] in (0, 1), nn)
    assert_true(g2t[0] == 0 and g2t[1] == 16, g2t)


def check_frechet():
    rng = np.random.RandomState(0)
    x = rng.randn(200, 8)
    y = x + 0.01
    mu1, s1 = x.mean(0), np.cov(x, rowvar=False)
    mu2, s2 = y.mean(0), np.cov(y, rowvar=False)
    d = frechet_distance(mu1, s1, mu2, s2)
    assert_true(d >= 0, d)
    d0 = frechet_distance(mu1, s1, mu1, s1)
    assert_true(d0 < 1e-6, d0)


def check_metrics_writer():
    tmp = tempfile.mkdtemp(prefix='phase0_metrics_')
    grid = os.path.join(tmp, 'grid.png')
    dummy = np.zeros((8, 32, 32, 3), dtype=np.uint8)
    dummy[:, :, :, 0] = 255
    write_grid(dummy, dummy, dummy, np.zeros(8, dtype=np.int32), grid, n_show=8)
    assert_true(os.path.isfile(grid), grid)
    metrics = build_metrics(
        n=2,
        exp_name='phase0_check',
        image_list='splits/n2.txt',
        train_rel_paths=[IMAGE_A, IMAGE_B],
        generation=generation_defaults(2),
        fid={
            'generated_vs_ref': 12.3,
            'vqgan_recon_vs_ref': 8.9,
            'generated_dir': 'outputs/phase0_check/n2/gen',
            'ref_dir': 'data/tiny-imagenet-val256',
            'recon_dir': 'outputs/phase0_check/n2/ref_vqgan_recon',
        },
        coverage={'exact': 0.0, 'approx': 0.0, 'approx_token_errors_max': 16, 'n_train': 2},
        match_error={'token_frac': 0.5, 'pixel_mse_vqgan': 0.1, 'n_generated': 1000},
        oneshot_vs_self={'token_acc': 0.0, 'probe': [IMAGE_A, IMAGE_B]},
        grid_path='outputs/phase0_check/n2/grid.png',
    )
    for k in REQUIRED_METRIC_KEYS:
        assert_true(k in metrics, k)
    dest = os.path.join(tmp, 'metrics.json')
    write_metrics_json(metrics, dest)
    with open(dest) as f:
        loaded = json.load(f)
    for k in ('temperature', 'num_iter', 'argmax', 'num_images'):
        assert_true(k in loaded['generation'], k)
    assert_true('generated_vs_ref' in loaded['fid'], 'gen fid')
    assert_true('vqgan_recon_vs_ref' in loaded['fid'], 'recon fid')
    example = os.path.join(default_splits_dir(), 'metrics.example.json')
    write_metrics_json(metrics, example)
    return dest, example


def check_same_fid_function():
    path = os.path.join(ROOT, 'eval_exp.py')
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    calls = []

    class V(ast.NodeVisitor):
        def visit_Call(self, node):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name == 'fid_two_folders':
                calls.append(node.lineno)
            self.generic_visit(node)

    V().visit(tree)
    assert_true(len(calls) >= 2, 'eval_exp.py must call fid_two_folders at least twice')
    src = open(path).read()
    assert_true('generated_vs_ref' in src and 'vqgan_recon_vs_ref' in src, 'both FID fields')
    return calls


def check_fid_smoke():
    """Two tiny folders through the same FID function (CPU, cached Inception)."""
    tmp = tempfile.mkdtemp(prefix='phase0_fid_')
    a = os.path.join(tmp, 'a')
    b = os.path.join(tmp, 'b')
    os.makedirs(a)
    os.makedirs(b)
    rng = np.random.RandomState(0)
    for i in range(8):
        Image.fromarray(rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(
            os.path.join(a, '{}.png'.format(i)))
        Image.fromarray(rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(
            os.path.join(b, '{}.png'.format(i)))
    device = 'cpu'
    same = fid_two_folders(a, a, device, batch_size=4, num_workers=0)
    score = fid_two_folders(a, b, device, batch_size=4, num_workers=0)
    assert_true(np.isfinite(same) and abs(same) < 1e-3, same)
    assert_true(np.isfinite(score), score)
    return score, same


def main():
    os.chdir(ROOT)
    print('check lists')
    check_lists()
    print('check defaults')
    check_defaults()
    print('check coverage/match')
    check_metrics_math()
    print('check frechet')
    check_frechet()
    print('check metrics writer')
    dest, example = check_metrics_writer()
    print(' wrote', dest)
    print(' wrote', example)
    print('check same FID function')
    lines = check_same_fid_function()
    print(' fid_two_folders calls at lines', lines)
    if '--fid-smoke' in sys.argv:
        print('check FID smoke on CPU')
        score, same = check_fid_smoke()
        print(' fid(a,b)={} fid(a,a)={}'.format(score, same))
    print('phase0 checks passed')


if __name__ == '__main__':
    main()
