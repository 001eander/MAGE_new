"""Coverage, nearest-neighbor match error, grid, and metrics.json schema."""
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw

from util.exp_data import (
    APPROX_COVER_MAX_ERRORS,
    GRID_N,
    IMAGE_A,
    IMAGE_B,
    generation_defaults,
    source_info,
)

REQUIRED_METRIC_KEYS = (
    'n',
    'exp_name',
    'image_list',
    'A',
    'B',
    'source',
    'generation',
    'fid',
    'coverage',
    'match_error',
    'oneshot_vs_self',
    'grid',
)


def nearest_token(query, gallery, q_batch=32, g_batch=4096):
    """Hamming nearest neighbor over 256 codebook indices.

    Returns (min_errors [Nq], nn_index [Nq]) where errors are counts in 0..256.
    """
    query = np.asarray(query, dtype=np.int32)
    gallery = np.asarray(gallery, dtype=np.int32)
    if query.ndim != 2 or gallery.ndim != 2 or query.shape[1] != 256 or gallery.shape[1] != 256:
        raise ValueError('tokens must be [N, 256], got {} and {}'.format(
            query.shape, gallery.shape))
    nq = query.shape[0]
    min_err = np.full(nq, 256, dtype=np.int32)
    nn = np.zeros(nq, dtype=np.int32)
    if gallery.shape[0] == 0 or nq == 0:
        return min_err, nn
    for qi in range(0, nq, q_batch):
        q = query[qi:qi + q_batch]
        local_min = np.full(len(q), 256, dtype=np.int32)
        local_nn = np.zeros(len(q), dtype=np.int32)
        for gi in range(0, gallery.shape[0], g_batch):
            g = gallery[gi:gi + g_batch]
            err = (q[:, None, :] != g[None, :, :]).sum(axis=-1)
            idx = err.argmin(axis=1)
            val = err[np.arange(len(q)), idx]
            better = val < local_min
            local_min[better] = val[better]
            local_nn[better] = idx[better] + gi
        min_err[qi:qi + len(q)] = local_min
        nn[qi:qi + len(q)] = local_nn
    return min_err, nn


def coverage_from_errors(train_to_gen_errors, approx_max=APPROX_COVER_MAX_ERRORS):
    err = np.asarray(train_to_gen_errors)
    n = max(int(err.size), 1)
    return {
        'exact': float((err == 0).sum()) / n,
        'approx': float((err <= approx_max).sum()) / n,
        'approx_token_errors_max': int(approx_max),
        'n_train': int(err.size),
    }


def token_match_frac(gen_to_train_errors):
    err = np.asarray(gen_to_train_errors, dtype=np.float64)
    if err.size == 0:
        return 0.0
    return float((err / 256.0).mean())


def _pairwise_cosine_distance(pred_e, gt_e, eps=1e-8):
    pred_n = np.linalg.norm(pred_e, axis=1)
    gt_n = np.linalg.norm(gt_e, axis=1)
    dot = np.sum(pred_e * gt_e, axis=1)
    sim = dot / np.maximum(pred_n * gt_n, eps)
    return 1.0 - sim


def _pairwise_euclidean(pred_e, gt_e):
    return np.linalg.norm(pred_e - gt_e, axis=1)


def wrong_token_embed_stats(pred, gt, embeddings, seed=0, vocab_size=1024):
    """Distances between wrong argmax ids and the correct id in word-embedding space.

    ``embeddings`` is [vocab, dim] for the frozen token embeddings (first 1024
    codebook rows). Random baseline draws a different wrong id for each error.
    """
    pred = np.asarray(pred).reshape(-1)
    gt = np.asarray(gt).reshape(-1)
    emb = np.asarray(embeddings, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError('pred/gt shape {} vs {}'.format(pred.shape, gt.shape))
    if emb.ndim != 2 or emb.shape[0] < vocab_size:
        raise ValueError('embeddings must be [>=1024, dim], got {}'.format(emb.shape))
    n_total = int(pred.size)
    wrong = pred != gt
    n_wrong = int(wrong.sum())
    out = {
        'n_wrong': n_wrong,
        'n_total': n_total,
        'frac_wrong': float(n_wrong) / float(max(n_total, 1)),
        'wrong_cosine_distance': None,
        'wrong_euclidean': None,
        'random_wrong_cosine_distance': None,
        'random_wrong_euclidean': None,
        'note': (
            'Mean distance in token-embedding space between the wrong argmax '
            'id and the correct id; random baseline is a uniformly drawn id '
            'that is not the correct one.'
        ),
    }
    if n_wrong == 0:
        out['note'] = 'no wrong tokens; distances undefined'
        return out
    pred_e = emb[pred[wrong]]
    gt_e = emb[gt[wrong]]
    out['wrong_cosine_distance'] = float(_pairwise_cosine_distance(pred_e, gt_e).mean())
    out['wrong_euclidean'] = float(_pairwise_euclidean(pred_e, gt_e).mean())
    rng = np.random.RandomState(seed)
    rand = rng.randint(0, vocab_size, size=n_wrong)
    collide = rand == gt[wrong]
    rand[collide] = (rand[collide] + 1) % vocab_size
    rand_e = emb[rand]
    out['random_wrong_cosine_distance'] = float(
        _pairwise_cosine_distance(rand_e, gt_e).mean())
    out['random_wrong_euclidean'] = float(_pairwise_euclidean(rand_e, gt_e).mean())
    return out


def pixel_mse_uint8(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean((a - b) ** 2) / (255.0 ** 2))


def to_uint8_image(x):
    """Accept HWC uint8, or CHW/BCHW float in [0, 1]."""
    if hasattr(x, 'detach'):
        x = x.detach().cpu()
    arr = np.asarray(x)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(np.round(arr.astype(np.float64) * 255.0), 0, 255).astype(np.uint8)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return arr


def write_grid(gen_images, train_images, train_recons, nn_index, path, n_show=GRID_N):
    """Rows: generated | nearest train | that train's VQGAN recon."""
    n_show = min(int(n_show), len(gen_images), max(len(train_images), 0))
    if n_show <= 0:
        raise ValueError('need at least one generated and one train image for the grid')
    panels = []
    labels = []
    for i in range(n_show):
        j = int(nn_index[i]) if i < len(nn_index) else 0
        g = to_uint8_image(gen_images[i])
        t = to_uint8_image(train_images[j])
        r = to_uint8_image(train_recons[j])
        panels.append((g, t, r))
        labels.append(('gen {}'.format(i), 'train {}'.format(j), 'vqgan {}'.format(j)))
    h, w = panels[0][0].shape[:2]
    bar = 22
    canvas = np.full((n_show * (h + bar), 3 * w, 3), 255, dtype=np.uint8)
    for i, (g, t, r) in enumerate(panels):
        y = i * (h + bar) + bar
        canvas[y:y + h, 0:w] = g
        canvas[y:y + h, w:2 * w] = t
        canvas[y:y + h, 2 * w:3 * w] = r
    im = Image.fromarray(canvas)
    draw = ImageDraw.Draw(im)
    for i, labs in enumerate(labels):
        for k, lab in enumerate(labs):
            draw.text((k * w + 6, i * (h + bar) + 4), lab, fill=(0, 0, 0))
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    im.save(path)
    return path


def json_safe(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, np.floating):
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def build_metrics(
        n,
        exp_name,
        image_list,
        train_rel_paths,
        generation,
        fid,
        coverage,
        match_error,
        oneshot_vs_self,
        grid_path,
        ref_dir=None,
        gen_dir=None,
        wrong_token_embed=None,
        extra=None):
    gen = dict(generation_defaults(n))
    gen.update(generation or {})
    metrics = {
        'n': int(n),
        'exp_name': exp_name,
        'image_list': image_list,
        'n_train': len(train_rel_paths),
        'train_paths': list(train_rel_paths),
        'A': IMAGE_A,
        'B': IMAGE_B if int(n) >= 2 else None,
        'source': source_info(),
        'generation': {
            'temperature': json_safe(gen['temperature']),
            'num_iter': int(gen['num_iter']),
            'argmax': bool(gen['argmax']),
            'num_images': int(gen['num_images']) if gen.get('num_images') is not None else None,
            'seed': int(gen.get('seed', 0)),
            'schedule': gen.get('schedule', 'cosine'),
            'uncover_per_step': gen.get('uncover_per_step'),
            'remask': bool(gen.get('remask', True)),
            'repeats_allowed': True,
            'note': gen.get('note') or (
                'Compare settings with these defaults unless the comparison '
                'is about temperature / steps / argmax / count / schedule.'
            ),
        },
        'fid': {
            'generated_vs_ref': json_safe((fid or {}).get('generated_vs_ref')),
            'vqgan_recon_vs_ref': json_safe((fid or {}).get('vqgan_recon_vs_ref')),
            'tokens_exact_equals_vqgan_recon': True,
            'generated_dir': (fid or {}).get('generated_dir', gen_dir),
            'ref_dir': (fid or {}).get('ref_dir', ref_dir),
            'recon_dir': (fid or {}).get('recon_dir'),
            'implementation': (fid or {}).get(
                'implementation',
                'pytorch-fid calculate_fid_given_paths (same as LGANs-TT realdata.py); '
                'same function for generated and VQGAN-recon FID',
            ),
        },
        'coverage': coverage,
        'match_error': match_error,
        'oneshot_vs_self': oneshot_vs_self,
        'wrong_token_embed': wrong_token_embed,
        'grid': grid_path,
    }
    if extra:
        metrics.update(extra)
    return metrics


def write_metrics_json(metrics, path):
    missing = [k for k in REQUIRED_METRIC_KEYS if k not in metrics]
    if missing:
        raise ValueError('metrics.json missing keys: {}'.format(missing))
    gen = metrics['generation']
    for k in ('temperature', 'num_iter', 'argmax', 'num_images'):
        if k not in gen:
            raise ValueError('generation missing {}'.format(k))
    fid = metrics['fid']
    if 'generated_vs_ref' not in fid or 'vqgan_recon_vs_ref' not in fid:
        raise ValueError('fid must include generated_vs_ref and vqgan_recon_vs_ref')
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
        f.write('\n')
    return path
