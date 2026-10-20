"""Phase-2 eval: half-A / half-B stitching in token space.

Take two training images A and B. Keep a fixed set of positions from A, mask the
rest, and generate. The ideal answer is B on the masked positions. Report how
often the model matches B (composition), how often it matches A (copying), and
the pixel distance to each.

Tokens are a 16x16 row-major grid, the same layout decode_tokens reshapes.

Usage:
  ~/miniforge3/envs/mage/bin/python eval_half_ab.py \
      --n 10 --exp_name lin_nosmooth --pattern leftright \
      --schedules oneshot,uncover1
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

import eval_exp
from util.exp_data import (
    DEFAULT_ARGMAX,
    DEFAULT_GEN_SEED,
    DEFAULT_NUM_ITER,
    DEFAULT_TEMPERATURE,
    default_source_root,
    default_splits_dir,
    load_n_list,
    train_eval_transform,
)
from util.exp_metrics import pixel_mse_uint8, to_uint8_image

PATTERNS = ('leftright', 'checker', 'random')
SCHEDULES = ('oneshot', 'uncover1', 'cosine')
GRID_SIZE = 16


def get_args_parser():
    parser = argparse.ArgumentParser(
        description='Phase 2: half-A half-B stitching eval')
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--splits_dir', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--vqgan_ckpt', type=str, default='vqgan_jax_strongaug.ckpt')
    parser.add_argument('--model', type=str, default='mage_vit_base_patch16')
    parser.add_argument('--pattern', choices=PATTERNS, default='leftright')
    parser.add_argument('--schedules', type=str, default='oneshot,uncover1')
    parser.add_argument('--max_pairs', type=int, default=16)
    parser.add_argument('--num_iter', type=int, default=DEFAULT_NUM_ITER)
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument('--argmax', action='store_true', default=DEFAULT_ARGMAX)
    parser.add_argument('--sample', action='store_false', dest='argmax')
    parser.add_argument('--gen_seed', type=int, default=DEFAULT_GEN_SEED)
    parser.add_argument('--checker_seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--no_pixel', action='store_true',
                        help='skip VQGAN decode and pixel metrics')
    parser.add_argument('--device', type=str, default='cuda')
    return parser


def default_ckpt(exp_name, n):
    return os.path.join('outputs', exp_name, 'n{}'.format(n), 'checkpoint-last.pth')


def make_pairs(n, max_pairs):
    """Deterministic adjacent pairs, so n=2 gives exactly A and B."""
    count = max(1, min(max_pairs, n // 2))
    pairs = []
    for k in range(count):
        i = (2 * k) % n
        j = (2 * k + 1) % n
        if i == j:
            continue
        pairs.append((i, j))
    return pairs


def known_mask(pattern, device, seed=0):
    """True = visible. Returns [256] bool and a text description."""
    idx = np.arange(GRID_SIZE * GRID_SIZE).reshape(GRID_SIZE, GRID_SIZE)
    rows, cols = np.divmod(idx, GRID_SIZE)
    if pattern == 'leftright':
        mask = cols < GRID_SIZE // 2
        desc = 'left half visible (cols 0-7), right half masked'
    elif pattern == 'checker':
        mask = (rows + cols) % 2 == 0
        desc = 'checkerboard visible ((row+col)%2==0), 128 positions'
    else:
        rng = np.random.RandomState(seed)
        flat = np.zeros(GRID_SIZE * GRID_SIZE, dtype=bool)
        flat[rng.choice(GRID_SIZE * GRID_SIZE, GRID_SIZE * GRID_SIZE // 2,
                        replace=False)] = True
        mask = flat
        desc = 'random 128 visible, seed={}'.format(seed)
    return torch.from_numpy(mask.reshape(-1)).to(device), desc


def sample_tokens(model, logits, argmax):
    if argmax:
        return logits.argmax(dim=-1)
    return torch.distributions.categorical.Categorical(logits=logits).sample()


def generate_oneshot(model, known_tokens, visible, seed, argmax, device):
    """One forward pass: every masked position is sampled at once."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    mask_id = model.mask_token_label
    token_indices = torch.where(visible, known_tokens, mask_id)
    with torch.no_grad():
        logits = eval_exp.mage_codebook_logits(model, token_indices.long())
        sampled = sample_tokens(model, logits, argmax)
    return torch.where(visible, known_tokens, sampled)


def generate_uncover1(model, known_tokens, visible, seed, argmax, device):
    """Commit one most-confident masked position per step; never remask."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    mask_id = model.mask_token_label
    token_indices = torch.where(visible, known_tokens, mask_id).long()
    # Confidence of a position only exists once it is committed, so the first
    # step has to sample all masked positions to rank them.
    with torch.no_grad():
        logits = eval_exp.mage_codebook_logits(model, token_indices)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        sampled = sample_tokens(model, logits, argmax)
        confidence = torch.gather(
            probs, -1, sampled.unsqueeze(-1)).squeeze(-1)
    unknown = ~visible
    confidence = torch.where(unknown, confidence, torch.full_like(confidence, -1.0))
    token_indices = torch.where(visible, known_tokens, sampled).long()
    steps = int(unknown.sum(dim=1).max().item())
    for _ in range(steps - 1):
        open_mask = token_indices == mask_id
        if not bool(open_mask.any()):
            break
        with torch.no_grad():
            logits = eval_exp.mage_codebook_logits(model, token_indices)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            sampled = sample_tokens(model, logits, argmax)
            new_conf = torch.gather(
                probs, -1, sampled.unsqueeze(-1)).squeeze(-1)
        # Fill every still-open position with its current sample, then hide all
        # but the most confident one again.
        filled = torch.where(open_mask, sampled, token_indices)
        conf = torch.where(open_mask, new_conf,
                           torch.full_like(new_conf, -1.0))
        best = conf.argmax(dim=1)
        keep = torch.zeros_like(open_mask)
        keep[torch.arange(keep.size(0), device=device), best] = True
        token_indices = torch.where(keep | ~open_mask, filled, mask_id)
    return token_indices


def generate_cosine(model, known_tokens, visible, seed, num_iter, temperature,
                    argmax, device):
    """Paper cosine-remask sampler restricted to the masked positions."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    mask_id = model.mask_token_label
    token_indices = torch.where(visible, known_tokens, mask_id).long()
    unknown_total = (~visible).sum(dim=1).float()
    sampled = token_indices
    for step in range(num_iter):
        with torch.no_grad():
            logits = eval_exp.mage_codebook_logits(model, token_indices)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            sampled = sample_tokens(model, logits, argmax)
        unknown = token_indices == mask_id
        filled = torch.where(unknown, sampled, token_indices)
        conf = torch.gather(probs, -1, sampled.unsqueeze(-1)).squeeze(-1)
        conf = torch.where(unknown, conf, torch.full_like(conf, float('inf')))
        ratio = float(step + 1) / num_iter
        mask_ratio = math.cos(math.pi / 2.0 * ratio)
        for i in range(token_indices.size(0)):
            open_i = int(unknown[i].sum().item())
            if open_i <= 1:
                continue
            mask_len = int(math.floor(float(unknown_total[i]) * mask_ratio))
            mask_len = max(1, min(open_i - 1, mask_len))
            hide = eval_exp.mask_by_random_topk(
                mask_len, conf[i].unsqueeze(0), temperature * (1 - ratio))
            filled[i] = torch.where(hide[0], mask_id, filled[i])
        token_indices = filled
    return torch.where(token_indices == mask_id, sampled, token_indices)


def generate(model, schedule, known_tokens, visible, args):
    if schedule == 'oneshot':
        return generate_oneshot(model, known_tokens, visible, args.gen_seed,
                                args.argmax, args.device)
    if schedule == 'uncover1':
        return generate_uncover1(model, known_tokens, visible, args.gen_seed,
                                 args.argmax, args.device)
    return generate_cosine(model, known_tokens, visible, args.gen_seed,
                           args.num_iter, args.temperature, args.argmax,
                           args.device)


def pair_metrics(gen, ideal, a_tokens, visible):
    """Token match counts on the masked positions."""
    masked = ~visible
    n_masked = int(masked.sum().item())
    match_ideal = int((gen[masked] == ideal[masked]).sum().item())
    match_a = int((gen[masked] == a_tokens[masked]).sum().item())
    full = int((gen == ideal).sum().item())
    return {
        'n_visible': int(visible.sum().item()),
        'n_masked': n_masked,
        'masked_match_ideal': match_ideal / float(n_masked),
        'masked_match_a': match_a / float(n_masked),
        'full_match_ideal': full / float(gen.numel()),
    }


def write_pair_grid(rows, path):
    """Each row: A recon | B recon | ideal stitch | generated."""
    h, w = rows[0][0].shape[:2]
    bar = 22
    canvas = np.full((len(rows) * (h + bar), 4 * w, 3), 255, dtype=np.uint8)
    labels = ('A recon', 'B recon', 'ideal stitch', 'generated')
    for i, panels in enumerate(rows):
        y = i * (h + bar) + bar
        for k, panel in enumerate(panels):
            canvas[y:y + h, k * w:(k + 1) * w] = panel
    im = Image.fromarray(canvas)
    draw = ImageDraw.Draw(im)
    for i, _ in enumerate(rows):
        y = i * (h + bar) + 4
        for k, lab in enumerate(labels):
            draw.text((k * w + 6, y), lab, fill=(0, 0, 0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path)
    return path


def main():
    args = get_args_parser().parse_args()
    args.ckpt = args.ckpt or default_ckpt(args.exp_name, args.n)
    splits_dir = args.splits_dir or default_splits_dir()
    source_root = args.data_path or default_source_root()
    schedules = [s.strip() for s in args.schedules.split(',') if s.strip()]
    for name in schedules:
        if name not in SCHEDULES:
            raise ValueError('unknown schedule {}'.format(name))

    out_root = args.output_dir or os.path.join(
        'outputs', args.exp_name, 'n{}'.format(args.n), 'phase2_half_ab')
    source = os.path.join(out_root, args.pattern)
    os.makedirs(source, exist_ok=True)

    device = torch.device(args.device)
    rel_paths = load_n_list(args.n, splits_dir)
    pairs = make_pairs(len(rel_paths), args.max_pairs)
    tf = train_eval_transform()

    print('loading checkpoint {}'.format(args.ckpt), flush=True)
    linear_head = eval_exp.infer_linear_head(args, os.path.dirname(args.ckpt))
    model = eval_exp.load_mage(args, device, linear_head=linear_head)
    vqgan = model.vqgan

    a_paths = [rel_paths[i] for i, _ in pairs]
    b_paths = [rel_paths[j] for _, j in pairs]
    a_imgs = torch.stack([tf(Image.open(os.path.join(source_root, p)).convert('RGB'))
                          for p in a_paths]).to(device)
    b_imgs = torch.stack([tf(Image.open(os.path.join(source_root, p)).convert('RGB'))
                          for p in b_paths]).to(device)
    with torch.no_grad():
        a_tokens = eval_exp.encode_tokens(vqgan, a_imgs)
        b_tokens = eval_exp.encode_tokens(vqgan, b_imgs)

    visible, mask_desc = known_mask(args.pattern, device, args.checker_seed)
    visible = visible.unsqueeze(0).expand(a_tokens.size(0), -1).contiguous()
    known = torch.where(visible, a_tokens, b_tokens)
    ideal = known  # A where visible, B where masked

    payload = {
        'n': args.n,
        'exp_name': args.exp_name,
        'checkpoint': args.ckpt,
        'pattern': args.pattern,
        'pattern_desc': mask_desc,
        'schedules': {},
        'pairs': [{'index': k, 'A': a_paths[k], 'B': b_paths[k]}
                  for k in range(len(pairs))],
        'generation': {
            'argmax': bool(args.argmax),
            'temperature': args.temperature,
            'num_iter': args.num_iter,
            'seed': args.gen_seed,
        },
    }

    for schedule in schedules:
        t0 = time.time()
        print('schedule {}'.format(schedule), flush=True)
        gen = generate(model, schedule, known, visible, args)
        rows = []
        for k in range(len(pairs)):
            rows.append(pair_metrics(gen[k], ideal[k], a_tokens[k], visible[k]))
        if args.no_pixel:
            pixel = {'gen_vs_ideal': None, 'gen_vs_a': None, 'gen_vs_b': None}
            panels = None
        else:
            with torch.no_grad():
                dec_gen = eval_exp.decode_tokens(vqgan, gen)
                dec_ideal = eval_exp.decode_tokens(vqgan, ideal)
                dec_a = eval_exp.decode_tokens(vqgan, a_tokens)
                dec_b = eval_exp.decode_tokens(vqgan, b_tokens)
            g = [to_uint8_image(dec_gen[i]) for i in range(len(pairs))]
            iu = [to_uint8_image(dec_ideal[i]) for i in range(len(pairs))]
            au = [to_uint8_image(dec_a[i]) for i in range(len(pairs))]
            bu = [to_uint8_image(dec_b[i]) for i in range(len(pairs))]
            pixel = {
                'gen_vs_ideal': float(np.mean([pixel_mse_uint8(g[i], iu[i])
                                               for i in range(len(pairs))])),
                'gen_vs_a': float(np.mean([pixel_mse_uint8(g[i], au[i])
                                           for i in range(len(pairs))])),
                'gen_vs_b': float(np.mean([pixel_mse_uint8(g[i], bu[i])
                                           for i in range(len(pairs))])),
            }
            panels = [(au[i], bu[i], iu[i], g[i]) for i in range(len(pairs))]

        agg = {
            'masked_match_ideal': float(np.mean([r['masked_match_ideal'] for r in rows])),
            'masked_match_a': float(np.mean([r['masked_match_a'] for r in rows])),
            'full_match_ideal': float(np.mean([r['full_match_ideal'] for r in rows])),
        }
        entry = {'aggregate': agg, 'pixel': pixel,
                 'seconds': time.time() - t0, 'per_pair': rows}
        payload['schedules'][schedule] = entry
        if panels is not None:
            write_pair_grid(panels, os.path.join(source, 'grid_{}.png'.format(schedule)))
        print(schedule, json.dumps(agg), flush=True)

    path = os.path.join(source, 'metrics.json')
    with open(path, 'w') as f:
        json.dump(payload, f, indent=1, sort_keys=True)
    print('wrote', path, flush=True)


if __name__ == '__main__':
    main()
