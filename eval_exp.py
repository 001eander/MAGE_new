"""Phase-0 eval: FID, coverage, match error, grid -> outputs/<exp>/n<N>/metrics.json.

VQGAN reconstruction FID and generated FID use the same fid_two_folders(),
which is pytorch-fid's calculate_fid_given_paths (LGANs-TT realdata.py).
Generation defaults (temperature / steps / argmax / count) are written into
metrics.json and stay the same across setting comparisons unless overridden.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from PIL import Image

import models_mage
from util.exp_data import (
    DEFAULT_ARGMAX,
    DEFAULT_GEN_SEED,
    DEFAULT_NUM_ITER,
    DEFAULT_TEMPERATURE,
    GRID_N,
    ImageListDataset,
    default_num_images,
    default_ref_dir,
    default_source_root,
    default_splits_dir,
    generation_defaults,
    list_path,
    load_n_list,
    read_list,
    train_eval_transform,
)
from util.exp_fid import (
    FID_BATCH_SIZE,
    IMPLEMENTATION as FID_IMPLEMENTATION,
    build_inception,
    fid_two_folders,
)
from util.exp_metrics import (
    build_metrics,
    coverage_from_errors,
    nearest_token,
    pixel_mse_uint8,
    to_uint8_image,
    token_match_frac,
    write_grid,
    write_metrics_json,
    wrong_token_embed_stats,
)


def get_args_parser():
    parser = argparse.ArgumentParser('MAGE experiment eval', add_help=False)
    parser.add_argument('--n', type=int, required=True, help='train set size')
    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--output_dir', type=str, default='',
                        help='default: outputs/<exp_name>/n<N>')
    parser.add_argument('--splits_dir', type=str, default='')
    parser.add_argument('--data_path', type=str, default='',
                        help='Tiny-ImageNet root; list paths are relative to this')
    parser.add_argument('--ref_dir', type=str, default='',
                        help='FID reference images (default: Tiny val upscaled to 256)')
    parser.add_argument('--ckpt', type=str, default='',
                        help='MAGE checkpoint. Required to generate or oneshot.')
    parser.add_argument('--model', type=str, default='mage_vit_base_patch16')
    parser.add_argument('--vqgan_ckpt', type=str, default='vqgan_jax_strongaug.ckpt')
    parser.add_argument('--gen_dir', type=str, default='',
                        help='existing generated images; skip sampling if set')
    parser.add_argument('--recon_dir', type=str, default='',
                        help='existing VQGAN recon of --ref_dir; computed if empty')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument('--num_iter', type=int, default=DEFAULT_NUM_ITER)
    parser.add_argument('--schedule', type=str, default='cosine',
                        choices=('cosine', 'uncover1'),
                        help='cosine: paper remask (default). '
                             'uncover1: commit one unknown position per step, never remask')
    parser.add_argument('--argmax', action='store_true', default=DEFAULT_ARGMAX)
    parser.add_argument('--sample', action='store_false', dest='argmax')
    parser.add_argument('--num_images', type=int, default=-1,
                        help='-1 = max(1000, 50n) or 50000 if n>=10000')
    parser.add_argument('--gen_seed', type=int, default=DEFAULT_GEN_SEED)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--skip_fid', action='store_true')
    parser.add_argument('--recon_fid_only', action='store_true',
                        help='only VQGAN-recon FID vs ref (same FID function)')
    parser.add_argument('--probe_list', type=str, default='',
                        help='oneshot-vs-self probe; default splits/probe.txt')
    parser.add_argument('--linear_head', action='store_true',
                        help='Load the independent linear codebook head. '
                             'Also inferred from args.json or checkpoint keys.')
    return parser


def output_dir_for(args):
    if args.output_dir:
        return args.output_dir
    return os.path.join('outputs', args.exp_name, 'n{}'.format(args.n))


def load_vqgan_only(vqgan_ckpt, device):
    from omegaconf import OmegaConf
    from taming.models.vqgan import VQModel
    config = OmegaConf.load('config/vqgan.yaml').model
    vqgan = VQModel(
        ddconfig=config.params.ddconfig,
        n_embed=config.params.n_embed,
        embed_dim=config.params.embed_dim,
        ckpt_path=vqgan_ckpt,
    )
    vqgan.eval()
    vqgan.to(device)
    for p in vqgan.parameters():
        p.requires_grad = False
    return vqgan


def _label_smoothing_from_args(out_dir):
    path = os.path.join(out_dir, 'args.json')
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f).get('label_smoothing')


def infer_linear_head(args, out_dir):
    if getattr(args, 'linear_head', False):
        return True
    candidates = []
    if out_dir:
        candidates.append(os.path.join(out_dir, 'args.json'))
    if args.ckpt:
        candidates.append(os.path.join(os.path.dirname(args.ckpt), 'args.json'))
    for path in candidates:
        if os.path.isfile(path):
            with open(path) as f:
                payload = json.load(f)
            return bool(payload.get('linear_head', False))
    if args.ckpt and os.path.isfile(args.ckpt):
        checkpoint = torch.load(args.ckpt, map_location='cpu')
        keys = checkpoint.get('model', checkpoint).keys()
        return any(k.startswith('token_classifier') for k in keys)
    return False


def load_mage(args, device, linear_head=False):
    model = models_mage.__dict__[args.model](
        norm_pix_loss=False,
        mask_ratio_mu=0.55,
        mask_ratio_std=0.25,
        mask_ratio_min=0.0,
        mask_ratio_max=1.0,
        label_smoothing=0.0,
        linear_head=linear_head,
        vqgan_ckpt_path=args.vqgan_ckpt,
    )
    model.to(device)
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def vqgan_of(model_or_vqgan):
    return getattr(model_or_vqgan, 'vqgan', model_or_vqgan)


def encode_tokens(vqgan, images):
    z_q, _, token_tuple = vqgan.encode(images)
    _, _, token_indices = token_tuple
    return token_indices.reshape(z_q.size(0), -1).long()


def decode_tokens(vqgan, tokens, codebook_emb_dim=256):
    bsz = tokens.size(0)
    z_q = vqgan.quantize.get_codebook_entry(
        tokens.reshape(-1), shape=(bsz, 16, 16, codebook_emb_dim))
    return vqgan.decode(z_q)


def save_uint8_folder(images, folder, prefix=''):
    os.makedirs(folder, exist_ok=True)
    paths = []
    for i, img in enumerate(images):
        arr = to_uint8_image(img)
        path = os.path.join(folder, '{}{}.png'.format(prefix, str(i).zfill(5)))
        Image.fromarray(arr).save(path)
        paths.append(path)
    return paths


def list_image_paths(folder):
    paths = []
    for dirpath, _, filenames in os.walk(folder):
        for name in sorted(filenames):
            if name.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths.append(os.path.join(dirpath, name))
    if not paths:
        raise FileNotFoundError('no images in {}'.format(folder))
    return paths


def load_folder_images(folder):
    paths = list_image_paths(folder)
    images = []
    for p in paths:
        images.append(np.array(Image.open(p).convert('RGB')))
    return images, paths


def reconstruct_folder(vqgan, src_dir, dest_dir, device, batch_size):
    """Encode/decode one batch at a time. Do not load the whole folder."""
    os.makedirs(dest_dir, exist_ok=True)
    paths = list_image_paths(src_dir)
    tf = train_eval_transform()
    n_done = 0
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        tensors = [tf(Image.open(p).convert('RGB')) for p in batch_paths]
        batch = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            rec = decode_tokens(vqgan, encode_tokens(vqgan, batch))
        for j in range(rec.size(0)):
            Image.fromarray(to_uint8_image(rec[j])).save(
                os.path.join(dest_dir, '{}.png'.format(str(n_done + j).zfill(5))))
        n_done += rec.size(0)
        if n_done % 500 == 0 or n_done == len(paths):
            print('reconstructed', n_done, '/', len(paths), flush=True)
    return dest_dir


def oneshot_predict(model, images):
    """All positions masked, nothing dropped, argmax over codebook."""
    device = images.device
    bsz = images.size(0)
    mask_id = model.mask_token_label
    token_indices = mask_id * torch.ones(bsz, 256, device=device)
    token_indices = torch.cat(
        [torch.zeros(bsz, 1, device=device), token_indices], dim=1)
    token_indices[:, 0] = model.fake_class_label
    token_indices = token_indices.long()
    token_all_mask = torch.ones_like(token_indices, dtype=torch.float32)
    token_all_mask[:, 0] = 0
    token_drop_mask = torch.zeros_like(token_indices, dtype=torch.float32)
    x = model.token_emb(token_indices)
    for blk in model.blocks:
        x = blk(x)
    x = model.norm(x)
    logits = model.forward_decoder(x, token_drop_mask, token_all_mask)
    return logits[:, 1:, :model.codebook_size].argmax(dim=-1)


def mage_codebook_logits(model, token_indices):
    """Logits [B, 256, 1024] from current token ids (mask_token_label = unknown)."""
    bsz = token_indices.size(0)
    padded = torch.cat(
        [torch.zeros(bsz, 1, device=token_indices.device), token_indices], dim=1)
    padded[:, 0] = model.fake_class_label
    padded = padded.long()
    mask_token_id = model.mask_token_label
    token_all_mask = (padded == mask_token_id).float()
    token_drop_mask = torch.zeros_like(padded)
    x = model.token_emb(padded)
    for blk in model.blocks:
        x = blk(x)
    x = model.norm(x)
    logits = model.forward_decoder(x, token_drop_mask, token_all_mask)
    return logits[:, 1:, :model.codebook_size]


def mask_by_random_topk(mask_len, probs, temperature=1.0):
    mask_len = mask_len.squeeze()
    confidence = torch.log(probs) + torch.Tensor(
        temperature * np.random.gumbel(size=probs.shape)).to(probs.device)
    sorted_confidence, _ = torch.sort(confidence, axis=-1)
    cut_off = sorted_confidence[:, mask_len.long() - 1:mask_len.long()]
    return confidence <= cut_off


def generate_batch(model, bsz, seed, num_iter, temperature, argmax, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    codebook_emb_dim = 256
    mask_token_id = model.mask_token_label
    unknown_number_in_the_beginning = 256
    known = float('inf')
    token_indices = mask_token_id * torch.ones(
        bsz, unknown_number_in_the_beginning, device=device)
    sampled_ids = None
    for step in range(num_iter):
        cur_ids = token_indices.clone().long()
        logits = mage_codebook_logits(model, cur_ids)
        if argmax:
            sampled_ids = torch.argmax(logits, dim=-1)
        else:
            sampled_ids = torch.distributions.categorical.Categorical(
                logits=logits).sample()
        unknown_map = (cur_ids == mask_token_id)
        sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids)
        ratio = 1.0 * (step + 1) / num_iter
        mask_ratio = np.cos(math.pi / 2.0 * ratio)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        selected_probs = torch.squeeze(
            torch.gather(probs, dim=-1, index=torch.unsqueeze(sampled_ids, -1)),
            -1)
        selected_probs = torch.where(
            unknown_map, selected_probs.double(),
            torch.tensor(known, device=device, dtype=torch.float64)).float()
        mask_len = torch.tensor(
            [np.floor(unknown_number_in_the_beginning * mask_ratio)],
            device=device)
        mask_len = torch.maximum(
            torch.tensor([1.0], device=device),
            torch.minimum(torch.sum(unknown_map, dim=-1, keepdims=True) - 1, mask_len))
        masking = mask_by_random_topk(
            mask_len[0], selected_probs, temperature * (1 - ratio))
        token_indices = torch.where(masking, mask_token_id, sampled_ids)
    images = decode_tokens(model.vqgan, sampled_ids, codebook_emb_dim)
    return images, sampled_ids


def generate_batch_uncover1(model, bsz, seed, argmax, device):
    """256 steps: commit the most-confident unknown position; never remask.

    Positions still predict independently each step (same as the paper
    sampler). The difference is the reveal schedule: exactly one id is
    written per image per step, and committed ids stay visible.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    mask_token_id = model.mask_token_label
    token_indices = mask_token_id * torch.ones(bsz, 256, device=device, dtype=torch.long)
    batch_idx = torch.arange(bsz, device=device)
    for step in range(256):
        unknown = token_indices == mask_token_id
        if not bool(unknown.any()):
            break
        logits = mage_codebook_logits(model, token_indices)
        if argmax:
            pred = torch.argmax(logits, dim=-1)
        else:
            pred = torch.distributions.categorical.Categorical(
                logits=logits).sample()
        probs = torch.nn.functional.softmax(logits, dim=-1)
        conf = torch.gather(probs, dim=-1, index=pred.unsqueeze(-1)).squeeze(-1)
        conf = conf.masked_fill(~unknown, -1.0e9)
        pick = conf.argmax(dim=-1)
        token_indices = token_indices.clone()
        token_indices[batch_idx, pick] = pred[batch_idx, pick]
        if (step + 1) % 32 == 0 or step == 0:
            n_left = int(unknown.sum().item())
            print('uncover1 step', step + 1, '/ 256 unknown', n_left, flush=True)
    images = decode_tokens(model.vqgan, token_indices, 256)
    return images, token_indices


def encode_folder(vqgan, image_tensors, device, batch_size):
    tokens = []
    recons = []
    for i in range(0, len(image_tensors), batch_size):
        batch = torch.stack(image_tensors[i:i + batch_size], dim=0).to(device)
        with torch.no_grad():
            tok = encode_tokens(vqgan, batch)
            rec = decode_tokens(vqgan, tok)
        tokens.append(tok.detach().cpu().numpy())
        recons.extend([to_uint8_image(rec[j]) for j in range(rec.size(0))])
    return np.concatenate(tokens, axis=0), recons


def tensors_from_list(rel_paths, source_root, input_size=256):
    ds = ImageListDataset(
        rel_paths, source_root=source_root, transform=train_eval_transform(input_size))
    return [ds[i][0] for i in range(len(ds))], ds.paths


def encode_image_list(vqgan, rel_paths, source_root, device, batch_size, recon_dir=None):
    """Encode train images in batches. Do not hold the float tensors."""
    ds = ImageListDataset(
        rel_paths, source_root=source_root, transform=train_eval_transform())
    tokens = []
    recons = []
    if recon_dir:
        os.makedirs(recon_dir, exist_ok=True)
    n_done = 0
    for start in range(0, len(ds), batch_size):
        end = min(start + batch_size, len(ds))
        batch = torch.stack([ds[i][0] for i in range(start, end)], dim=0).to(device)
        with torch.no_grad():
            tok = encode_tokens(vqgan, batch)
            rec = decode_tokens(vqgan, tok)
        tokens.append(tok.detach().cpu().numpy())
        for j in range(rec.size(0)):
            arr = to_uint8_image(rec[j])
            recons.append(arr)
            if recon_dir:
                Image.fromarray(arr).save(
                    os.path.join(recon_dir, '{}.png'.format(str(n_done + j).zfill(5))))
        n_done = end
        if n_done % 500 == 0 or n_done == len(ds):
            print('encoded train', n_done, '/', len(ds), flush=True)
    return np.concatenate(tokens, axis=0), recons, ds.paths


def gen_png_path(gen_dir, index):
    return os.path.join(gen_dir, '{}.png'.format(str(index).zfill(5)))


def pixel_mse_gen_dir(gen_dir, n_gen, train_recons, gen_nn):
    """Match-error pixels from saved PNGs so 50k gens stay off the heap."""
    mses = []
    for i in range(int(n_gen)):
        arr = np.array(Image.open(gen_png_path(gen_dir, i)).convert('RGB'))
        mses.append(pixel_mse_uint8(arr, train_recons[int(gen_nn[i])]))
        if (i + 1) % 5000 == 0 or (i + 1) == n_gen:
            print('pixel mse', i + 1, '/', n_gen, flush=True)
    return mses


def load_train_uint8(rel_paths, source_root, indices):
    tf = train_eval_transform()
    out = {}
    for j in indices:
        path = os.path.join(source_root, rel_paths[int(j)])
        out[int(j)] = to_uint8_image(tf(Image.open(path).convert('RGB')).numpy())
    return out


def cached_shared_recon_fid(recon_dir):
    """Reuse outputs/_ref/metrics.json when eval points at the shared recon dir."""
    if not recon_dir:
        return None
    shared = os.path.normpath('outputs/_ref/tiny_val256_vqgan_recon')
    if os.path.normpath(recon_dir) != shared:
        return None
    path = 'outputs/_ref/metrics.json'
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            metrics = json.load(f)
        value = (metrics.get('fid') or {}).get('vqgan_recon_vs_ref')
        if value is None:
            return None
        return float(value)
    except (OSError, ValueError, TypeError, KeyError):
        return None


def run_recon_fid(args, device, out_dir):
    ref_dir = args.ref_dir or default_ref_dir()
    if not os.path.isdir(ref_dir):
        raise FileNotFoundError(
            'FID reference missing: {} (run scripts/prepare_tiny_val256.py)'.format(ref_dir))
    vqgan = load_vqgan_only(args.vqgan_ckpt, device)
    recon_dir = args.recon_dir or os.path.join(out_dir, 'ref_vqgan_recon')
    if not os.path.isdir(recon_dir) or not os.listdir(recon_dir):
        print('reconstructing reference set ->', recon_dir, flush=True)
        reconstruct_folder(vqgan, ref_dir, recon_dir, device, args.batch_size)
    print('FID VQGAN-recon vs ref (pytorch-fid / LGANs-TT realdata.py)')
    recon_fid = fid_two_folders(
        recon_dir, ref_dir, device, batch_size=FID_BATCH_SIZE,
        num_workers=args.num_workers)
    generation = generation_defaults(args.n)
    generation.update({
        'temperature': args.temperature,
        'num_iter': args.num_iter,
        'argmax': args.argmax,
        'num_images': default_num_images(args.n) if args.num_images < 0 else args.num_images,
        'seed': args.gen_seed,
    })
    metrics = build_metrics(
        n=args.n,
        exp_name=args.exp_name,
        image_list=list_path(args.n, args.splits_dir or default_splits_dir()),
        train_rel_paths=load_n_list(args.n, args.splits_dir or default_splits_dir()),
        generation=generation,
        fid={
            'generated_vs_ref': None,
            'vqgan_recon_vs_ref': recon_fid,
            'generated_dir': None,
            'ref_dir': ref_dir,
            'recon_dir': recon_dir,
            'implementation': FID_IMPLEMENTATION,
        },
        coverage=None,
        match_error=None,
        oneshot_vs_self=None,
        grid_path=None,
        wrong_token_embed=None,
        extra={'mode': 'recon_fid_only'},
    )
    path = os.path.join(out_dir, 'metrics.json')
    write_metrics_json(metrics, path)
    print(path)
    return metrics


def main():
    args = get_args_parser().parse_args()
    splits_dir = args.splits_dir or default_splits_dir()
    source_root = args.data_path or default_source_root()
    ref_dir = args.ref_dir or default_ref_dir()
    out_dir = output_dir_for(args)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(args.device)
    num_images = default_num_images(args.n) if args.num_images < 0 else args.num_images
    schedule = args.schedule
    if schedule == 'uncover1':
        gen_iters = 256
        gen_note = (
            'uncover1: each step commits the most-confident unknown position; '
            'committed ids stay visible (no remask). Token sample/argmax matches '
            '--sample/--argmax. Not the paper cosine remask sampler.'
        )
    else:
        gen_iters = args.num_iter
        gen_note = (
            'Compare settings with these defaults unless the comparison '
            'is about temperature / steps / argmax / count / schedule.'
        )
    generation = {
        'temperature': args.temperature,
        'num_iter': gen_iters,
        'argmax': bool(args.argmax),
        'num_images': num_images,
        'seed': args.gen_seed,
        'schedule': schedule,
        'uncover_per_step': 1 if schedule == 'uncover1' else None,
        'remask': schedule == 'cosine',
        'note': gen_note,
    }

    if args.recon_fid_only:
        run_recon_fid(args, device, out_dir)
        return

    rel_paths = load_n_list(args.n, splits_dir)
    model = None
    vqgan = None
    linear_head = infer_linear_head(args, out_dir)
    if args.ckpt:
        model = load_mage(args, device, linear_head=linear_head)
        vqgan = model.vqgan
    else:
        vqgan = load_vqgan_only(args.vqgan_ckpt, device)

    print('encoding train set, n={}'.format(len(rel_paths)), flush=True)
    profile = {}
    t0 = time.time()
    train_recon_dir = os.path.join(out_dir, 'train_vqgan_recon')
    train_tokens, train_recons, train_abs = encode_image_list(
        vqgan, rel_paths, source_root, device, args.batch_size,
        recon_dir=train_recon_dir)
    profile['encode_train_s'] = time.time() - t0
    np.save(os.path.join(out_dir, 'train_tokens.npy'), train_tokens)

    gen_tag = 'uncover1' if schedule == 'uncover1' else ''
    gen_dir = args.gen_dir or os.path.join(
        out_dir, 'gen_uncover1' if gen_tag else 'gen')
    gen_token_path = os.path.join(
        out_dir, 'gen_tokens_uncover1.npy' if gen_tag else 'gen_tokens.npy')
    if args.gen_dir:
        gen_images, _ = load_folder_images(gen_dir)
        if os.path.isfile(gen_token_path):
            gen_tokens = np.load(gen_token_path)
        else:
            print('encoding generated images')
            tf = train_eval_transform()
            gen_tensors = [tf(Image.fromarray(im)) for im in gen_images]
            with torch.no_grad():
                gen_tokens, _ = encode_folder(vqgan, gen_tensors, device, args.batch_size)
            np.save(gen_token_path, gen_tokens)
        generation['num_images'] = len(gen_images)
    else:
        if model is None:
            raise ValueError('need --ckpt to generate, or pass --gen_dir')
        os.makedirs(gen_dir, exist_ok=True)
        print('generating', num_images, 'images schedule={}'.format(schedule), flush=True)
        t0 = time.time()
        gen_preview = []
        gen_token_rows = []
        n_done = 0
        step = 0
        while n_done < num_images:
            bsz = min(args.batch_size, num_images - n_done)
            with torch.no_grad():
                if schedule == 'uncover1':
                    imgs, toks = generate_batch_uncover1(
                        model, bsz=bsz, seed=args.gen_seed + step,
                        argmax=args.argmax, device=device)
                else:
                    imgs, toks = generate_batch(
                        model, bsz=bsz, seed=args.gen_seed + step,
                        num_iter=args.num_iter, temperature=args.temperature,
                        argmax=args.argmax, device=device)
            for j in range(bsz):
                arr = to_uint8_image(imgs[j])
                Image.fromarray(arr).save(gen_png_path(gen_dir, n_done + j))
                if len(gen_preview) < GRID_N:
                    gen_preview.append(arr)
            gen_token_rows.append(toks.detach().cpu().numpy())
            n_done += bsz
            step += 1
            if n_done % 256 == 0 or n_done == num_images:
                print('generated', n_done, '/', num_images, flush=True)
        gen_tokens = np.concatenate(gen_token_rows, axis=0)
        np.save(gen_token_path, gen_tokens)
        profile['generate_s'] = time.time() - t0
        gen_images = gen_preview

    print('coverage and match error', flush=True)
    t0 = time.time()
    train_to_gen_err, _ = nearest_token(train_tokens, gen_tokens)
    gen_to_train_err, gen_nn = nearest_token(gen_tokens, train_tokens)
    coverage = coverage_from_errors(train_to_gen_err)
    pixel_mses = pixel_mse_gen_dir(gen_dir, len(gen_tokens), train_recons, gen_nn)
    match_error = {
        'token_frac': token_match_frac(gen_to_train_err),
        'pixel_mse_vqgan': float(np.mean(pixel_mses)) if pixel_mses else None,
        'n_generated': int(len(gen_tokens)),
    }

    grid_path = os.path.join(
        out_dir, 'grid_uncover1.png' if gen_tag else 'grid.png')
    n_show = min(GRID_N, len(gen_images), len(gen_nn))
    nn_preview = [int(gen_nn[i]) for i in range(n_show)]
    uniq = []
    for j in nn_preview:
        if j not in uniq:
            uniq.append(j)
    loaded_train = load_train_uint8(rel_paths, source_root, uniq)
    remap = {old: new for new, old in enumerate(uniq)}
    write_grid(
        gen_images[:n_show],
        [loaded_train[j] for j in uniq],
        [train_recons[j] for j in uniq],
        np.array([remap[j] for j in nn_preview], dtype=np.int32),
        grid_path,
        n_show=n_show,
    )
    profile['coverage_match_s'] = time.time() - t0

    oneshot_vs_self = None
    wrong_token_embed = None
    existing_metrics = os.path.join(out_dir, 'metrics.json')
    if schedule == 'uncover1' and os.path.isfile(existing_metrics):
        with open(existing_metrics) as f:
            old = json.load(f)
        oneshot_vs_self = old.get('oneshot_vs_self')
        wrong_token_embed = old.get('wrong_token_embed')
        print('reused oneshot/wrong-token from', existing_metrics, flush=True)
    elif model is not None:
        probe_file = args.probe_list or os.path.join(splits_dir, 'probe.txt')
        if os.path.isfile(probe_file) and args.n >= 1000:
            probe_rel = [p for p in read_list(probe_file) if p in set(rel_paths)]
        else:
            probe_rel = list(rel_paths)
        if probe_rel:
            probe_tensors, _ = tensors_from_list(probe_rel, source_root)
            t0 = time.time()
            accs = []
            pred_rows = []
            gt_rows = []
            with torch.no_grad():
                for i in range(0, len(probe_tensors), args.batch_size):
                    batch = torch.stack(
                        probe_tensors[i:i + args.batch_size], dim=0).to(device)
                    gt = encode_tokens(vqgan, batch)
                    pred = oneshot_predict(model, batch)
                    accs.append((pred == gt).float().mean(dim=1).detach().cpu().numpy())
                    pred_rows.append(pred.detach().cpu().numpy())
                    gt_rows.append(gt.detach().cpu().numpy())
            accs = np.concatenate(accs, axis=0)
            pred_all = np.concatenate(pred_rows, axis=0)
            gt_all = np.concatenate(gt_rows, axis=0)
            oneshot_vs_self = {
                'token_acc': float(accs.mean()),
                'per_image': [float(x) for x in accs],
                'probe': probe_rel,
                'note': 'all-mask one-step argmax vs that image own VQGAN tokens',
            }
            emb = model.token_emb.word_embeddings.weight.detach().cpu().numpy()
            wrong_token_embed = wrong_token_embed_stats(
                pred_all, gt_all, emb[:model.codebook_size],
                seed=args.gen_seed, vocab_size=model.codebook_size)
            profile['oneshot_s'] = time.time() - t0

    fid = {
        'generated_vs_ref': None,
        'vqgan_recon_vs_ref': None,
        'generated_dir': gen_dir,
        'ref_dir': ref_dir,
        'recon_dir': args.recon_dir or os.path.join(out_dir, 'ref_vqgan_recon'),
    }
    if not args.skip_fid:
        if not os.path.isdir(ref_dir):
            raise FileNotFoundError(
                'FID reference missing: {} (run scripts/prepare_tiny_val256.py)'.format(ref_dir))
        recon_dir = fid['recon_dir']
        if not os.path.isdir(recon_dir) or not os.listdir(recon_dir):
            print('reconstructing reference set ->', recon_dir, flush=True)
            reconstruct_folder(vqgan, ref_dir, recon_dir, device, args.batch_size)
        print('FID generated vs ref and VQGAN-recon vs ref (pytorch-fid / LGANs-TT realdata.py)')
        t0 = time.time()
        extractor = build_inception(device)
        fid['generated_vs_ref'] = fid_two_folders(
            gen_dir, ref_dir, device, batch_size=FID_BATCH_SIZE,
            num_workers=args.num_workers, extractor=extractor)
        cached_recon = cached_shared_recon_fid(recon_dir)
        if cached_recon is not None:
            fid['vqgan_recon_vs_ref'] = cached_recon
            print('reused shared VQGAN-recon FID', cached_recon, flush=True)
        else:
            fid['vqgan_recon_vs_ref'] = fid_two_folders(
                recon_dir, ref_dir, device, batch_size=FID_BATCH_SIZE,
                num_workers=args.num_workers, extractor=extractor)
        fid['implementation'] = FID_IMPLEMENTATION
        profile['fid_s'] = time.time() - t0

    metrics = build_metrics(
        n=args.n,
        exp_name=args.exp_name,
        image_list=os.path.relpath(list_path(args.n, splits_dir), start=os.getcwd()),
        train_rel_paths=rel_paths,
        generation=generation,
        fid=fid,
        coverage=coverage,
        match_error=match_error,
        oneshot_vs_self=oneshot_vs_self,
        grid_path=os.path.relpath(grid_path, start=os.getcwd()),
        ref_dir=ref_dir,
        gen_dir=gen_dir,
        wrong_token_embed=wrong_token_embed,
        extra={
            'train_abs_paths': train_abs,
            'setting': {
                'linear_head': bool(linear_head),
                'head': 'linear' if linear_head else 'dot',
                'label_smoothing': _label_smoothing_from_args(out_dir),
            },
            'profile': profile,
        },
    )
    metrics_name = 'metrics_uncover1.json' if gen_tag else 'metrics.json'
    metrics_path = os.path.join(out_dir, metrics_name)
    write_metrics_json(metrics, metrics_path)
    print(metrics_path)


if __name__ == '__main__':
    main()
