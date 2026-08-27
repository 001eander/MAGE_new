"""Phase-0 eval: FID, coverage, match error, grid -> outputs/<exp>/n<N>/metrics.json.

VQGAN reconstruction FID and generated FID use the same fid_two_folders(),
which is pytorch-fid's calculate_fid_given_paths (LGANs-TT realdata.py).
Generation defaults (temperature / steps / argmax / count) are written into
metrics.json and stay the same across setting comparisons unless overridden.
"""
import argparse
import math
import os

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


def load_mage(args, device):
    model = models_mage.__dict__[args.model](
        norm_pix_loss=False,
        mask_ratio_mu=0.55,
        mask_ratio_std=0.25,
        mask_ratio_min=0.0,
        mask_ratio_max=1.0,
        label_smoothing=0.0,
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


def load_folder_images(folder):
    paths = []
    for dirpath, _, filenames in os.walk(folder):
        for name in sorted(filenames):
            if name.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths.append(os.path.join(dirpath, name))
    if not paths:
        raise FileNotFoundError('no images in {}'.format(folder))
    images = []
    for p in paths:
        images.append(np.array(Image.open(p).convert('RGB')))
    return images, paths


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
    codebook_size = 1024
    mask_token_id = model.mask_token_label
    unknown_number_in_the_beginning = 256
    known = float('inf')
    token_indices = mask_token_id * torch.ones(
        bsz, unknown_number_in_the_beginning, device=device)
    sampled_ids = None
    for step in range(num_iter):
        cur_ids = token_indices.clone().long()
        padded = torch.cat(
            [torch.zeros(bsz, 1, device=device), token_indices], dim=1)
        padded[:, 0] = model.fake_class_label
        padded = padded.long()
        token_all_mask = (padded == mask_token_id).float()
        token_drop_mask = torch.zeros_like(padded)
        x = model.token_emb(padded)
        for blk in model.blocks:
            x = blk(x)
        x = model.norm(x)
        logits = model.forward_decoder(x, token_drop_mask, token_all_mask)
        logits = logits[:, 1:, :codebook_size]
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


def run_recon_fid(args, device, out_dir):
    ref_dir = args.ref_dir or default_ref_dir()
    if not os.path.isdir(ref_dir):
        raise FileNotFoundError(
            'FID reference missing: {} (run scripts/prepare_tiny_val256.py)'.format(ref_dir))
    vqgan = load_vqgan_only(args.vqgan_ckpt, device)
    recon_dir = args.recon_dir or os.path.join(out_dir, 'ref_vqgan_recon')
    if not os.path.isdir(recon_dir) or not os.listdir(recon_dir):
        print('reconstructing reference set ->', recon_dir)
        images, _ = load_folder_images(ref_dir)
        tf = train_eval_transform()
        tensors = [tf(Image.fromarray(im)) for im in images]
        _, recons = encode_folder(vqgan, tensors, device, args.batch_size)
        save_uint8_folder(recons, recon_dir)
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
    generation = {
        'temperature': args.temperature,
        'num_iter': args.num_iter,
        'argmax': bool(args.argmax),
        'num_images': num_images,
        'seed': args.gen_seed,
    }

    if args.recon_fid_only:
        run_recon_fid(args, device, out_dir)
        return

    rel_paths = load_n_list(args.n, splits_dir)
    train_tensors, train_abs = tensors_from_list(rel_paths, source_root)
    model = None
    vqgan = None
    if args.ckpt:
        model = load_mage(args, device)
        vqgan = model.vqgan
    else:
        vqgan = load_vqgan_only(args.vqgan_ckpt, device)

    print('encoding train set, n={}'.format(len(rel_paths)))
    with torch.no_grad():
        train_tokens, train_recons = encode_folder(
            vqgan, train_tensors, device, args.batch_size)
    np.save(os.path.join(out_dir, 'train_tokens.npy'), train_tokens)
    train_recon_dir = os.path.join(out_dir, 'train_vqgan_recon')
    save_uint8_folder(train_recons, train_recon_dir)
    train_uint8 = [to_uint8_image(t.numpy()) for t in train_tensors]

    gen_dir = args.gen_dir or os.path.join(out_dir, 'gen')
    gen_token_path = os.path.join(out_dir, 'gen_tokens.npy')
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
        print('generating', num_images, 'images')
        gen_images = []
        gen_token_rows = []
        n_done = 0
        step = 0
        while n_done < num_images:
            bsz = min(args.batch_size, num_images - n_done)
            with torch.no_grad():
                imgs, toks = generate_batch(
                    model, bsz=bsz, seed=args.gen_seed + step,
                    num_iter=args.num_iter, temperature=args.temperature,
                    argmax=args.argmax, device=device)
            for j in range(bsz):
                arr = to_uint8_image(imgs[j])
                Image.fromarray(arr).save(
                    os.path.join(gen_dir, '{}.png'.format(str(n_done + j).zfill(5))))
                gen_images.append(arr)
            gen_token_rows.append(toks.detach().cpu().numpy())
            n_done += bsz
            step += 1
        gen_tokens = np.concatenate(gen_token_rows, axis=0)
        np.save(gen_token_path, gen_tokens)

    print('coverage and match error')
    train_to_gen_err, _ = nearest_token(train_tokens, gen_tokens)
    gen_to_train_err, gen_nn = nearest_token(gen_tokens, train_tokens)
    coverage = coverage_from_errors(train_to_gen_err)
    pixel_mses = [
        pixel_mse_uint8(gen_images[i], train_recons[int(gen_nn[i])])
        for i in range(len(gen_images))
    ]
    match_error = {
        'token_frac': token_match_frac(gen_to_train_err),
        'pixel_mse_vqgan': float(np.mean(pixel_mses)) if pixel_mses else None,
        'n_generated': int(len(gen_images)),
    }

    grid_path = os.path.join(out_dir, 'grid.png')
    write_grid(gen_images, train_uint8, train_recons, gen_nn, grid_path, n_show=GRID_N)

    oneshot_vs_self = None
    if model is not None:
        probe_file = args.probe_list or os.path.join(splits_dir, 'probe.txt')
        if os.path.isfile(probe_file) and args.n >= 1000:
            probe_rel = [p for p in read_list(probe_file) if p in set(rel_paths)]
        else:
            probe_rel = list(rel_paths)
        if probe_rel:
            probe_tensors, _ = tensors_from_list(probe_rel, source_root)
            accs = []
            with torch.no_grad():
                for i in range(0, len(probe_tensors), args.batch_size):
                    batch = torch.stack(
                        probe_tensors[i:i + args.batch_size], dim=0).to(device)
                    gt = encode_tokens(vqgan, batch)
                    pred = oneshot_predict(model, batch)
                    accs.append((pred == gt).float().mean(dim=1).detach().cpu().numpy())
            accs = np.concatenate(accs, axis=0)
            oneshot_vs_self = {
                'token_acc': float(accs.mean()),
                'per_image': [float(x) for x in accs],
                'probe': probe_rel,
                'note': 'all-mask one-step argmax vs that image own VQGAN tokens',
            }

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
            print('reconstructing reference set ->', recon_dir)
            images, _ = load_folder_images(ref_dir)
            tf = train_eval_transform()
            tensors = [tf(Image.fromarray(im)) for im in images]
            with torch.no_grad():
                _, recons = encode_folder(vqgan, tensors, device, args.batch_size)
            save_uint8_folder(recons, recon_dir)
        print('FID generated vs ref and VQGAN-recon vs ref (pytorch-fid / LGANs-TT realdata.py)')
        extractor = build_inception(device)
        fid['generated_vs_ref'] = fid_two_folders(
            gen_dir, ref_dir, device, batch_size=FID_BATCH_SIZE,
            num_workers=args.num_workers, extractor=extractor)
        fid['vqgan_recon_vs_ref'] = fid_two_folders(
            recon_dir, ref_dir, device, batch_size=FID_BATCH_SIZE,
            num_workers=args.num_workers, extractor=extractor)
        fid['implementation'] = FID_IMPLEMENTATION

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
        extra={'train_abs_paths': train_abs},
    )
    metrics_path = os.path.join(out_dir, 'metrics.json')
    write_metrics_json(metrics, path=metrics_path)
    print(metrics_path)


if __name__ == '__main__':
    main()
