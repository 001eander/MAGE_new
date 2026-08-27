"""Judge single-image memorization by token accuracy against VQGAN codes.

MAGE predicts 256 codebook indices. Pixel-perfect match to the original
image is impossible; the ceiling is VQGAN encode→decode of the same crop.
Success is oneshot_token_acc == 1.0 (256/256). Iterative remask is reported but not required.
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import PIL.Image
import PIL.ImageDraw
import torchvision.transforms as transforms

import models_mage
from gen_img_uncond import gen_image


def load_image_tensor(path, input_size=256):
    img = PIL.Image.open(path).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize(input_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
    ])
    return tf(img).unsqueeze(0)


def to_uint8(tensor):
    x = tensor.detach().float().cpu()
    if x.dim() == 4:
        x = x[0]
    x = x.clamp(0, 1).permute(1, 2, 0).numpy()
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


def mse_psnr(a, b, max_val=1.0):
    mse = float(((a - b) ** 2).mean().item())
    if mse <= 0.0:
        return mse, float('inf')
    return mse, 10.0 * math.log10((max_val ** 2) / mse)


def json_float(x):
    if x is None:
        return None
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return None
    return x


def save_compare(panels, labels, path, bar_h=28):
    assert len(panels) == len(labels)
    h, w = panels[0].shape[:2]
    row = np.concatenate(panels, axis=1)
    canvas = np.full((h + bar_h, row.shape[1], 3), 255, dtype=np.uint8)
    canvas[bar_h:] = row
    im = PIL.Image.fromarray(canvas)
    draw = PIL.ImageDraw.Draw(im)
    for i, lab in enumerate(labels):
        draw.text((i * w + 6, 6), lab, fill=(0, 0, 0))
    im.save(path)


def get_args_parser():
    parser = argparse.ArgumentParser('Single-image MAGE fit eval', add_help=False)
    parser.add_argument('--ckpt', required=True, type=str)
    parser.add_argument('--image', required=True, type=str)
    parser.add_argument('--model', default='mage_vit_base_patch16', type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--input_size', default=256, type=int)
    parser.add_argument('--num_iter', default=12, type=int)
    parser.add_argument('--vqgan_ckpt', default='vqgan_jax_strongaug.ckpt', type=str)
    return parser


def main():
    args = get_args_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    ckpt_args = checkpoint.get('args', None)
    model = models_mage.__dict__[args.model](
        norm_pix_loss=False,
        mask_ratio_mu=0.55, mask_ratio_std=0.25,
        mask_ratio_min=0.0, mask_ratio_max=1.0,
        dropout=0.0,
        decoder_mask_token=bool(getattr(ckpt_args, 'decoder_mask_token', False)),
        linear_head=bool(getattr(ckpt_args, 'linear_head', False)),
        vqgan_ckpt_path=args.vqgan_ckpt,
    )
    model.to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    imgs = load_image_tensor(args.image, args.input_size).to(device)
    with torch.no_grad():
        gt_indices = model.encode_to_indices(imgs)
        oneshot_pred, gt_check = model.predict_full_mask(imgs)
        assert torch.equal(gt_indices, gt_check)
        vqgan_img = model.decode_indices(gt_indices)
        oneshot_img = model.decode_indices(oneshot_pred)
        iter_img, iter_pred = gen_image(
            model, bsz=1, seed=0, num_iter=args.num_iter,
            choice_temperature=0.0, greedy=True)

    n_tokens = int(gt_indices.numel())
    oneshot_correct = int((oneshot_pred == gt_indices).sum().item())
    iter_correct = int((iter_pred == gt_indices).sum().item())
    oneshot_acc = oneshot_correct / float(n_tokens)
    iter_acc = iter_correct / float(n_tokens)

    mse_vq_in, psnr_vq_in = mse_psnr(vqgan_img, imgs)
    mse_os_vq, psnr_os_vq = mse_psnr(oneshot_img, vqgan_img)
    mse_it_vq, psnr_it_vq = mse_psnr(iter_img, vqgan_img)
    mse_os_in, psnr_os_in = mse_psnr(oneshot_img, imgs)
    mse_it_in, psnr_it_in = mse_psnr(iter_img, imgs)

    perfect = oneshot_correct == n_tokens
    metrics = {
        'n_tokens': n_tokens,
        'oneshot_token_correct': oneshot_correct,
        'oneshot_token_acc': oneshot_acc,
        'iterative_token_correct': iter_correct,
        'iterative_token_acc': iter_acc,
        'token_exact': perfect,
        'mse_vqgan_vs_input': mse_vq_in,
        'psnr_vqgan_vs_input': json_float(psnr_vq_in),
        'mse_oneshot_vs_vqgan': mse_os_vq,
        'psnr_oneshot_vs_vqgan': json_float(psnr_os_vq),
        'mse_iterative_vs_vqgan': mse_it_vq,
        'psnr_iterative_vs_vqgan': json_float(psnr_it_vq),
        'mse_oneshot_vs_input': mse_os_in,
        'psnr_oneshot_vs_input': json_float(psnr_os_in),
        'mse_iterative_vs_input': mse_it_in,
        'psnr_iterative_vs_input': json_float(psnr_it_in),
        'perfect': perfect,
        'psnr_oneshot_vs_vqgan_is_inf': math.isinf(psnr_os_vq),
        'psnr_iterative_vs_vqgan_is_inf': math.isinf(psnr_it_vq),
        'success_rule': 'oneshot_token_acc == 1.0 (256/256); generation is all-mask greedy decode; pixels vs VQGAN recon',
    }

    input_u8 = to_uint8(imgs)
    vq_u8 = to_uint8(vqgan_img)
    os_u8 = to_uint8(oneshot_img)
    it_u8 = to_uint8(iter_img)
    diff_os_u8 = np.abs(os_u8.astype(np.int16) - vq_u8.astype(np.int16)).astype(np.uint8)
    diff_it_u8 = np.abs(it_u8.astype(np.int16) - vq_u8.astype(np.int16)).astype(np.uint8)

    PIL.Image.fromarray(input_u8).save(os.path.join(args.output_dir, 'input_256.png'))
    PIL.Image.fromarray(vq_u8).save(os.path.join(args.output_dir, 'vqgan_recon.png'))
    PIL.Image.fromarray(os_u8).save(os.path.join(args.output_dir, 'mage_oneshot.png'))
    PIL.Image.fromarray(it_u8).save(os.path.join(args.output_dir, 'mage_iterative.png'))
    PIL.Image.fromarray(diff_os_u8).save(os.path.join(args.output_dir, 'absdiff_oneshot_vqgan.png'))
    PIL.Image.fromarray(diff_it_u8).save(os.path.join(args.output_dir, 'absdiff_iterative_vqgan.png'))

    labels = ['input', 'vqgan', 'oneshot', 'iterative', '|oneshot-vqgan|', '|iter-vqgan|']
    save_compare(
        [input_u8, vq_u8, os_u8, it_u8, diff_os_u8, diff_it_u8],
        labels,
        os.path.join(args.output_dir, 'compare.png'))
    with open(os.path.join(args.output_dir, 'compare_labels.txt'), 'w') as f:
        f.write(' | '.join(labels) + '\n')

    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
        f.write('\n')

    print(json.dumps(metrics, indent=2))
    print('wrote', metrics_path)
    if not perfect:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
