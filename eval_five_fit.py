"""Judge 5-image memorization by per-image oneshot token accuracy.

Each ImageFolder class id is a condition token (codebook_size + id).
Success: every image has oneshot_token_acc == 1.0 (256/256).
Pixels are judged vs that image's VQGAN recon, not the raw JPEG.
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import PIL.Image

import models_mage
from eval_single_fit import json_float, load_image_tensor, mse_psnr, save_compare, to_uint8
from gen_img_uncond import gen_image


def get_args_parser():
    parser = argparse.ArgumentParser('Five-image MAGE fit eval', add_help=False)
    parser.add_argument('--ckpt', required=True, type=str)
    parser.add_argument('--data_path', required=True, type=str,
                        help='Folder with train/id0 ... train/id4')
    parser.add_argument('--model', default='mage_vit_base_patch16', type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--input_size', default=256, type=int)
    parser.add_argument('--num_iter', default=12, type=int)
    parser.add_argument('--vqgan_ckpt', default='vqgan_jax_strongaug.ckpt', type=str)
    parser.add_argument('--n_images', default=5, type=int)
    return parser


def discover_images(data_path, n_images):
    items = []
    train_root = os.path.join(data_path, 'train')
    for i in range(n_images):
        folder = os.path.join(train_root, 'id{}'.format(i))
        if not os.path.isdir(folder):
            raise FileNotFoundError(folder)
        files = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith(('.jpeg', '.jpg', '.png')))
        if not files:
            raise FileNotFoundError('no image in {}'.format(folder))
        items.append((i, os.path.join(folder, files[0])))
    return items


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
        use_cond_ids=bool(getattr(ckpt_args, 'cond_ids', False)),
        vqgan_ckpt_path=args.vqgan_ckpt,
    )
    model.to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    items = discover_images(args.data_path, args.n_images)
    per_image = []
    montage_rows = []
    all_perfect = True
    oneshot_correct_sum = 0
    iter_correct_sum = 0
    n_tokens_sum = 0

    for cond_id, path in items:
        out_i = os.path.join(args.output_dir, 'id{}'.format(cond_id))
        os.makedirs(out_i, exist_ok=True)
        imgs = load_image_tensor(path, args.input_size).to(device)
        cond = torch.tensor([cond_id], device=device, dtype=torch.long)
        with torch.no_grad():
            gt_indices = model.encode_to_indices(imgs, cond_ids=cond)
            oneshot_pred, gt_check = model.predict_full_mask(imgs, cond_ids=cond)
            assert torch.equal(gt_indices, gt_check)
            vqgan_img = model.decode_indices(gt_indices)
            oneshot_img = model.decode_indices(oneshot_pred)
            iter_img, iter_pred = gen_image(
                model, bsz=1, seed=0, num_iter=args.num_iter,
                choice_temperature=0.0, greedy=True, cond_ids=cond)

        n_tokens = int(gt_indices.numel())
        oneshot_correct = int((oneshot_pred == gt_indices).sum().item())
        iter_correct = int((iter_pred == gt_indices).sum().item())
        oneshot_acc = oneshot_correct / float(n_tokens)
        iter_acc = iter_correct / float(n_tokens)
        mse_vq_in, psnr_vq_in = mse_psnr(vqgan_img, imgs)
        mse_os_vq, psnr_os_vq = mse_psnr(oneshot_img, vqgan_img)
        mse_it_vq, psnr_it_vq = mse_psnr(iter_img, vqgan_img)
        perfect = oneshot_correct == n_tokens
        all_perfect = all_perfect and perfect
        oneshot_correct_sum += oneshot_correct
        iter_correct_sum += iter_correct
        n_tokens_sum += n_tokens

        row = {
            'id': cond_id,
            'path': path,
            'n_tokens': n_tokens,
            'oneshot_token_correct': oneshot_correct,
            'oneshot_token_acc': oneshot_acc,
            'iterative_token_correct': iter_correct,
            'iterative_token_acc': iter_acc,
            'mse_vqgan_vs_input': mse_vq_in,
            'psnr_vqgan_vs_input': json_float(psnr_vq_in),
            'mse_oneshot_vs_vqgan': mse_os_vq,
            'psnr_oneshot_vs_vqgan': json_float(psnr_os_vq),
            'mse_iterative_vs_vqgan': mse_it_vq,
            'psnr_iterative_vs_vqgan': json_float(psnr_it_vq),
            'perfect': perfect,
            'psnr_oneshot_vs_vqgan_is_inf': math.isinf(psnr_os_vq),
        }
        per_image.append(row)

        input_u8 = to_uint8(imgs)
        vq_u8 = to_uint8(vqgan_img)
        os_u8 = to_uint8(oneshot_img)
        it_u8 = to_uint8(iter_img)
        diff_os_u8 = np.abs(os_u8.astype(np.int16) - vq_u8.astype(np.int16)).astype(np.uint8)
        diff_it_u8 = np.abs(it_u8.astype(np.int16) - vq_u8.astype(np.int16)).astype(np.uint8)
        labels = [
            'id{} input'.format(cond_id),
            'vqgan',
            'oneshot',
            'iterative',
            '|oneshot-vqgan|',
            '|iter-vqgan|',
        ]
        save_compare(
            [input_u8, vq_u8, os_u8, it_u8, diff_os_u8, diff_it_u8],
            labels,
            os.path.join(out_i, 'compare.png'))
        PIL.Image.fromarray(input_u8).save(os.path.join(out_i, 'input_256.png'))
        PIL.Image.fromarray(vq_u8).save(os.path.join(out_i, 'vqgan_recon.png'))
        PIL.Image.fromarray(os_u8).save(os.path.join(out_i, 'mage_oneshot.png'))
        PIL.Image.fromarray(it_u8).save(os.path.join(out_i, 'mage_iterative.png'))
        montage_rows.append(np.array(PIL.Image.open(os.path.join(out_i, 'compare.png'))))

    montage = np.concatenate(montage_rows, axis=0)
    PIL.Image.fromarray(montage).save(os.path.join(args.output_dir, 'compare.png'))

    metrics = {
        'n_images': len(per_image),
        'n_tokens': n_tokens_sum,
        'oneshot_token_correct': oneshot_correct_sum,
        'oneshot_token_acc': oneshot_correct_sum / float(max(n_tokens_sum, 1)),
        'iterative_token_correct': iter_correct_sum,
        'iterative_token_acc': iter_correct_sum / float(max(n_tokens_sum, 1)),
        'perfect': all_perfect,
        'per_image': per_image,
        'success_rule': (
            'each image oneshot_token_acc == 1.0 (256/256) with cond_id; '
            'pixels vs that image VQGAN recon'
        ),
    }
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
        f.write('\n')

    print(json.dumps(metrics, indent=2))
    print('wrote', metrics_path)
    if not all_perfect:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
