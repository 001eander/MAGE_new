import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import PIL.Image
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mage

from engine_pretrain import train_one_epoch


def encode_dataset_tokens(model, dataset, device, batch_size):
    """Walk dataset index order and cache VQGAN codebook ids."""
    model.eval()
    rows = []
    n_total = len(dataset)
    # Train batch can be 100; VQGAN encode of 100×256 hung on V100. Cap cache.
    enc_bs = min(int(batch_size), 32)
    with torch.no_grad():
        for start in range(0, n_total, enc_bs):
            end = min(start + enc_bs, n_total)
            batch = torch.stack([dataset[i][0] for i in range(start, end)], dim=0)
            batch = batch.to(device, non_blocking=True)
            z_q, _, token_tuple = model.vqgan.encode(batch)
            _, _, token_indices = token_tuple
            rows.append(token_indices.reshape(z_q.size(0), -1).detach().cpu().long())
            if end == n_total or (end // 500) != (start // 500):
                print('cache_tokens', end, '/', n_total, flush=True)
    model.train()
    return torch.cat(rows, dim=0)


def rebuild_loader(dataset, args):
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    return sampler, loader


def get_args_parser():
    parser = argparse.ArgumentParser('MAGE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mage_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=256, type=int,
                        help='images input size')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # MAGE params
    parser.add_argument('--mask_ratio_min', type=float, default=0.5,
                        help='Minimum mask ratio')
    parser.add_argument('--mask_ratio_max', type=float, default=1.0,
                        help='Maximum mask ratio')
    parser.add_argument('--mask_ratio_mu', type=float, default=0.55,
                        help='Mask ratio distribution peak')
    parser.add_argument('--mask_ratio_std', type=float, default=0.25,
                        help='Mask ratio distribution std')
    parser.add_argument('--grad_clip', type=float, default=3.0,
                        help='Gradient clip')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Token CE label smoothing; set 0 for memorization')
    parser.add_argument('--linear_head', action='store_true',
                        help='Independent Linear(decoder_dim, 1024) head instead of tied MlmLayer')
    parser.add_argument('--save_last_freq', default=1, type=int,
                        help='Write checkpoint-last.pth every N epochs (always on the final epoch)')
    parser.add_argument('--save_ckpt_freq', default=40, type=int,
                        help='Write checkpoint-<epoch>.pth every N epochs; 0 disables numbered checkpoints')
    parser.add_argument('--log_tb_freq', default=1, type=int,
                        help='Write TensorBoard scalars every N epochs, using the epoch '
                             'index as the step (always on the final epoch); 0 disables')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--max_samples', default=None, type=int,
                        help='Use only the first N training images (for smoke tests)')
    parser.add_argument('--image_list', default='', type=str,
                        help='Frozen relative-path list (e.g. splits/n2.txt). Paths are relative to --data_path.')
    parser.add_argument('--no_aug', action='store_true',
                        help='Deterministic Resize+CenterCrop (for memorization); default is random crop and flip')
    parser.add_argument('--cache_tokens', action='store_true',
                        help='Encode the train set with VQGAN once and train on codebook ids. Requires --no_aug.')
    parser.add_argument('--profile', action='store_true',
                        help='Time data / H2D / forward / backward each epoch; write profile.json')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if args.no_aug:
        transform_train = transforms.Compose([
            transforms.Resize(args.input_size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
        ])
        print("Using deterministic Resize+CenterCrop")
    else:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    if args.image_list:
        from util.exp_data import ImageListDataset, read_list
        rel_paths = read_list(args.image_list)
        dataset_train = ImageListDataset(
            rel_paths, source_root=args.data_path, transform=transform_train)
        print('Using frozen image list', args.image_list, 'n={}'.format(len(dataset_train)))
    else:
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
        if args.max_samples is not None:
            n = min(int(args.max_samples), len(dataset_train))
            dataset_train = torch.utils.data.Subset(dataset_train, list(range(n)))
    print(dataset_train)
    if len(dataset_train) % args.batch_size != 0:
        raise ValueError(
            "dataset size ({}) must be divisible by batch_size ({})".format(
                len(dataset_train), args.batch_size
            )
        )

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    
    # define the model
    vqgan_ckpt_path = 'vqgan_jax_strongaug.ckpt'

    model = models_mage.__dict__[args.model](mask_ratio_mu=args.mask_ratio_mu, mask_ratio_std=args.mask_ratio_std,
                                             mask_ratio_min=args.mask_ratio_min, mask_ratio_max=args.mask_ratio_max,
                                             label_smoothing=args.label_smoothing,
                                             linear_head=args.linear_head,
                                             vqgan_ckpt_path=vqgan_ckpt_path)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if getattr(args, 'cache_tokens', False):
        if not args.no_aug:
            print('cache_tokens requires --no_aug; skipping cache')
        else:
            cache_path = ''
            if args.output_dir:
                cache_path = os.path.join(args.output_dir, 'cached_train_tokens.pt')
            tokens = None
            if cache_path and os.path.isfile(cache_path):
                loaded = torch.load(cache_path, map_location='cpu')
                if (torch.is_tensor(loaded)
                        and loaded.dim() == 2
                        and loaded.size(0) == len(dataset_train)
                        and loaded.size(1) == 256):
                    tokens = loaded.long()
                    print('reused cached tokens', tuple(tokens.shape),
                          cache_path, flush=True)
                else:
                    print('cached tokens mismatch, recaching', flush=True)
            if tokens is None:
                print('caching VQGAN tokens once (no_aug, frozen list)')
                t0 = time.time()
                tokens = encode_dataset_tokens(
                    model_without_ddp, dataset_train, device, args.batch_size)
                print('cached tokens', tuple(tokens.shape),
                      'in {:.1f}s'.format(time.time() - t0), flush=True)
                if cache_path and misc.is_main_process():
                    torch.save(tokens, cache_path)
            from util.exp_data import TokenIndexDataset
            dataset_train = TokenIndexDataset(tokens)
            sampler_train, data_loader_train = rebuild_loader(dataset_train, args)

    if args.output_dir and misc.is_main_process():
        args_path = os.path.join(args.output_dir, 'args.json')
        with open(args_path, 'w') as f:
            json.dump(vars(args), f, indent=2, default=str)
            f.write('\n')
        print('wrote', args_path)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    profile_sum = {}
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        save_ckpt_freq = int(getattr(args, 'save_ckpt_freq', 40))
        if args.output_dir and save_ckpt_freq > 0 and (
                epoch % save_ckpt_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        save_last_freq = max(int(getattr(args, 'save_last_freq', 1)), 1)
        if args.output_dir and ((epoch + 1) % save_last_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model_last(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}
        if getattr(args, 'profile', False):
            for key in ('data', 'h2d', 'forward', 'backward_optim', 'steps'):
                if key in train_stats:
                    profile_sum[key] = profile_sum.get(key, 0.0) + float(train_stats[key])

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_tb_freq = int(getattr(args, 'log_tb_freq', 1))
                if log_tb_freq > 0 and (
                        (epoch + 1) % log_tb_freq == 0 or epoch + 1 == args.epochs):
                    log_writer.add_scalar('train_loss', train_stats['loss'], epoch)
                    log_writer.add_scalar('lr', train_stats['lr'], epoch)
                    if getattr(args, 'profile', False):
                        for key in ('data', 'h2d', 'forward', 'backward_optim'):
                            if key in train_stats:
                                log_writer.add_scalar('profile/' + key, train_stats[key], epoch)
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if getattr(args, 'profile', False) and args.output_dir and misc.is_main_process():
        steps = max(float(profile_sum.get('steps', 0.0)), 1.0)
        compute = profile_sum.get('forward', 0.0) + profile_sum.get('backward_optim', 0.0)
        dataish = profile_sum.get('data', 0.0) + profile_sum.get('h2d', 0.0)
        notes = []
        if not getattr(args, 'cache_tokens', False):
            notes.append('enable --cache_tokens with --no_aug to skip per-step VQGAN encode')
        if dataish > 0.25 * max(compute + dataish, 1e-6):
            notes.append('data/H2D is a large slice; cached tokens plus pin_memory help more than num_workers on n<=100')
        if compute > 0:
            notes.append('remaining time is encoder/decoder/optimizer; batch size is the next lever')
        payload = {
            'seconds': profile_sum,
            'ms_per_step': {
                k: 1000.0 * profile_sum.get(k, 0.0) / steps
                for k in ('data', 'h2d', 'forward', 'backward_optim')
            },
            'train_wall_s': total_time,
            'cache_tokens': bool(getattr(args, 'cache_tokens', False)),
            'notes': notes,
        }
        path = os.path.join(args.output_dir, 'profile.json')
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)
            f.write('\n')
        print('wrote', path, payload['ms_per_step'])


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    main(args)
