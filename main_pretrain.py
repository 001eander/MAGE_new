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

from engine_pretrain import train_one_epoch, eval_reproduction_token_acc


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
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Transformer / token-embedding dropout; set 0 for memorization')
    parser.add_argument('--fixed_mask_ratio', type=float, default=None,
                        help='If set, use this mask ratio every step (1.0 = full-mask, matches generation)')
    parser.add_argument('--full_mask_prob', type=float, default=0.0,
                        help='Probability of training on a fully-masked sample (generation start)')
    parser.add_argument('--gen_schedule_prob', type=float, default=0.0,
                        help='Probability of using the iterative-generation mask lengths')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable AMP (more stable for exact token memorization)')
    parser.add_argument('--cache_tokens', action='store_true',
                        help='Encode VQGAN tokens once (single-image memorization)')
    parser.add_argument('--repeat', type=int, default=1,
                        help='Repeat the training set this many times per epoch')
    parser.add_argument('--eval_num_iter', type=int, default=12,
                        help='Iterative greedy steps when evaluating generation token acc')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='Save checkpoint-last every N epochs (always saved at the last epoch)')
    parser.add_argument('--save_ckpt_freq', type=int, default=40,
                        help='Save numbered checkpoint-N every N epochs; 0 disables')
    parser.add_argument('--stop_token_acc', type=float, default=None,
                        help='Stop when oneshot and iterative token acc both reach this (e.g. 1.0)')
    parser.add_argument('--refine_lr', type=float, default=None,
                        help='After oneshot acc reaches --refine_after, set lr and min_lr to this')
    parser.add_argument('--refine_after', type=float, default=0.9,
                        help='Oneshot acc that triggers --refine_lr')
    parser.add_argument('--phase2_after_oneshot', action='store_true',
                        help='After oneshot hits --stop_token_acc, switch to gen-schedule training')
    parser.add_argument('--require_iterative', action='store_true',
                        help='Also require iterative token acc before early stop')
    parser.add_argument('--decoder_mask_token', action='store_true',
                        help='Fill masked decoder slots with mask_token+pos instead of CLS')
    parser.add_argument('--linear_head', action='store_true',
                        help='Use a Linear codebook classifier instead of tied MLM embeddings')
    parser.add_argument('--cond_ids', action='store_true',
                        help='Use ImageFolder class id as the class token (codebook_size + id)')
    parser.add_argument('--eval_freq', type=int, default=1,
                        help='Run full-mask token-acc eval every N epochs when --stop_token_acc is set')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--max_samples', default=None, type=int,
                        help='Use only the first N training images (for smoke tests)')
    parser.add_argument('--no_aug', action='store_true',
                        help='Deterministic Resize+CenterCrop (for memorization); default is random crop and flip')

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
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    if args.max_samples is not None:
        n = min(int(args.max_samples), len(dataset_train))
        dataset_train = torch.utils.data.Subset(dataset_train, list(range(n)))
    dataset_unique = dataset_train
    if args.repeat > 1:
        dataset_train = torch.utils.data.ConcatDataset([dataset_unique] * int(args.repeat))
    print(dataset_train)
    print("unique images:", len(dataset_unique))
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
    eval_bs = min(args.batch_size, max(1, len(dataset_unique)))
    data_loader_eval = torch.utils.data.DataLoader(
        dataset_unique,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=0,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    
    # define the model
    vqgan_ckpt_path = 'vqgan_jax_strongaug.ckpt'

    model = models_mage.__dict__[args.model](mask_ratio_mu=args.mask_ratio_mu, mask_ratio_std=args.mask_ratio_std,
                                             mask_ratio_min=args.mask_ratio_min, mask_ratio_max=args.mask_ratio_max,
                                             label_smoothing=args.label_smoothing,
                                             dropout=args.dropout,
                                             fixed_mask_ratio=args.fixed_mask_ratio,
                                             full_mask_prob=args.full_mask_prob,
                                             gen_schedule_prob=args.gen_schedule_prob,
                                             gen_num_iter=args.eval_num_iter,
                                             decoder_mask_token=args.decoder_mask_token,
                                             linear_head=args.linear_head,
                                             use_cond_ids=args.cond_ids,
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

    if args.cache_tokens:
        print("Caching VQGAN tokens for memorization")
        model_without_ddp.eval()
        with torch.no_grad():
            cache = {}
            first = None
            for samples, labels in data_loader_eval:
                samples = samples.to(device, non_blocking=True)
                toks = model_without_ddp.vqgan_encode(samples)
                if first is None:
                    first = toks
                for t, lab in zip(toks, labels):
                    cache[int(lab.item())] = t.detach()
        if args.cond_ids:
            model_without_ddp._cached_gt_by_id = cache
            print("cached ids", sorted(cache.keys()),
                  "token shape", tuple(next(iter(cache.values())).shape))
        else:
            model_without_ddp._cached_gt_indices = first
            print("cached token shape", tuple(first.shape))
        model_without_ddp.train()

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        last_epoch = epoch + 1 == args.epochs
        stop = False
        if args.stop_token_acc is not None and ((epoch + 1) % args.eval_freq == 0 or last_epoch):
            eval_stats = eval_reproduction_token_acc(
                model, data_loader_eval, device,
                num_iter=args.eval_num_iter, max_batches=None)
            print("Eval oneshot token acc: {:.4f} ({}/{})  iterative: {:.4f} ({}/{})".format(
                eval_stats['oneshot_acc'], eval_stats['oneshot_correct'], eval_stats['n_tokens'],
                eval_stats['iterative_acc'], eval_stats['iterative_correct'], eval_stats['n_tokens']))
            train_stats['eval_token_acc'] = eval_stats['oneshot_acc']
            train_stats['eval_token_correct'] = eval_stats['oneshot_correct']
            train_stats['eval_token_total'] = eval_stats['n_tokens']
            train_stats['eval_iterative_acc'] = eval_stats['iterative_acc']
            train_stats['eval_iterative_correct'] = eval_stats['iterative_correct']
            oneshot_ok = eval_stats['oneshot_acc'] >= args.stop_token_acc
            iterative_ok = eval_stats['iterative_acc'] >= args.stop_token_acc
            if (args.refine_lr is not None
                    and eval_stats['oneshot_acc'] >= args.refine_after
                    and abs(args.lr - args.refine_lr) > 1e-12):
                print("Refine lr: oneshot {:.4f} >= {:.4f}, {} -> {}".format(
                    eval_stats['oneshot_acc'], args.refine_after, args.lr, args.refine_lr))
                args.lr = args.refine_lr
                args.min_lr = args.refine_lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.refine_lr
            if oneshot_ok and (iterative_ok or not args.require_iterative):
                print("Early stop: oneshot {:.4f} iterative {:.4f} (require_iterative={})".format(
                    eval_stats['oneshot_acc'], eval_stats['iterative_acc'], args.require_iterative))
                stop = True
                last_epoch = True
            elif (args.phase2_after_oneshot and oneshot_ok and
                    not getattr(args, '_phase2_started', False)):
                print("Phase 2: oneshot {:.4f}, switch to gen-schedule masks".format(
                    eval_stats['oneshot_acc']))
                args._phase2_started = True
                model_without_ddp.fixed_mask_ratio = None
                model_without_ddp.full_mask_prob = 0.2
                model_without_ddp.gen_schedule_prob = 0.8
                train_stats['phase2'] = 1

        if args.output_dir and args.save_ckpt_freq > 0 and (
                epoch % args.save_ckpt_freq == 0 or last_epoch):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        if args.output_dir and (
                last_epoch or (args.save_freq > 0 and (epoch + 1) % args.save_freq == 0)):
            misc.save_model_last(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if stop:
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    main(args)
