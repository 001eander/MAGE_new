import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        cond_ids = None
        if getattr(args, 'cond_ids', False):
            cond_ids = targets.to(device, non_blocking=True)

        if getattr(args, 'no_amp', False):
            loss, acc, _ = model(samples, cond_ids)
        else:
            with torch.cuda.amp.autocast():
                loss, acc, _ = model(samples, cond_ids)

        loss_value = loss.item()
        acc_value = acc.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter
        if getattr(args, 'no_amp', False):
            loss.backward()
            if (data_iter_step + 1) % accum_iter == 0:
                if args.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
        else:
            loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value, acc=acc_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        acc_value_reduce = misc.all_reduce_mean(acc_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_acc', acc_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def eval_reproduction_token_acc(model, data_loader, device, num_iter=12, max_batches=None):
    """Eval-mode oneshot + iterative greedy token accuracy."""
    from gen_img_uncond import gen_image

    was_training = model.training
    model.eval()
    raw = model.module if hasattr(model, 'module') else model
    oneshot_correct = 0
    iterative_correct = 0
    total = 0
    use_cond = getattr(raw, 'use_cond_ids', False)
    for i, (samples, labels) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        cond_ids = labels.to(device, non_blocking=True) if use_cond else None
        pred, gt = raw.predict_full_mask(samples, cond_ids=cond_ids)
        oneshot_correct += int((pred == gt).sum().item())
        _, iter_pred = gen_image(
            raw, bsz=samples.size(0), seed=0, num_iter=num_iter,
            choice_temperature=0.0, greedy=True, cond_ids=cond_ids)
        iterative_correct += int((iter_pred == gt).sum().item())
        total += int(gt.numel())
        if max_batches is not None and i + 1 >= max_batches:
            break
    if was_training:
        model.train(True)
    n_tokens = max(total, 1)
    return {
        'oneshot_acc': oneshot_correct / float(n_tokens),
        'oneshot_correct': oneshot_correct,
        'iterative_acc': iterative_correct / float(n_tokens),
        'iterative_correct': iterative_correct,
        'n_tokens': total,
    }