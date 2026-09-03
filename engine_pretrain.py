import math
import sys
import time
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def _cuda_sync(enabled):
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()


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
    profile = bool(getattr(args, 'profile', False))
    prof = {'data': 0.0, 'h2d': 0.0, 'forward': 0.0, 'backward_optim': 0.0, 'steps': 0}
    last_end = None

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if profile:
            now = time.perf_counter()
            if last_end is not None:
                prof['data'] += now - last_end

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        if profile:
            _cuda_sync(True)
            t0 = time.perf_counter()
        samples = samples.to(device, non_blocking=True)
        if profile:
            _cuda_sync(True)
            prof['h2d'] += time.perf_counter() - t0
            t0 = time.perf_counter()

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples)

        if profile:
            _cuda_sync(True)
            prof['forward'] += time.perf_counter() - t0
            t0 = time.perf_counter()

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        if profile:
            prof['backward_optim'] += time.perf_counter() - t0
            prof['steps'] += 1
            last_end = time.perf_counter()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if profile:
        stats.update(prof)
        if epoch == 0 or (epoch + 1) % 200 == 0:
            steps = max(int(prof['steps']), 1)
            parts = ['{}={:.3f}s'.format(k, prof[k]) for k in ('data', 'h2d', 'forward', 'backward_optim')]
            print('profile epoch {} steps {} {} ({:.1f} ms/step)'.format(
                epoch, steps, ' '.join(parts),
                1000.0 * (prof['data'] + prof['h2d'] + prof['forward'] + prof['backward_optim']) / steps),
                flush=True)
    return stats
