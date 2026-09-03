"""Phase 1 (section 4.1) shared recipe: 4 heads/smoothing cells on each n.

Only ``linear_head`` and ``label_smoothing`` change across the four cells.
Everything else is identical for a given n. Mask stays paper 0.5-1
(phase 3 compares 0-1). Augmentation is off so VQGAN tokens match eval.
"""
import json
import os

# Indices 1-16 stay n=1,2,10,100. 17-20 = n=1000, 21-24 = n=10000.
PHASE1_NS = (1, 2, 10, 100, 1000, 10000)
COMBOS = (
    ('dot_smooth', False, 0.1),
    ('dot_nosmooth', False, 0.0),
    ('lin_smooth', True, 0.1),
    ('lin_nosmooth', True, 0.0),
)

# Absolute step size, not blr, so changing batch with n does not silently
# rescale lr inside one n's four-cell table.
RECIPE = {
    'model': 'mage_vit_base_patch16',
    'no_aug': True,
    'mask_ratio_min': 0.5,
    'mask_ratio_max': 1.0,
    'mask_ratio_mu': 0.55,
    'mask_ratio_std': 0.25,
    'lr': 1.0e-4,
    'min_lr': 1.0e-6,
    'weight_decay': 0.05,
    'warmup_epochs': 200,
    'grad_clip': 3.0,
    'seed': 0,
    'save_last_freq': 200,
    'save_ckpt_freq': 0,
    'log_tb_freq': 10,
    'num_workers': 0,
    'cache_tokens': True,
    'profile': True,
    'version': 'v2',
    'archive_v1': 'outputs/_archive/phase1_v1',
    'epochs_by_n': {
        '1': 8000, '2': 8000, '10': 8000, '100': 8000,
        '1000': 8000, '10000': 8000,
    },
    'batch_size_by_n': {
        '1': 1, '2': 2, '10': 10, '100': 20,
        '1000': 50, '10000': 100,
    },
    'data_path': 'data/tiny-imagenet-200',
    'vqgan_ckpt': 'vqgan_jax_strongaug.ckpt',
    'recon_dir': 'outputs/_ref/tiny_val256_vqgan_recon',
    'ref_metrics': 'outputs/_ref/metrics.json',
    'star50000': 'not run; phase-1 slice is Tiny-ImageNet n=1,2,10,100,1000,10000',
    'note': (
        'Shared across the four cells except classification head and label '
        'smoothing. Official mask 0.5-1 and 50% encoder drop stay as in the '
        'paper. --no_aug so train tokens equal eval tokens. Absolute --lr, '
        'not --blr. v2: 8000 epochs for every n (equal visits/image), '
        'warmup 200 (2.5%), min_lr 1e-6 so cosine does not die, '
        '--cache_tokens skips per-step VQGAN encode. Batch grows with n '
        '(50 at n=1000, 100 at n=10000) so the dataset stays divisible; '
        'lr does not scale with batch. Tiny 64→256 FID is not the paper '
        'ImageNet-256 number. v1 (2000/400 epochs, min_lr 0) is in '
        'outputs/_archive/phase1_v1.'
    ),
}


def combo_by_name(name):
    for exp_name, linear_head, label_smoothing in COMBOS:
        if exp_name == name:
            return exp_name, linear_head, label_smoothing
    raise KeyError('unknown phase1 combo: {}'.format(name))


def job_from_index(index):
    """1-based LSF array index 1..24 -> (n, exp_name, linear_head, label_smoothing)."""
    index = int(index)
    n_jobs = len(PHASE1_NS) * len(COMBOS)
    if index < 1 or index > n_jobs:
        raise ValueError('phase1 job index must be 1..{}, got {}'.format(n_jobs, index))
    i0 = index - 1
    n = PHASE1_NS[i0 // len(COMBOS)]
    exp_name, linear_head, label_smoothing = COMBOS[i0 % len(COMBOS)]
    return n, exp_name, linear_head, label_smoothing


def output_dir(exp_name, n):
    return os.path.join('outputs', exp_name, 'n{}'.format(int(n)))


def ckpt_path(exp_name, n):
    return os.path.join(output_dir(exp_name, n), 'checkpoint-last.pth')


def last_logged_epoch(exp_name, n, root='.'):
    """Last epoch written to log.txt, or None if the log is missing/empty."""
    path = os.path.join(root, output_dir(exp_name, n), 'log.txt')
    last = None
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if 'epoch' in row:
                last = int(row['epoch'])
    return last


def train_finished(index, root='.'):
    """True only after the recipe's last epoch is in log.txt.

    A mid-run checkpoint-last.pth is not enough: fill must resume train,
    not eval, if the job dies before epoch epochs-1.
    """
    n, exp_name, _, _ = job_from_index(index)
    last = last_logged_epoch(exp_name, n, root=root)
    if last is None:
        return False
    return last >= int(train_settings(n)['epochs']) - 1


def metrics_path(exp_name, n, schedule='cosine'):
    name = 'metrics_uncover1.json' if schedule == 'uncover1' else 'metrics.json'
    return os.path.join(output_dir(exp_name, n), name)


def table_dir(n):
    return os.path.join('outputs', 'phase1', 'n{}'.format(int(n)))


def table_stem(schedule='cosine'):
    return 'table_uncover1' if schedule == 'uncover1' else 'table'


def train_settings(n):
    n = int(n)
    key = str(n)
    return {
        'epochs': int(RECIPE['epochs_by_n'][key]),
        'batch_size': int(RECIPE['batch_size_by_n'][key]),
        'lr': float(RECIPE['lr']),
        'min_lr': float(RECIPE['min_lr']),
        'warmup_epochs': int(RECIPE['warmup_epochs']),
        'weight_decay': float(RECIPE['weight_decay']),
        'save_last_freq': int(RECIPE['save_last_freq']),
        'log_tb_freq': int(RECIPE['log_tb_freq']),
        'cache_tokens': bool(RECIPE['cache_tokens']),
        'profile': bool(RECIPE['profile']),
        'mask_ratio_min': float(RECIPE['mask_ratio_min']),
        'mask_ratio_max': float(RECIPE['mask_ratio_max']),
        'mask_ratio_mu': float(RECIPE['mask_ratio_mu']),
        'mask_ratio_std': float(RECIPE['mask_ratio_std']),
        'seed': int(RECIPE['seed']),
        'save_last_freq': int(RECIPE['save_last_freq']),
        'num_workers': int(RECIPE['num_workers']),
        'model': RECIPE['model'],
        'no_aug': bool(RECIPE['no_aug']),
        'data_path': RECIPE['data_path'],
        'image_list': 'splits/n{}.txt'.format(n),
    }


def write_recipe_json(path):
    payload = {
        'phase': 1,
        'section': '4.1',
        'n': list(PHASE1_NS),
        'combos': [
            {
                'exp_name': name,
                'linear_head': linear_head,
                'label_smoothing': label_smoothing,
                'head': 'linear' if linear_head else 'dot',
            }
            for name, linear_head, label_smoothing in COMBOS
        ],
        'recipe': RECIPE,
        'job_array': 'scripts/phase1_fill_pack.py packs 1-{}'.format(
            len(PHASE1_NS) * len(COMBOS)),
    }
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
        f.write('\n')
    return path


def shell_exports(index=None, n=None, exp_name=None):
    """Print VAR=value lines for the bsub script."""
    if index is not None:
        n, exp_name, linear_head, label_smoothing = job_from_index(index)
    else:
        exp_name, linear_head, label_smoothing = combo_by_name(exp_name)
        n = int(n)
    settings = train_settings(n)
    lines = [
        'N={}'.format(n),
        'EXP={}'.format(exp_name),
        'LINEAR_HEAD={}'.format(1 if linear_head else 0),
        'LABEL_SMOOTHING={}'.format(label_smoothing),
        'EPOCHS={}'.format(settings['epochs']),
        'BATCH_SIZE={}'.format(settings['batch_size']),
        'LR={}'.format(settings['lr']),
        'MIN_LR={}'.format(settings['min_lr']),
        'WARMUP_EPOCHS={}'.format(settings['warmup_epochs']),
        'SAVE_LAST_FREQ={}'.format(settings['save_last_freq']),
        'LOG_TB_FREQ={}'.format(settings['log_tb_freq']),
        'CACHE_TOKENS={}'.format(1 if settings['cache_tokens'] else 0),
        'PROFILE={}'.format(1 if settings['profile'] else 0),
        'OUT_DIR={}'.format(output_dir(exp_name, n)),
        'CKPT={}'.format(ckpt_path(exp_name, n)),
        'IMAGE_LIST={}'.format(settings['image_list']),
        'RECON_DIR={}'.format(RECIPE['recon_dir']),
    ]
    return '\n'.join(lines) + '\n'
