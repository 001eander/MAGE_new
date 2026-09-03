#!/usr/bin/env python3
"""Build the four-cell phase-1 table for one n, or for every phase-1 n."""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from util.phase1 import (  # noqa: E402
    COMBOS,
    PHASE1_NS,
    RECIPE,
    metrics_path,
    table_dir,
    table_stem,
    write_recipe_json,
)


def load_metrics(path):
    with open(path) as f:
        return json.load(f)


def cell_row(exp_name, linear_head, label_smoothing, n, schedule='cosine'):
    path = metrics_path(exp_name, n, schedule=schedule)
    row = {
        'exp_name': exp_name,
        'head': 'linear' if linear_head else 'dot',
        'linear_head': linear_head,
        'label_smoothing': label_smoothing,
        'metrics_path': path,
        'present': os.path.isfile(path),
    }
    if not row['present']:
        return row
    m = load_metrics(path)
    oneshot = m.get('oneshot_vs_self') or {}
    wrong = m.get('wrong_token_embed') or {}
    fid = m.get('fid') or {}
    cov = m.get('coverage') or {}
    match = m.get('match_error') or {}
    row.update({
        'oneshot_token_acc': oneshot.get('token_acc'),
        'wrong_cosine_distance': wrong.get('wrong_cosine_distance'),
        'wrong_euclidean': wrong.get('wrong_euclidean'),
        'random_wrong_cosine_distance': wrong.get('random_wrong_cosine_distance'),
        'random_wrong_euclidean': wrong.get('random_wrong_euclidean'),
        'fid_generated_vs_ref': (fid or {}).get('generated_vs_ref'),
        'fid_vqgan_recon_vs_ref': (fid or {}).get('vqgan_recon_vs_ref'),
        'coverage_exact': cov.get('exact'),
        'coverage_approx': cov.get('approx'),
        'match_token_frac': match.get('token_frac'),
        'match_pixel_mse_vqgan': match.get('pixel_mse_vqgan'),
        'grid': m.get('grid'),
        'generation': m.get('generation'),
    })
    return row


def complete(row):
    if not row.get('present'):
        return False
    needed = (
        'oneshot_token_acc',
        'fid_generated_vs_ref',
        'fid_vqgan_recon_vs_ref',
        'coverage_exact',
        'coverage_approx',
        'match_token_frac',
        'match_pixel_mse_vqgan',
        'grid',
    )
    for key in needed:
        if row.get(key) is None:
            return False
    # Distances may be null only when every token was correct.
    if row.get('oneshot_token_acc') != 1.0:
        if row.get('wrong_cosine_distance') is None or row.get('wrong_euclidean') is None:
            return False
        if row.get('random_wrong_cosine_distance') is None:
            return False
    if not row.get('grid') or not os.path.isfile(os.path.join(ROOT, row['grid'])):
        return False
    return True


def write_table_md(n, rows, dest):
    lines = [
        '# Phase 1 n={}'.format(n),
        '',
        'Shared settings except head and label smoothing. See `splits/phase1_recipe.json`.',
        'Generation schedule is recorded in each metrics file (`generation.schedule`).',
        '',
        '| combo | head | LS | oneshot acc | wrong cos | wrong L2 | rand cos | rand L2 | FID gen | FID VQGAN | cover exact | cover approx | match token | match pixel | grid |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for row in rows:
        def fmt(key):
            val = row.get(key)
            if val is None:
                return '—' if row.get('present') else 'missing'
            if isinstance(val, float):
                return '{:.4f}'.format(val)
            return str(val)
        lines.append(
            '| {exp} | {head} | {ls} | {acc} | {wc} | {wl} | {rc} | {rl} | {fg} | {fr} | {ce} | {ca} | {mt} | {mp} | {grid} |'.format(
                exp=row['exp_name'],
                head=row['head'],
                ls=row['label_smoothing'],
                acc=fmt('oneshot_token_acc'),
                wc=fmt('wrong_cosine_distance'),
                wl=fmt('wrong_euclidean'),
                rc=fmt('random_wrong_cosine_distance'),
                rl=fmt('random_wrong_euclidean'),
                fg=fmt('fid_generated_vs_ref'),
                fr=fmt('fid_vqgan_recon_vs_ref'),
                ce=fmt('coverage_exact'),
                ca=fmt('coverage_approx'),
                mt=fmt('match_token_frac'),
                mp=fmt('match_pixel_mse_vqgan'),
                grid=row.get('grid') or '—',
            )
        )
    lines.append('')
    with open(dest, 'w') as f:
        f.write('\n'.join(lines))
    return dest


def build_one(n, schedule='cosine'):
    rows = [cell_row(name, lh, ls, n, schedule=schedule) for name, lh, ls in COMBOS]
    dest_dir = os.path.join(ROOT, table_dir(n))
    os.makedirs(dest_dir, exist_ok=True)
    stem = table_stem(schedule)
    payload = {
        'n': int(n),
        'section': '4.1',
        'schedule': schedule,
        'star50000': RECIPE['star50000'],
        'complete': all(complete(row) for row in rows),
        'rows': rows,
    }
    json_path = os.path.join(dest_dir, '{}.json'.format(stem))
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)
        f.write('\n')
    md_path = write_table_md(n, rows, os.path.join(dest_dir, '{}.md'.format(stem)))
    return payload, json_path, md_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=0, help='0 = all phase-1 n')
    parser.add_argument('--schedule', type=str, default='cosine',
                        choices=('cosine', 'uncover1'))
    args = parser.parse_args()
    os.chdir(ROOT)
    write_recipe_json(os.path.join(ROOT, 'splits', 'phase1_recipe.json'))
    note_path = os.path.join(ROOT, 'splits', 'phase1_star50000.txt')
    with open(note_path, 'w') as f:
        f.write(RECIPE['star50000'] + '\n')
    os.makedirs(os.path.join(ROOT, 'outputs', 'phase1'), exist_ok=True)
    with open(os.path.join(ROOT, 'outputs', 'phase1', 'star50000.txt'), 'w') as f:
        f.write(RECIPE['star50000'] + '\n')
    ns = PHASE1_NS if args.n == 0 else (args.n,)
    all_ok = True
    for n in ns:
        payload, json_path, md_path = build_one(n, schedule=args.schedule)
        print(n, 'complete' if payload['complete'] else 'incomplete', json_path, md_path)
        all_ok = all_ok and payload['complete']
    if not all_ok:
        sys.exit(2)


if __name__ == '__main__':
    main()
