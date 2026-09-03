#!/usr/bin/env python3
"""Submit phase-1 cells without exceeding the user's LSF job cap (MAX=5)."""
from __future__ import print_function

import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from util.phase1 import COMBOS, PHASE1_NS, RECIPE, job_from_index, metrics_path  # noqa: E402

BSUB = '/nfsshare/lsf/10.1/linux3.10-glibc2.17-x86_64/bin/bsub'
BJOBS = '/nfsshare/lsf/10.1/linux3.10-glibc2.17-x86_64/bin/bjobs'
USER_MAX_JOBS = 5
PHASE1_NAME_PREFIX = 'p1_'
PREP_METRICS = os.path.join(ROOT, RECIPE['ref_metrics'])
PREP_RECON = os.path.join(ROOT, RECIPE['recon_dir'])
PREP_RECON_MIN = 10000
LOG = os.path.join(ROOT, 'outputs', 'lsf', 'phase1_fill_queue.log')


def log(msg):
    line = time.strftime('%Y-%m-%d %H:%M:%S ') + msg
    print(line, flush=True)


def run(cmd):
    return subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True)


def all_indices():
    return list(range(1, len(PHASE1_NS) * len(COMBOS) + 1))


def cell_done(index):
    n, exp_name, _, _ = job_from_index(index)
    path = os.path.join(ROOT, metrics_path(exp_name, n))
    if not os.path.isfile(path):
        return False
    try:
        with open(path) as f:
            m = json.load(f)
    except Exception:
        return False
    fid = (m.get('fid') or {}).get('generated_vs_ref')
    oneshot = m.get('oneshot_vs_self')
    grid = m.get('grid')
    return fid is not None and oneshot is not None and bool(grid) and os.path.isfile(
        os.path.join(ROOT, grid))


def user_jobs():
    out = run([BJOBS, '-w'])
    text = out.stdout.strip()
    if out.returncode != 0 or not text or 'No unfinished' in text:
        return []
    jobs = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 7:
            continue
        jobs.append({'id': parts[0], 'stat': parts[2], 'name': parts[6]})
    return jobs


def phase1_running_indices(jobs):
    indices = set()
    for job in jobs:
        name = job['name']
        if name.startswith(PHASE1_NAME_PREFIX):
            try:
                indices.add(int(name[len(PHASE1_NAME_PREFIX):]))
            except ValueError:
                pass
    return indices


def submit(index):
    n, exp_name, _, _ = job_from_index(index)
    env = os.environ.copy()
    env['INDEX'] = str(index)
    env['LSB_NTRIES'] = '1'
    job_name = '{}{}'.format(PHASE1_NAME_PREFIX, index)
    script = os.path.join(ROOT, 'scripts', 'phase1_one.sh')
    with open(script, 'rb') as fh:
        proc = subprocess.run(
            ['timeout', '15', BSUB, '-J', job_name],
            stdin=fh, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, env=env, cwd=ROOT)
    msg = ((proc.stdout or '') + ' ' + (proc.stderr or '')).strip()
    if proc.returncode != 0:
        log('submit {} n={} {} FAILED: {}'.format(index, n, exp_name, msg))
        return False
    log('submit {} n={} {} -> {}'.format(index, n, exp_name, msg))
    return True


def rebuild_tables():
    proc = run([
        os.path.expanduser('~/miniforge3/envs/mage/bin/python'),
        os.path.join(ROOT, 'scripts', 'make_phase1_table.py'),
    ])
    log('tables rc={} {}'.format(proc.returncode, (proc.stdout or '').strip()))


def once():
    n_recon = 0
    if os.path.isdir(PREP_RECON):
        n_recon = len([
            name for name in os.listdir(PREP_RECON)
            if name.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
    if n_recon < PREP_RECON_MIN and not os.path.isfile(PREP_METRICS):
        log('wait prep recon={}/{} metrics={}'.format(
            n_recon, PREP_RECON_MIN, os.path.isfile(PREP_METRICS)))
        return False
    jobs = user_jobs()
    n_jobs = len(jobs)
    running = phase1_running_indices(jobs)
    pend_phase1 = [j for j in jobs if j['name'].startswith(PHASE1_NAME_PREFIX)
                   and j['stat'] == 'PEND']
    pending = [i for i in all_indices() if (not cell_done(i)) and i not in running]
    slots = max(USER_MAX_JOBS - n_jobs, 0)
    if pend_phase1:
        slots = 0
    log('jobs={} phase1_running={} remaining={} slots={} pend={}'.format(
        n_jobs, sorted(running), pending, slots,
        [j['name'] for j in pend_phase1]))
    if not pending and not running:
        rebuild_tables()
        log('all 16 cells have metrics')
        return True
    if not pending and running:
        log('waiting for in-flight cells {}'.format(sorted(running)))
        return False
    submitted = 0
    for index in pending:
        if submitted >= slots:
            break
        if submit(index):
            submitted += 1
            n_jobs += 1
            slots = max(USER_MAX_JOBS - n_jobs, 0)
        else:
            break
    return False


def main():
    # v2: pack several cells onto one multi-GPU job when a V100 node has free cards.
    import runpy
    runpy.run_path(os.path.join(ROOT, 'scripts', 'phase1_fill_pack.py'), run_name='__main__')


if __name__ == '__main__':
    main()
