#!/usr/bin/env python3
"""Submit uncover1 evals for phase-1 cells that already have a checkpoint."""
from __future__ import print_function

import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from util.phase1 import COMBOS, PHASE1_NS, ckpt_path, job_from_index, metrics_path  # noqa: E402

BSUB = '/nfsshare/lsf/10.1/linux3.10-glibc2.17-x86_64/bin/bsub'
BJOBS = '/nfsshare/lsf/10.1/linux3.10-glibc2.17-x86_64/bin/bjobs'
USER_MAX_JOBS = 5
EVAL_PREFIX = 'p1u_'
CKPT_MIN_BYTES = 500 * 1024 * 1024


def log(msg):
    print(time.strftime('%Y-%m-%d %H:%M:%S ') + msg, flush=True)


def run(cmd):
    return subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True)


def all_indices():
    return list(range(1, len(PHASE1_NS) * len(COMBOS) + 1))


def cell_done(index):
    n, exp_name, _, _ = job_from_index(index)
    path = os.path.join(ROOT, metrics_path(exp_name, n, schedule='uncover1'))
    if not os.path.isfile(path):
        return False
    try:
        with open(path) as f:
            m = json.load(f)
    except Exception:
        return False
    if (m.get('generation') or {}).get('schedule') != 'uncover1':
        return False
    fid = (m.get('fid') or {}).get('generated_vs_ref')
    grid = m.get('grid')
    return fid is not None and bool(grid) and os.path.isfile(os.path.join(ROOT, grid))


def has_train_ckpt(index):
    n, exp_name, _, _ = job_from_index(index)
    path = os.path.join(ROOT, ckpt_path(exp_name, n))
    try:
        return os.path.isfile(path) and os.path.getsize(path) >= CKPT_MIN_BYTES
    except OSError:
        return False


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


def running_indices(jobs):
    found = set()
    for job in jobs:
        name = job['name']
        if name.startswith(EVAL_PREFIX):
            tail = name[len(EVAL_PREFIX):]
            if tail.isdigit():
                found.add(int(tail))
    return found


def rebuild_tables():
    proc = run([
        os.path.expanduser('~/miniforge3/envs/mage/bin/python'),
        os.path.join(ROOT, 'scripts', 'make_phase1_table.py'),
        '--schedule', 'uncover1',
    ])
    log('tables rc={} {}'.format(proc.returncode, (proc.stdout or '').strip()))


def submit_eval(index):
    env = os.environ.copy()
    env['INDEX'] = str(index)
    env['SCHEDULE'] = 'uncover1'
    env['LSB_NTRIES'] = '1'
    script = os.path.join(ROOT, 'scripts', 'phase1_uncover1_one.sh')
    cmd = ['timeout', '15', BSUB, '-J', '{}{}'.format(EVAL_PREFIX, index)]
    with open(script, 'rb') as fh:
        proc = subprocess.run(
            cmd, stdin=fh, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, env=env, cwd=ROOT)
    msg = ((proc.stdout or '') + ' ' + (proc.stderr or '')).strip()
    if proc.returncode != 0:
        log('uncover1 {} FAILED: {}'.format(index, msg))
        return False
    log('uncover1 {} -> {}'.format(index, msg))
    return True


def once():
    jobs = user_jobs()
    running = running_indices(jobs)
    pending = [
        i for i in all_indices()
        if (not cell_done(i)) and i not in running and has_train_ckpt(i)
    ]
    slots = max(USER_MAX_JOBS - len(jobs), 0)
    log('jobs={} running={} pending={} slots={}'.format(
        len(jobs), sorted(running), pending, slots))
    if not pending and not running:
        rebuild_tables()
        log('all 16 uncover1 evals have metrics')
        return True
    if not pending:
        log('waiting for in-flight {}'.format(sorted(running)))
        return False
    submitted = 0
    for index in list(pending):
        if submitted >= slots:
            break
        if not submit_eval(index):
            break
        submitted += 1
        slots -= 1
        time.sleep(2)
    return False


def main():
    os.chdir(ROOT)
    log('fill-uncover1 start cwd={}'.format(os.getcwd()))
    while True:
        try:
            if once():
                log('fill-uncover1 complete')
                return
        except Exception as exc:
            log('error {}'.format(exc))
        time.sleep(90)


if __name__ == '__main__':
    main()
