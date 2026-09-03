#!/usr/bin/env python3
"""Submit phase-1 cells, packing several onto one multi-GPU LSF job when a V100 node has free cards."""
from __future__ import print_function

import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from util.phase1 import (  # noqa: E402
    COMBOS, PHASE1_NS, RECIPE, ckpt_path, job_from_index, metrics_path,
    train_finished,
)

BSUB = '/nfsshare/lsf/10.1/linux3.10-glibc2.17-x86_64/bin/bsub'
BJOBS = '/nfsshare/lsf/10.1/linux3.10-glibc2.17-x86_64/bin/bjobs'
LSLOAD = '/nfsshare/lsf/10.1/linux3.10-glibc2.17-x86_64/bin/lsload'
BHOSTS = '/nfsshare/lsf/10.1/linux3.10-glibc2.17-x86_64/bin/bhosts'
USER_MAX_JOBS = 5
V100_HOSTS = (
    'gpu01', 'gpu03', 'gpu04', 'gpu05', 'gpu06',
    'gpu07', 'gpu08', 'gpu11', 'gpu12',
)
GPU_ALLOC_RE = re.compile(r'^([^:]+):([0-9,]+)$')
JOBID_RE = re.compile(r'^(\d+)')
PACK_PREFIX = 'p1p_'
ONE_PREFIX = 'p1_'
EVAL_PREFIX = 'p1e_'
CKPT_MIN_BYTES = 500 * 1024 * 1024
LOG = os.path.join(ROOT, 'outputs', 'lsf', 'phase1_fill_pack.log')
PREP_METRICS = os.path.join(ROOT, RECIPE['ref_metrics'])
PREP_RECON = os.path.join(ROOT, RECIPE['recon_dir'])
PREP_RECON_MIN = 10000


def log(msg):
    # Fill is started with stdout appended to LOG; do not write the file again.
    print(time.strftime('%Y-%m-%d %H:%M:%S ') + msg, flush=True)


def run(cmd):
    return subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True)


def all_indices():
    raw = os.environ.get('PHASE1_INDICES', '').strip()
    if raw:
        return [int(part) for part in raw.split(',') if part.strip()]
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


def indices_from_name(name):
    found = []
    if name.startswith(PACK_PREFIX):
        for part in name[len(PACK_PREFIX):].split('-'):
            if part.isdigit():
                found.append(int(part))
    elif name.startswith(EVAL_PREFIX):
        tail = name[len(EVAL_PREFIX):]
        if tail.isdigit():
            found.append(int(tail))
    elif name.startswith(ONE_PREFIX):
        tail = name[len(ONE_PREFIX):]
        if tail.isdigit():
            found.append(int(tail))
    return found


def has_train_ckpt(index):
    n, exp_name, _, _ = job_from_index(index)
    path = os.path.join(ROOT, ckpt_path(exp_name, n))
    try:
        return os.path.isfile(path) and os.path.getsize(path) >= CKPT_MIN_BYTES
    except OSError:
        return False


def running_indices(jobs):
    indices = set()
    for job in jobs:
        indices.update(indices_from_name(job['name']))
    return indices


def closed_hosts():
    closed = set()
    out = run([BHOSTS])
    for line in out.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] != 'ok':
            closed.add(parts[0])
    return closed


def gpu_counts():
    counts = {}
    out = run([LSLOAD, '-gpu', '-w'])
    for line in out.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 6 and fields[0] != 'HOST_NAME':
            try:
                counts[fields[0]] = int(fields[5])
            except ValueError:
                continue
    return counts


def occupied_gpus():
    """Map host -> set of busy GPU indices. Retry: empty bjobs looks like all-free."""
    occupied = {}
    for attempt in range(3):
        occupied = {}
        out = run([
            BJOBS, '-u', 'all', '-r', '-noheader',
            '-o', "jobid exec_host gpu_alloc delimiter='|'",
        ])
        for line in out.stdout.splitlines():
            fields = line.split('|')
            if len(fields) < 3:
                continue
            for part in fields[2].split():
                match = GPU_ALLOC_RE.match(part)
                if not match:
                    continue
                host = match.group(1)
                occupied.setdefault(host, set())
                for idx in match.group(2).split(','):
                    if idx.isdigit():
                        occupied[host].add(int(idx))
        if occupied:
            return occupied
        time.sleep(0.4 * (attempt + 1))
    return occupied


def free_v100_groups():
    """Largest-first list of (host, nfree) on open V100 nodes."""
    closed = closed_hosts()
    counts = gpu_counts()
    occupied = occupied_gpus()
    if not occupied:
        log('occupied_gpus empty; skip host pick this round')
        return []
    groups = []
    for host in V100_HOSTS:
        if host in closed:
            continue
        n = int(counts.get(host, 8))
        busy = occupied.get(host, set())
        nfree = sum(1 for i in range(n) if i not in busy)
        if nfree > 0:
            groups.append((host, nfree))
    groups.sort(key=lambda item: (-item[1], item[0]))
    return groups


def submit_pack(indices, host, ngpu):
    env = os.environ.copy()
    env['PACK_INDICES'] = ','.join(str(i) for i in indices)
    env['LSB_NTRIES'] = '1'
    name = PACK_PREFIX + '-'.join(str(i) for i in indices)
    script = os.path.join(ROOT, 'scripts', 'phase1_pack.sh')
    cmd = [
        'timeout', '20', BSUB,
        '-J', name,
        '-gpu', 'num={}:mode=exclusive_process'.format(ngpu),
        '-R', 'order[ut]',
        '-R', 'span[hosts=1]',
        '-R', 'select[hname=={}]'.format(host),
    ]
    with open(script, 'rb') as fh:
        proc = subprocess.run(
            cmd, stdin=fh, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, env=env, cwd=ROOT)
    msg = ((proc.stdout or '') + ' ' + (proc.stderr or '')).strip()
    if proc.returncode != 0:
        log('pack {} host={} FAILED: {}'.format(indices, host, msg))
        return False
    log('pack {} host={} ngpu={} -> {}'.format(indices, host, ngpu, msg))
    return True


def submit_one(index, host=None):
    env = os.environ.copy()
    env['INDEX'] = str(index)
    env['LSB_NTRIES'] = '1'
    script = os.path.join(ROOT, 'scripts', 'phase1_one.sh')
    cmd = ['timeout', '15', BSUB, '-J', '{}{}'.format(ONE_PREFIX, index)]
    if host:
        cmd.extend(['-R', 'select[hname=={}]'.format(host)])
    with open(script, 'rb') as fh:
        proc = subprocess.run(
            cmd, stdin=fh, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, env=env, cwd=ROOT)
    msg = ((proc.stdout or '') + ' ' + (proc.stderr or '')).strip()
    if proc.returncode != 0:
        log('one {} FAILED: {}'.format(index, msg))
        return False
    log('one {} host={} -> {}'.format(index, host, msg))
    return True


def submit_eval(index, host=None):
    env = os.environ.copy()
    env['INDEX'] = str(index)
    env['LSB_NTRIES'] = '1'
    script = os.path.join(ROOT, 'scripts', 'phase1_eval_one.sh')
    cmd = ['timeout', '15', BSUB, '-J', '{}{}'.format(EVAL_PREFIX, index)]
    if host:
        cmd.extend(['-R', 'select[hname=={}]'.format(host)])
    with open(script, 'rb') as fh:
        proc = subprocess.run(
            cmd, stdin=fh, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, env=env, cwd=ROOT)
    msg = ((proc.stdout or '') + ' ' + (proc.stderr or '')).strip()
    if proc.returncode != 0:
        log('eval {} FAILED: {}'.format(index, msg))
        return False
    log('eval {} host={} -> {}'.format(index, host, msg))
    return True


def rebuild_tables():
    """Write tables only for n values this fill is tracking.

    A no-arg make_phase1_table.py walk rewrites every PHASE1_NS file, including
    n=1-100 cosine tables. Restrict to all_indices() so PHASE1_INDICES=20,24
    only touches n=1000 and n=10000.
    """
    python = os.path.expanduser('~/miniforge3/envs/mage/bin/python')
    script = os.path.join(ROOT, 'scripts', 'make_phase1_table.py')
    ns = sorted({job_from_index(i)[0] for i in all_indices()})
    for n in ns:
        proc = run([python, script, '--n', str(n)])
        log('table n={} rc={} {}'.format(
            n, proc.returncode, (proc.stdout or '').strip()))


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
    running = running_indices(jobs)
    pending = [i for i in all_indices() if (not cell_done(i)) and i not in running]
    # Mid-run checkpoint-last.pth is not "done": resume train until log hits last epoch.
    pending_eval = [
        i for i in pending if train_finished(i, ROOT) and has_train_ckpt(i)
    ]
    pending_train = [i for i in pending if i not in pending_eval]
    slots = max(USER_MAX_JOBS - n_jobs, 0)
    groups = free_v100_groups()
    log('jobs={} running={} eval={} train={} slots={} free={}'.format(
        n_jobs, sorted(running), pending_eval, pending_train, slots, groups))
    if not pending and not running:
        rebuild_tables()
        log('all {} cells have metrics'.format(len(all_indices())))
        return True
    if not pending:
        log('waiting for in-flight {}'.format(sorted(running)))
        return False
    submitted = 0
    eval_left = list(pending_eval)
    remain = {host: nfree for host, nfree in groups}
    for host, _nfree in groups:
        while submitted < slots and eval_left and remain.get(host, 0) > 0:
            index = eval_left[0]
            log('submit eval {} host={} slots_left={}'.format(index, host, slots))
            if not submit_eval(index, host):
                eval_left = []
                break
            eval_left = eval_left[1:]
            remain[host] -= 1
            submitted += 1
            n_jobs += 1
            slots = max(USER_MAX_JOBS - n_jobs, 0)
            time.sleep(2)
    left = list(pending_train)
    for host, _nfree in groups:
        nfree = remain.get(host, 0)
        if submitted >= slots or not left or nfree <= 0:
            continue
        # Same n in one pack so a short cell does not hold a GPU for a long sibling.
        first_n = job_from_index(left[0])[0]
        same_n = []
        for index in left:
            if job_from_index(index)[0] == first_n:
                same_n.append(index)
        take = min(nfree, len(same_n), 8)
        chunk = same_n[:take]
        log('submit train {} host={} ngpu={} slots_left={}'.format(
            chunk, host, take, slots))
        ok = submit_pack(chunk, host, take) if take > 1 else submit_one(chunk[0], host)
        if not ok:
            break
        taken = set(chunk)
        left = [index for index in left if index not in taken]
        remain[host] -= take
        submitted += 1
        n_jobs += 1
        slots = max(USER_MAX_JOBS - n_jobs, 0)
        time.sleep(2)
    log('once submitted={} eval_left={} train_left={}'.format(
        submitted, eval_left, left))
    return False


def main():
    os.chdir(ROOT)
    log('fill-pack start cwd={}'.format(os.getcwd()))
    while True:
        delay = 90
        try:
            if once():
                log('fill-pack complete')
                return
            jobs = user_jobs()
            pending = [
                i for i in all_indices()
                if (not cell_done(i)) and i not in running_indices(jobs)
            ]
            slots = max(USER_MAX_JOBS - len(jobs), 0)
            if pending and slots > 0:
                delay = 20
        except Exception as exc:
            log('error {}'.format(exc))
        time.sleep(delay)


if __name__ == '__main__':
    main()
