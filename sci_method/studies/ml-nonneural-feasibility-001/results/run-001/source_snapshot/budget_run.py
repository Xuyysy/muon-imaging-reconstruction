"""Run one bounded research command, preserving logs and CPU accounting."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--limit', type=int, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    directory = Path(__file__).resolve().parent / 'verification'
    directory.mkdir(exist_ok=True)
    log = directory / (args.tag + '.log')
    record = directory / (args.tag + '.json')
    if log.exists() or record.exists():
        raise FileExistsError('never overwrite a verification record')
    prior = [json.loads(p.read_text()) for p in directory.glob('*.json')]
    used = sum(x.get('child_cpu_seconds', 0) for x in prior)
    if args.limit <= 0 or used + args.limit > 3600:
        raise RuntimeError('cumulative reserved CPU exceeds one core-hour')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', MKL_NUM_THREADS='1')
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    def cap():
        resource.setrlimit(resource.RLIMIT_CPU, (args.limit, args.limit))
    meta = dict(command=command, cpu_limit=args.limit, state='started',
                thread_environment={k: env[k] for k in ('OPENBLAS_NUM_THREADS',
                    'OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS')})
    record.write_text(json.dumps(meta, indent=2)+'\n')
    with log.open('x') as stream:
        proc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                              env=env, preexec_fn=cap)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    meta.update(state='completed' if proc.returncode == 0 else 'failed',
        exit_code=proc.returncode, wall_seconds=time.perf_counter()-start,
        child_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)
    record.write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(meta, indent=2))
    print(log.read_text())
    raise SystemExit(proc.returncode)


if __name__ == '__main__':
    main()
