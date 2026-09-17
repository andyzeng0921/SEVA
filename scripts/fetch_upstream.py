"""Fetch pinned dependencies without starting or installing robot software."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=['exploration', 'gateway', 'all'], default='exploration')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    upstream = root / 'upstream'
    upstream.mkdir(exist_ok=True)
    entries = json.loads((root / 'dependencies/sources.lock.json').read_text())['sources']
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1', GIT_TERMINAL_PROMPT='0')

    def git(path, *arguments, capture=False):
        return subprocess.run(['git', '-C', str(path), *arguments], check=True,
                              env=env, text=True, capture_output=capture)

    for entry in entries:
        if args.group != 'all' and entry['group'] != args.group:
            continue
        destination = upstream / entry['name']
        if destination.exists():
            revision = git(destination, 'rev-parse', 'HEAD', capture=True).stdout.strip()
            dirty = git(destination, 'status', '--porcelain', capture=True).stdout.strip()
            if revision != entry['commit'] or dirty:
                raise SystemExit(f'Refusing to overwrite an existing checkout: {entry["name"]}')
        else:
            destination.mkdir()
            git(destination, 'init', '--quiet')
            git(destination, 'remote', 'add', 'origin', entry['repository'])
            git(destination, 'fetch', '--depth', '1', 'origin', entry['commit'])
            git(destination, 'checkout', '--detach', '--quiet', 'FETCH_HEAD')
        print(f'{entry["name"]}: pinned source ready')

    hashes = json.loads((root / 'dependencies/entrypoints.sha256.json').read_text())
    for relative, expected in hashes.items():
        path = upstream / relative
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f'Entry-point integrity mismatch: {relative}')
    for name in ('sources.lock.json', 'entrypoints.sha256.json'):
        shutil.copyfile(root / 'dependencies' / name, upstream / name)


if __name__ == '__main__':
    main()
