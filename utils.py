from tabulate import tabulate_formats, tabulate
import os
import re
import hashlib
import json
from pathlib import Path
import tempfile


def cache_path(kind, config, sources=()):
    """Invalidate derived data when its settings, code, or source files change."""
    files = []
    for source in sorted(map(Path, sources)):
        stat = source.stat()
        identity = hashlib.sha256(source.read_bytes()).hexdigest() if source.suffix == '.py' else (stat.st_size, stat.st_mtime_ns)
        files.append((str(source.resolve()), identity))
    key = hashlib.sha256(json.dumps([config, files], sort_keys=True, default=str).encode()).hexdigest()[:20]
    folder = Path('data/cache') / kind
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f'{key}.parquet'


def save_parquet(frame, path):
    """Publish completed cache files atomically, including across worker processes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.parquet', delete=False) as temp:
        name = temp.name
    try:
        frame.to_parquet(name, index=False)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)

def pdf(df):
    print(tabulate(df, headers='keys', tablefmt=tabulate_formats[2]))


def make_dir(dir):
    if not os.path.exists(dir):
        # exist_ok: workers now run in separate processes, so two can race
        # between the exists() check and the makedirs() call
        os.makedirs(dir, exist_ok=True)
        print(f'Directory {dir} created.')
    # else:
    #     print(f'Directory {dir} already exists.')

def strip_suffix(name):
    return re.sub(r'\s+(Jr\.|Sr\.|I{1,3})$', '', name)
