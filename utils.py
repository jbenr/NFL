from tabulate import tabulate_formats, tabulate
import os
import re
import hashlib
import json
import shutil
import time
from pathlib import Path
import tempfile


# A full rerun ("recompute everything, trust nothing on disk") sets this to the
# run's start time: every cache entry written before then is deleted the first
# time cache_path() hands out its name, so the value gets recomputed and
# re-saved, while anything this same run already wrote is still reused. It
# lives in the environment because the heavy work happens in worker processes
# (forkserver/loky), which do not inherit module globals but do inherit os.environ.
_BYPASS_SINCE = float(os.environ.get('NFL_CACHE_BYPASS') or 0)


def _filesystem_now():
    """'Now' as this filesystem would stamp it. File mtimes come from the
    kernel's coarse clock and can land a tick BEHIND time.time(), so a cache
    written seconds into the run can look older than a wall-clock cutoff --
    take the cutoff off the same clock the comparison uses instead."""
    folder = Path('data/cache')
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=folder, prefix='.refresh-', delete=False) as marker:
            name = marker.name
        try:
            return os.stat(name).st_mtime
        finally:
            os.unlink(name)
    except OSError:
        return time.time()


def bypass_caches(enabled=True, since=None):
    """Ignore (and delete) every derived cache written before `since` (default: now)."""
    global _BYPASS_SINCE
    _BYPASS_SINCE = float(_filesystem_now() if since is None else since) if enabled else 0.
    if _BYPASS_SINCE:
        os.environ['NFL_CACHE_BYPASS'] = repr(_BYPASS_SINCE)
    else:
        os.environ.pop('NFL_CACHE_BYPASS', None)
    return _BYPASS_SINCE


_INVALIDATED = set()


def _drop_stale(folder, key):
    """Remove one key's pre-cutoff cache entries: the .parquet itself plus every
    sibling callers build from the same name -- .importance.parquet, .npz, and
    the extensionless artifacts directory. Checked once per key per process;
    after that, anything sitting there is this run's own work."""
    if (folder, key) in _INVALIDATED:
        return
    _INVALIDATED.add((folder, key))
    for path in sorted(folder.glob(f'{key}*')):  # Listed up front: deleting mid-scandir skips entries.
        try:
            if path.stat().st_mtime >= _BYPASS_SINCE:
                continue  # Written by this run already; reusing it is the point.
            shutil.rmtree(path) if path.is_dir() else path.unlink()
        except OSError:
            pass  # Another worker beat us to it -- a miss is all this needs.


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
    if _BYPASS_SINCE:
        _drop_stale(folder, key)
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
