"""Fresh path identities without Python walking every Windows mount ancestor.

Only the ordinary, existing path case uses openat2. Symlinks, missing paths,
external paths and older kernels keep Path.resolve semantics. No file bytes,
metadata or resolved path results are cached here.
"""
import os
from pathlib import Path
import stat


def _ordinary_open(path, root):
    from .checked_artifacts import _open_confined
    absolute = Path(path).absolute()
    if '\0' in os.fspath(absolute):
        # ctypes C strings would truncate this path. Preserve resolve's error.
        return None
    if not absolute.is_relative_to(Path(root).absolute()):
        return None
    try:
        return _open_confined(absolute, root)
    except (OSError, ValueError):
        # This is an identity optimization, not a new path authorization rule.
        # Existing symlink and missing-path behavior belongs to Path.resolve.
        return None


def resolve_path(path, *, root, strict=False):
    opened = _ordinary_open(path, root)
    if opened is None:
        return Path(path).resolve(strict=strict)
    fd, canonical = opened
    os.close(fd)
    return canonical


def regular_file_revision(path, *, root):
    opened = _ordinary_open(path, root)
    if opened is None:
        canonical = Path(path).resolve(strict=True)
        info = canonical.stat()
        if not canonical.is_file():
            raise ValueError('Pinned checkpoint must remain a regular file')
    else:
        fd, canonical = opened
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError('Pinned checkpoint must remain a regular file')
        finally:
            os.close(fd)
    return (str(canonical), info.st_dev, info.st_ino, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)
