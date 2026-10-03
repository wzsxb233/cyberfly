"""Bounded, fully SHA-checked artifact reads with kernel path confinement.

Linux openat2 resolves a relative path beneath a pinned trusted directory in
one syscall. Only the directory handle is cached: file bytes, metadata and SHA
are checked on every call. Older kernels retain the original resolve fallback.
"""
from __future__ import annotations

import atexit
from collections import OrderedDict
import ctypes
import errno
import hashlib
import os
from pathlib import Path
import platform
import stat
import sys
import threading

_lock=threading.RLock()
_roots=OrderedDict()
_max_roots=8
_openat2_available=None


class _OpenHow(ctypes.Structure):
    _fields_=[('flags',ctypes.c_uint64),('mode',ctypes.c_uint64),('resolve',ctypes.c_uint64)]


_libc=ctypes.CDLL(None,use_errno=True)
_syscall=_libc.syscall
_syscall.restype=ctypes.c_long
_supported_platform=sys.platform.startswith('linux') and platform.machine() in ('x86_64','aarch64')


def close_cached_roots():
    with _lock:
        for entry in _roots.values():os.close(entry['fd'])
        _roots.clear()


atexit.register(close_cached_roots)


def _root_handle(root):
    """Duplicate the cached fd; eviction cannot close an in-flight reader."""
    key=str(Path(root).absolute())
    with _lock:
        info=os.stat(key)
        if not stat.S_ISDIR(info.st_mode):raise ValueError('Artifact root is not a directory')
        entry=_roots.get(key)
        if entry is None or entry['identity']!=(info.st_dev,info.st_ino):
            canonical=Path(key).resolve(strict=True)
            fd=os.open(canonical,os.O_RDONLY|os.O_DIRECTORY|os.O_CLOEXEC)
            actual=os.fstat(fd)
            if (actual.st_dev,actual.st_ino)!=(info.st_dev,info.st_ino):
                os.close(fd)
                raise ValueError('Artifact root changed while opening')
            if entry is not None:os.close(entry['fd'])
            entry={'fd':fd,'canonical':canonical,'identity':(actual.st_dev,actual.st_ino)}
            _roots[key]=entry
        _roots.move_to_end(key)
        while len(_roots)>_max_roots:
            _,old=_roots.popitem(last=False);os.close(old['fd'])
        return os.dup(entry['fd']),Path(key),entry['canonical']


def _open_confined(path,root):
    global _openat2_available
    if not _supported_platform or _openat2_available is False:return None
    rootfd,lexical_root,canonical_root=_root_handle(root)
    try:
        absolute=Path(path).absolute()
        try:relative=absolute.relative_to(lexical_root)
        except ValueError:
            # Canonical root spelling is also accepted; unrelated prefixes do
            # not fall back to an unconfined open.
            try:relative=absolute.relative_to(canonical_root)
            except ValueError:raise ValueError('Artifact path is outside its root') from None
        how=_OpenHow(os.O_RDONLY|os.O_NONBLOCK|os.O_CLOEXEC,0,0x08|0x04)
        fd=_syscall(437,rootfd,ctypes.c_char_p(os.fsencode(relative)),ctypes.byref(how),ctypes.sizeof(how))
        if fd<0:
            code=ctypes.get_errno()
            if code in (errno.ENOSYS,errno.EINVAL):
                _openat2_available=False
                return None
            if code in (errno.ELOOP,errno.EXDEV):
                raise ValueError('Artifact path escapes its root or contains a symbolic link')
            if code in (errno.ENOENT,errno.ENOTDIR):
                raise FileNotFoundError(code,os.strerror(code),str(path))
            raise OSError(code,os.strerror(code),str(path))
        _openat2_available=True
        # openat2 has already resolved any in-root '..' components safely.
        return int(fd),Path(os.path.normpath(str(canonical_root/relative)))
    finally:os.close(rootfd)


def _open_resolved(path,root):
    path=Path(path).resolve()
    if not path.is_relative_to(Path(root).resolve()):raise ValueError('Artifact path is outside its root')
    try:return os.open(path,os.O_RDONLY|os.O_NONBLOCK|os.O_CLOEXEC),path
    except IsADirectoryError:
        raise ValueError('Artifact path is missing, outside its root or oversized') from None


def read_bounded_artifact(path,*,root,limit=64*1024*1024):
    """Confined regular-file bytes; caller must verify its external SHA.

    This lower level is for the latest-pointer transaction: JSON supplies the
    NPZ hash and a temporarily missing NPZ must remain distinguishable.
    """
    if type(limit)is not int or limit<0:raise ValueError('Invalid artifact size limit')
    path=os.fspath(path) if isinstance(path,os.PathLike) else path
    if not isinstance(path,str) or not path or '\0' in path:raise ValueError('Invalid artifact path')
    opened=_open_confined(path,root)
    fd,resolved=opened if opened is not None else _open_resolved(path,root)
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size>limit:
            raise ValueError('Artifact path is missing, outside its root or oversized')
        stream=os.fdopen(fd,'rb')
    except BaseException:
        os.close(fd)
        raise
    with stream:
        # Normal immutable artifacts allocate only their actual size. A file
        # growing while open is still bounded and must match the full SHA.
        raw=stream.read(min(info.st_size+1,limit+1))
        if len(raw)>info.st_size:raw+=stream.read(limit-len(raw)+1)
    if len(raw)>limit:raise ValueError('Artifact path is missing, outside its root or oversized')
    return resolved,raw


def read_checked_artifact(descriptor,*,root,limit=64*1024*1024):
    if not isinstance(descriptor,dict):raise ValueError('Expected path/SHA artifact descriptor')
    try:resolved,raw=read_bounded_artifact(descriptor.get('path'),root=root,limit=limit)
    except FileNotFoundError:
        raise ValueError('Artifact path is missing, outside its root or oversized') from None
    if hashlib.sha256(raw).hexdigest()!=descriptor.get('sha256'):raise ValueError('Artifact file SHA mismatch')
    return resolved,raw


def read_backend():
    return 'linux_openat2' if _openat2_available else 'resolve_fallback' if _openat2_available is False or not _supported_platform else 'unprobed'
