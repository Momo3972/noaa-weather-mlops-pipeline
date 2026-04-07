"""
sitecustomize.py — Chargé automatiquement par Python au démarrage.

Patch DrvFs/WSL2 : les opérations utime et chmod échouent avec EPERM
sur les fichiers dans des montages Windows (DrvFs) via Docker bind mounts.
Ce patch les rend silencieuses pour que MLflow fonctionne normalement.
"""
import shutil as _shutil
import os as _os

# --- Patch shutil.copystat (utime) ---
_orig_copystat = _shutil.copystat

def _safe_copystat(src, dst, *, follow_symlinks=True):
    try:
        _orig_copystat(src, dst, follow_symlinks=follow_symlinks)
    except (PermissionError, OSError):
        pass  # utime non supporté sur DrvFs

_shutil.copystat = _safe_copystat

# --- Patch os.chmod (chmod EPERM sur DrvFs) ---
_orig_chmod = _os.chmod

def _safe_chmod(path, mode, *, dir_fd=None, follow_symlinks=True):
    try:
        _orig_chmod(path, mode, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
    except (PermissionError, OSError):
        pass  # chmod non supporté sur DrvFs

_os.chmod = _safe_chmod
