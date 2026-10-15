#!/usr/bin/env python3
"""Environment and CUDA audit script for SkinAI workstation."""

import platform
import sys
import shutil
from pathlib import Path


def check_python() -> bool:
    print(f"[*] Python Version: {platform.python_version()} ({platform.architecture()[0]})")
    if sys.version_info < (3, 11):
        print("[-] Error: Python 3.11+ is required.")
        return False
    print("[+] Python version check passed.")
    return True


def check_cuda() -> bool:
    try:
        import torch

        print(f"[*] PyTorch Version: {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"[*] CUDA Available: {cuda_avail}")
        if cuda_avail:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[+] Found {device_count} GPU(s): {device_name} ({vram_gb:.2f} GB VRAM)")
            return True
        else:
            print("[!] Warning: PyTorch is installed but CUDA is NOT available.")
            return False
    except ImportError:
        print("[!] PyTorch is not yet installed in this Python environment.")
        return False


def check_disk_space() -> None:
    current_drive = Path.cwd().anchor
    total, used, free = shutil.disk_usage(current_drive)
    print(
        f"[*] Drive {current_drive} Space -> Total: {total / (1024**3):.1f} GB, "
        f"Used: {used / (1024**3):.1f} GB, Free: {free / (1024**3):.1f} GB"
    )
    if free / (1024**3) < 20:
        print("[!] Caution: Low free space on primary drive (< 20 GB).")
        print("    Ensure SKINAI_DATA_DIR in .env points to an external drive (e.g. G:\\).")


def main():
    print("=" * 60)
    print(" SkinAI System Environment Audit")
    print("=" * 60)
    py_ok = check_python()
    check_disk_space()
    _cuda_ok = check_cuda()
    print("=" * 60)
    if py_ok:
        print("[+] Base environment verification complete.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
