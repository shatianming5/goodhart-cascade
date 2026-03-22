"""
Disk space monitor - prevents disk full crashes during training.

Checks disk space periodically and triggers cleanup when needed.
"""

import os
import shutil
import threading
import time


class DiskMonitor:
    """Monitor disk space and auto-cleanup when low."""

    def __init__(
        self,
        watch_path: str = "/",
        min_free_gb: float = 50.0,
        check_interval_sec: int = 300,
        experiment_dirs: list[str] | None = None,
    ):
        self.watch_path = watch_path
        self.min_free_gb = min_free_gb
        self.check_interval_sec = check_interval_sec
        self.experiment_dirs = experiment_dirs or []
        self._stop = threading.Event()
        self._thread = None

    def get_free_gb(self) -> float:
        stat = shutil.disk_usage(self.watch_path)
        return stat.free / (1024**3)

    def emergency_cleanup(self):
        """Remove old checkpoints when disk is critically low."""
        print(f"[DiskMonitor] WARNING: Free space below {self.min_free_gb} GB!")
        print(f"[DiskMonitor] Running emergency checkpoint cleanup...")

        for exp_dir in self.experiment_dirs:
            if not os.path.exists(exp_dir):
                continue

            checkpoints = sorted([
                d for d in os.listdir(exp_dir)
                if d.startswith("checkpoint-") and os.path.isdir(os.path.join(exp_dir, d))
            ], key=lambda x: int(x.split("-")[1]))

            if len(checkpoints) <= 2:
                continue

            # Remove all but last 2 checkpoints
            for ckpt in checkpoints[:-2]:
                path = os.path.join(exp_dir, ckpt)
                try:
                    shutil.rmtree(path)
                    print(f"[DiskMonitor] Removed {path}")
                except Exception as e:
                    print(f"[DiskMonitor] Failed to remove {path}: {e}")

                if self.get_free_gb() > self.min_free_gb:
                    print(f"[DiskMonitor] Disk space recovered.")
                    return

        # Also clean HF cache
        hf_cache = os.path.expanduser("~/.cache/huggingface")
        if os.path.exists(hf_cache):
            cache_size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fns in os.walk(hf_cache)
                for f in fns
            ) / (1024**3)
            if cache_size > 20:
                print(f"[DiskMonitor] HF cache is {cache_size:.1f} GB, clearing...")
                shutil.rmtree(hf_cache, ignore_errors=True)

    def _monitor_loop(self):
        while not self._stop.is_set():
            free = self.get_free_gb()
            if free < self.min_free_gb:
                self.emergency_cleanup()
            self._stop.wait(self.check_interval_sec)

    def start(self):
        """Start background monitoring."""
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        free = self.get_free_gb()
        print(f"[DiskMonitor] Started. Current free space: {free:.1f} GB")

    def stop(self):
        """Stop monitoring."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
