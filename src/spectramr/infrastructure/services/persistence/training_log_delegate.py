import csv
import logging
import queue
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TrainingLogDelegate:
    """Delegate responsible for high-frequency training logs (Losses).

    ### Non-Blocking Logging Flow
    ```mermaid
    sequenceDiagram
        participant M as Main (Training) Thread
        participant Q as Sync Queue (FIFO)
        participant B as Background Writer Thread
        participant F as CSV File

        M->>Q: put_nowait(metric_row)
        alt Queue Full
            M->>Q: Drop Oldest + Put
        end
        B->>Q: get(timeout=1s)
        B->>B: _write_row_sync()
        B->>F: Append to Disk
    ```

    [PRODUCTION FIX] Uses background thread for non-blocking I/O.
    """

    def __init__(self, output_dir: str, log_file: Any = None):
        """__init__.

        Args:
            output_dir (str): Description.
            log_file (Any): Description.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if log_file:
            self.log_file = Path(log_file)
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.log_file = self.output_dir / "loss_log.csv"

        self._headers: list[str] = []
        self._standard_headers = ["timestamp", "epoch", "step", "loss"]

        # [PRODUCTION FIX] Async logging queue
        self._queue: queue.Queue = queue.Queue(maxsize=10000)
        self._shutdown = threading.Event()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

    def log_step(self, data: dict[str, Any], step: int, epoch: int):
        """Log step data (losses, LR, etc) - non-blocking."""
        row = data.copy()
        row["step"] = step
        row["epoch"] = epoch

        # [PRODUCTION FIX] Enqueue instead of blocking I/O
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            # Drop oldest if full (backpressure)
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(row)
            except queue.Empty as _exc:
                logger.debug("Suppressed exception: %s", _exc)

    def _writer_loop(self):
        """Background thread that writes queued rows to CSV."""
        while not self._shutdown.is_set():
            try:
                row = self._queue.get(timeout=1.0)
                self._write_row_sync(row)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Async log writer error: {e}")

    def _write_row_sync(self, row: dict[str, Any]):
        """Synchronous write (called from background thread)."""
        keys = list(row.keys())
        new_keys = [k for k in keys if k not in self._headers]

        if new_keys or not self.log_file.exists():
            self._update_schema(new_keys)

        try:
            with open(self.log_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._headers)
                writer.writerow(row)
        except Exception as e:
            logger.error(f"Failed to log training step: {e}")

    def _update_schema(self, new_keys: list):
        """Update CSV schema."""
        current_headers = self._headers.copy()
        if not current_headers:
            ordered_new = [k for k in self._standard_headers if k in new_keys] + [
                k for k in new_keys if k not in self._standard_headers
            ]
            updated_headers = ordered_new
        else:
            updated_headers = current_headers + [k for k in new_keys if k not in current_headers]

        if self.log_file.exists() and current_headers:
            backup_file = self.log_file.with_suffix(".csv.bak")
            try:
                self.log_file.rename(backup_file)
                with (
                    open(backup_file, newline="") as f_src,
                    open(self.log_file, "w", newline="") as f_dst,
                ):
                    reader = csv.DictReader(f_src)
                    writer = csv.DictWriter(f_dst, fieldnames=updated_headers)
                    writer.writeheader()
                    for r in reader:
                        writer.writerow(r)
            except Exception as e:
                logger.error(f"Loss log schema migration failed: {e}")
        else:
            self._headers = updated_headers
            with open(self.log_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._headers)
                writer.writeheader()

        self._headers = updated_headers

    def flush(self):
        """Flush remaining items in queue."""
        while not self._queue.empty():
            try:
                row = self._queue.get_nowait()
                self._write_row_sync(row)
            except queue.Empty:
                break

    def shutdown(self):
        """Graceful shutdown."""
        self._shutdown.set()
        self.flush()
        self._writer_thread.join(timeout=5.0)
