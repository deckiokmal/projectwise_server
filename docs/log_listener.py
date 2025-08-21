# log_listener.py
from __future__ import annotations

import logging
import pickle
import socketserver
import struct
from typing import Set

from mcp_server.utils.logger import (
    MonthAwareTimedRotatingFileHandler,
    _monthly_log_dir_for,
    _to_level,
    get_settings,
)

# Cache agar tiap logger name hanya diinisialisasi sekali
_inited: Set[str] = set()


def ensure_module_logger(name: str) -> logging.Logger:
    """
    Buat logger lokal per-module (file + rotasi), tanpa mode 'socket'.
    Digunakan hanya oleh listener agar tidak mem-forward lagi.
    """
    logger = logging.getLogger(name)
    if name in _inited and logger.handlers:
        return logger

    s = get_settings()
    logger.setLevel(_to_level(s.level))
    logger.propagate = False

    formatter = logging.Formatter(fmt=s.format, datefmt=s.datefmt)
    last_segment = (name.rsplit(".", 1)[-1] or "app").replace(":", "_")
    base_name = f"{last_segment}.log"

    fh = MonthAwareTimedRotatingFileHandler(
        base_name=base_name,
        month_dir_factory=_monthly_log_dir_for,
        when="midnight",
        backupCount=int(s.retention),
        encoding="utf-8",
        utc=bool(s.use_utc),
        delay=False,
    )
    fh.setLevel(_to_level(s.level))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if s.console:
        ch = logging.StreamHandler()
        ch.setLevel(_to_level(s.console_level or s.level))
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    _inited.add(name)
    return logger


class LogRecordStreamHandler(socketserver.StreamRequestHandler):
    """
    Format payload: [len:4 bytes big-endian][pickle(logrecord_dict)]
    Sesuai format default logging.SocketHandler.
    """

    def handle(self):
        while True:
            chunk = self.connection.recv(4)
            if len(chunk) < 4:
                break
            slen = struct.unpack(">L", chunk)[0]
            chunk = b""
            while len(chunk) < slen:
                data = self.connection.recv(slen - len(chunk))
                if not data:
                    break
                chunk += data
            if len(chunk) != slen:
                break

            try:
                obj = pickle.loads(chunk)
                record = logging.makeLogRecord(obj)
                logger = ensure_module_logger(record.name)
                logger.handle(record)
            except Exception:
                # Jangan matikan server hanya karena satu record gagal
                root = ensure_module_logger("listener")
                root.exception("Failed to handle incoming log record")


class LogRecordSocketReceiver(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def main():
    s = get_settings()
    server = LogRecordSocketReceiver(
        (s.socket_host, int(s.socket_port)), LogRecordStreamHandler
    )
    root = ensure_module_logger("listener")
    root.info("Log listener started at %s:%s", s.socket_host, s.socket_port)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        root.info("Log listener stopping...")
    finally:
        server.server_close()
        root.info("Log listener stopped.")


if __name__ == "__main__":
    main()
