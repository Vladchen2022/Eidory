"""Local runtime diagnostics with rotating logs and no remote telemetry.

No user content is deliberately recorded; exception messages may contain paths.
"""
import faulthandler
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
import threading
import time

_fault_file = None


def setup_diagnostics(directory: Path):
    global _fault_file
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('eidory.runtime')
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    handler = RotatingFileHandler(directory / 'runtime.log', maxBytes=2*1024*1024, backupCount=2)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(handler)
    fault = directory / 'native-fault.log'
    if fault.exists():
        fault.replace(directory / 'native-fault.previous.log')
    _fault_file = fault.open('w')
    faulthandler.enable(_fault_file)
    original = sys.excepthook
    def exception_hook(kind, value, traceback):
        logger.error('Unhandled Python exception', exc_info=(kind, value, traceback))
        original(kind, value, traceback)
    sys.excepthook = exception_hook
    original_thread = threading.excepthook
    def thread_hook(args):
        logger.error('Unhandled worker exception', exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
        original_thread(args)
    threading.excepthook = thread_hook
    logger.info('Application starting')
    return logger


def watch_event_loop(window, logger):
    from PySide6.QtCore import QTimer
    timer = QTimer(window)
    timer.setInterval(100)
    last = time.monotonic()
    last_log = 0.0
    def tick():
        nonlocal last, last_log
        now = time.monotonic()
        delay = now-last-.1
        if delay > .2 and now-last_log > 5:
            logger.info('Event loop delayed %.0f ms', delay*1000)
            last_log = now
        last = now
    timer.timeout.connect(tick)
    timer.start()
    return timer
