import datetime
import os
import sys
import atexit
import logging


def ensure_dir(dir_path):

    os.makedirs(dir_path, exist_ok=True)

def set_color(log, color, highlight=True):
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"

def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur

def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)

class TeeStream:
    """Mirror stdout/stderr to terminal and a UTF-8 log file."""

    def __init__(self, terminal_stream, log_file):
        self.terminal_stream = terminal_stream
        self.log_file = log_file

    def write(self, message):
        self.terminal_stream.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal_stream.flush()
        self.log_file.flush()

    def isatty(self):
        return self.terminal_stream.isatty()


def setup_logging(args, data_mode):
    # Keep log path aligned with Trainer.ckpt_dir.
    log_dir = os.path.join(args.ckpt_dir, data_mode, args.version, str(args.lamda))
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "log.txt")

    # Force UTF-8 text streams to reduce redirection encoding issues.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    # Print/write() output goes to both terminal and log file.
    print_file = open(log_file_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.__stdout__, print_file)
    sys.stderr = TeeStream(sys.__stderr__, print_file)
    atexit.register(print_file.close)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.__stdout__)
    console_handler.setFormatter(fmt)

    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    return log_file_path
