"""
Logging set for the repository
"""

import logging

from rich.logging import RichHandler

logger = logging.getLogger(__name__)
shell_handler = RichHandler()
file_handler = logging.FileHandler("debug.log")

logger.setLevel(logging.INFO)
shell_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.WARNING)

# the formatter determines how the logger looks like
FMT_SHELL = "%(message)s"
FMT_FILE = """%(levelname)s %(asctime)s [%(filename)s
    %(funcName)s %(lineno)d] %(message)s"""

shell_formatter = logging.Formatter(FMT_SHELL)
file_formatter = logging.Formatter(FMT_FILE)

# Putting them together
shell_handler.setFormatter(shell_formatter)
file_handler.setFormatter(file_formatter)

logger.addHandler(shell_handler)
logger.addHandler(file_handler)
