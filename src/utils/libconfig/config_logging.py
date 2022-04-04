# encoding: utf-8
# Author: Bingxin Ke
# Created: 2021/10/7
import logging
import os
import sys


def config_logging(cfg_logging, out_dir=None):

    file_level = cfg_logging.get('file_level', 10)
    console_level = cfg_logging.get('console_level', 10)

    log_formatter = logging.Formatter(cfg_logging['format'])

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    root_logger.setLevel(min(file_level, console_level))

    if out_dir is not None:
        _logging_file = os.path.join(out_dir, cfg_logging.get('filename', 'logging.log'))
        file_handler = logging.FileHandler(_logging_file)
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(file_level)
        root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(console_level)
    root_logger.addHandler(console_handler)

