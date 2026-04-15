#!/usr/bin/env python
# -*- coding:utf-8 -*-

import logging.handlers
import os
import datetime
import logging
from logging import handlers as log_handlers

import traceback

def make(logger_name,
        level=logging.INFO,
        save_dir='.',
        time_filename=True,
        time_rotating=False,
        backup_count=1,
        max_bytes=None, # 1048576,  # 1 mega bytes
        logging_=True,
        console_out=True):

    if not logging_:
        return get_stdout_logger()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = logger_name
    if time_filename:
        filename += '.' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S.%f")

    filename += '.log'
        
    log_path = os.path.join(save_dir, filename)

    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    if time_rotating:
        file_handler = logging.handlers.TimedRotatingFileHandler(log_path, backupCount=backup_count, when='midnight')
    elif max_bytes is not None:
        file_handler = logging.handlers.RotatingFileHandler(log_path, mode='a',
                                                        backupCount=backup_count, maxBytes=max_bytes)
    else:
        file_handler = logging.FileHandler(log_path)

    file_handler.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(file_handler)

    if console_out:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


def write_exception_log(logger, e, msg=""):
    if not logger:
        logger = get_stdout_logger()

    if msg:
        msg += " : "
    msg += "Exception.\n"
    msg += str(e) + "\n" + traceback.format_exc()
    logger.error(msg)


def get_stdout_logger():
    class PrintLogger:
        def info(self):
            pass

        def error(self):
            pass

    logger = PrintLogger()
    logger.info = print
    logger.error = print
    return logger
