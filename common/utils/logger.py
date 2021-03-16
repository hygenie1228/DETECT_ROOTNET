import time
import datetime
import logging

class Logger:
    def __init__(self):
        self.log_interval = 0
        self.start_time = time.time()
        
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        # to stream
        formatter = logging.Formatter('\x1b[2;32m[%(asctime)s]\x1b[0m\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def info(self, message):
        self.logger.info(message)

logger = Logger()