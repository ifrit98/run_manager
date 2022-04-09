import logging
from sys import stdout, stderr

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level=logging.DEBUG):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

def stream_logger(logname,
                  logfile=None,
                  level=logging.DEBUG, 
                  encoding='utf-8',
                  formatting='lite'):
    formatting = '%(levelname)s - %(message)s' \
        if formatting == 'lite' \
        else '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logger = logging.getLogger(logname)
    logger.setLevel(level)
    fh = logging.FileHandler(logfile or logname + '.txt', 'w', encoding)
    fh.setLevel(level)
    formatter = logging.Formatter(formatting)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    stdout = StreamToLogger(logger, logging.DEBUG)

def get_logger(logname, logfile=None, level=logging.DEBUG, encoding='utf-8', formatting='lite'):
    formatting = '%(levelname)s - %(message)s' \
        if formatting == 'lite' \
        else '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logger = logging.getLogger(logname)
    logger.setLevel(level)
    fh = logging.FileHandler(logfile or logname + '.txt', 'w', encoding)
    fh.setLevel(level)
    formatter = logging.Formatter(formatting)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def file_logger(logname,
                logfile=None,
                level=logging.DEBUG, 
                encoding='utf-8',
                formatting='lite'):
    formatting = '%(levelname)s - %(message)s' \
        if formatting == 'lite' \
        else '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logger = logging.getLogger(logname)
    logger.setLevel(level)
    fh = logging.FileHandler(logfile or logname + '.txt', 'w', encoding)
    fh.setLevel(level)
    formatter = logging.Formatter(formatting)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
