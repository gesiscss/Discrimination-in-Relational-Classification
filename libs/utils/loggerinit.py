import logging
import os.path


def initialize_logger(output,prefix,source):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s \t %(levelname)s \t %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create error file handler and set level to error
    handler = logging.FileHandler("{}/{}_{}_error.log".format(output,prefix,source), "a", encoding=None, delay="true")
    handler.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s \t %(levelname)s \t %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create debug file handler and set level to debug
    handler = logging.FileHandler("{}/{}_{}_debug.log".format(output,prefix,source), "a")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s \t %(levelname)s \t %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create debug file handler and set level to debug
    handler = logging.FileHandler("{}/{}_{}_warning.log".format(output, prefix, source), "a")
    handler.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)s \t %(levelname)s \t %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)