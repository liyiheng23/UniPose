import logging
from pytorch_lightning.utilities.rank_zero import rank_zero_only

logger_initialized = {}

def get_logger(log_file=None):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.

    Returns:
        logging.Logger: The expected logger.
    """
    # only rank 0 will add a FileHandler
    head = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if log_file is not None:
        logger = config_logger(log_file, head)
    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)
        logging.basicConfig(format=head)

    return logger

@rank_zero_only
def config_logger(log_file, head):
    logging.basicConfig(filename=str(log_file))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    formatter = logging.Formatter(head)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    file_handler = logging.FileHandler(log_file, 'w')
    file_handler.setFormatter(logging.Formatter(head))
    file_handler.setLevel(logging.INFO)
    logging.getLogger('').addHandler(file_handler)
    return logger
