import logging
import sys

out_hdlr = logging.StreamHandler(sys.stdout)
out_hdlr.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
out_hdlr.setLevel(logging.INFO)

def init_logger(logger: logging.Logger, level=logging.INFO) -> None:
    logger.addHandler(out_hdlr)
    logger.setLevel(level)
