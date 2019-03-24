from datetime import datetime as dt
from pathlib import Path
import logging


def setup(logger: logging.Logger, log_dir_path: str = None, log_level = logging.INFO) -> logging.Logger:
    logger.setLevel(log_level)  # TODO: 設定ファイルに持たせる

    # Setting Handler
    if log_dir_path:
        file_name = '{}.log'.format(dt.now().strftime('%Y%m%d'))
        Path(log_dir_path).mkdir(exist_ok=True, parents=True)
        file_handler = logging.FileHandler(str(Path(log_dir_path) / file_name), encoding='utf-8')
        file_handler.setLevel(log_level)

        fh_formatter = logging.Formatter('%(levelname)s, %(asctime)s, %(name)s, %(message)s')
        file_handler.setFormatter(fh_formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)

    sh_formatter = logging.Formatter('[%(levelname)s][%(asctime)s][%(name)s] %(message)s')
    stream_handler.setFormatter(sh_formatter)

    logger.addHandler(stream_handler)
    return logger
