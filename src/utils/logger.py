"""日志系统"""

import logging
from pathlib import Path


def setup_logger(name: str, log_dir: str = 'logs') -> logging.Logger:
    """
    设置日志记录器
    Args:
        name: 记录器名称
        log_dir: 日志目录
    Returns:
        Logger实例
    """
    Path(log_dir).mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 文件处理器
    fh = logging.FileHandler(f'{log_dir}/{name}.log')
    fh.setLevel(logging.DEBUG)

    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger