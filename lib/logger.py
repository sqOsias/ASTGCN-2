# -*- coding:utf-8 -*-

"""
统一日志模块。
基于 Python 标准 logging 库，提供控制台 + 文件双输出、
结构化格式、以及 TensorBoard 写入的封装。
"""

import os
import logging
import sys
from typing import Optional

from torch.utils.tensorboard import SummaryWriter


def get_logger(name: str = 'ASTGCN',
               log_dir: Optional[str] = None,
               level: int = logging.INFO) -> logging.Logger:
    """
    获取或创建一个 Logger 实例。

    Parameters
    ----------
    name : str
        Logger 名称
    log_dir : str, optional
        若提供，则同时输出日志到该目录下的 train.log 文件
    level : int
        日志级别，默认 INFO

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, 'train.log'), encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class TBWriter:
    """
    对 TensorBoard SummaryWriter 的轻量封装，
    提供统一的 scalar / histogram 写入接口。
    """

    def __init__(self, log_dir: str, flush_secs: int = 5):
        self.sw = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)

    def add_scalar(self, tag: str, value, step: int):
        self.sw.add_scalar(tag, value, step)

    def add_histogram(self, tag: str, values, step: int, bins: int = 1000):
        self.sw.add_histogram(tag=tag, values=values,
                              global_step=step, bins=bins)

    def log_gradients(self, net, global_step: int, logger: Optional[logging.Logger] = None):
        """记录模型所有参数的梯度直方图。"""
        for name, param in net.named_parameters():
            try:
                if param.grad is not None:
                    self.add_histogram(
                        tag=name + '_grad',
                        values=param.grad.detach().cpu().numpy(),
                        step=global_step,
                    )
            except Exception:
                if logger:
                    logger.warning("can't plot histogram of %s_grad", name)

    def close(self):
        self.sw.close()
