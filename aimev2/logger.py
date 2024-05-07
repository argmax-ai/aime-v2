import json
import logging
import os
from pprint import pformat

from torch.utils.tensorboard import SummaryWriter


def get_default_logger(root: str):
    logger = ListLogger(root)
    logger.add(TerminalLogger)
    logger.add(TensorboardLogger)
    logger.add(JsonlLogger)
    return logger


class Logger:
    def __init__(self, root: str) -> None:
        self.root = root

    def __call__(self, metrics: dict, step: int, **kwargs):
        raise NotImplementedError


class ListLogger(Logger):
    def __init__(self, root: str) -> None:
        super().__init__(root)
        self.loggers = []

    def add(self, logger_class):
        self.loggers.append(logger_class(self.root))

    def __call__(self, metrics: dict, step: int, **kwargs):
        for logger in self.loggers:
            logger(metrics, step, **kwargs)


class TerminalLogger(Logger):
    """log metrics to the terminal"""

    def __init__(self, root: str) -> None:
        super().__init__(root)
        self.log = logging.getLogger("log")

    def __call__(self, metrics: dict, step: int, **kwargs):
        metrics = metrics.copy()
        for k in list(metrics.keys()):
            if "video" in k or (
                hasattr(metrics[k], "shape") and len(metrics[k].shape) >= 3
            ):
                metrics.pop(k)
        self.log.info(pformat(metrics))


class TensorboardLogger(Logger):
    """log metrics to the tensorboard"""

    def __init__(self, root: str) -> None:
        super().__init__(root)
        self.writer = SummaryWriter(self.root)

    def __call__(self, metrics: dict, step: int, fps=25, **kwargs):
        for k, v in metrics.items():
            if "video" in k:
                v = v.permute(0, 3, 1, 2).unsqueeze(dim=0)
                v = v / 255  # hard coded covert
                self.writer.add_video(k, v, global_step=step, fps=fps)
            elif isinstance(v, list):
                continue
            elif hasattr(v, "shape") and len(v.shape) == 3:
                v = v.permute(2, 0, 1) / 255
                self.writer.add_image(k, v, global_step=step)
            else:
                self.writer.add_scalar(k, v, global_step=step)


class JsonlLogger(Logger):
    """log metrics to a jsonl file"""

    def __init__(self, root: str, file_name: str = "metrics.jsonl") -> None:
        super().__init__(root)
        self.writer = open(os.path.join(self.root, file_name), "w")

    def __call__(self, metrics: dict, step: int, **kwargs):
        metrics = metrics.copy()
        for k in list(metrics.keys()):
            if "video" in k or (
                hasattr(metrics[k], "shape") and len(metrics[k].shape) >= 3
            ):
                metrics.pop(k)
        metrics["step"] = step
        self.writer.write(json.dumps(metrics) + "\n")
        self.writer.flush()
