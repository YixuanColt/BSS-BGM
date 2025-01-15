import logging
from datetime import datetime


def setup_logger(name, log_file, level=logging.INFO):
    """
    Setup a logger for tracking experiments and debugging.
    :param name: Name of the logger.
    :param log_file: Path to the log file.
    :param level: Logging level (e.g., logging.INFO).
    :return: Configured logger instance.
    """
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


class ExperimentLogger:
    def __init__(self, log_dir):
        """
        Logger for tracking experiment details such as training metrics and configurations.
        :param log_dir: Directory to save the log file.
        """
        self.log_dir = log_dir
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_file = f"{self.log_dir}/experiment_{timestamp}.log"
        self.logger = setup_logger("experiment_logger", log_file)

    def log_config(self, config):
        """
        Log experiment configurations (e.g., hyperparameters).
        :param config: Dictionary of configuration parameters.
        """
        self.logger.info("Experiment Configurations:")
        for key, value in config.items():
            self.logger.info(f"{key}: {value}")

    def log_metrics(self, epoch, metrics):
        """
        Log training or evaluation metrics.
        :param epoch: Current epoch number.
        :param metrics: Dictionary of metric names and values.
        """
        self.logger.info(f"Epoch {epoch}:")
        for key, value in metrics.items():
            self.logger.info(f"{key}: {value:.4f}")


# Example usage:
# logger = ExperimentLogger("logs")
# logger.log_config({"learning_rate": 0.001, "batch_size": 32})
# logger.log_metrics(1, {"loss": 0.234, "accuracy": 0.89})
