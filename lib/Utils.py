import os
import random
import numpy as np
import torch


def set_seed(seed):
    """
    Set random seed for reproducibility.
    :param seed: Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dir(dir_path):
    """
    Create a directory if it does not exist.
    :param dir_path: Path to the directory.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_model(model, path):
    """
    Save a PyTorch model to a file.
    :param model: Model instance.
    :param path: Path to save the model.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path):
    """
    Load a PyTorch model from a file.
    :param model: Model instance to load weights into.
    :param path: Path to the model file.
    """
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
