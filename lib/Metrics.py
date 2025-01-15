import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


def rmse(predictions, targets):
    """
    Compute Root Mean Squared Error (RMSE) using NumPy.
    :param predictions: Predicted values (NumPy array or list).
    :param targets: Ground truth values (NumPy array or list).
    :return: RMSE value.
    """
    mse = mean_squared_error(targets, predictions)
    return np.sqrt(mse)


def mae(predictions, targets):
    """
    Compute Mean Absolute Error (MAE) using NumPy.
    :param predictions: Predicted values (NumPy array or list).
    :param targets: Ground truth values (NumPy array or list).
    :return: MAE value.
    """
    return mean_absolute_error(targets, predictions)


def torch_rmse(predictions, targets):
    """
    Compute Root Mean Squared Error (RMSE) using PyTorch tensors.
    :param predictions: Predicted values (torch.Tensor).
    :param targets: Ground truth values (torch.Tensor).
    :return: RMSE value as a scalar tensor.
    """
    mse = torch.mean((predictions - targets) ** 2)
    return torch.sqrt(mse)


def torch_mae(predictions, targets):
    """
    Compute Mean Absolute Error (MAE) using PyTorch tensors.
    :param predictions: Predicted values (torch.Tensor).
    :param targets: Ground truth values (torch.Tensor).
    :return: MAE value as a scalar tensor.
    """
    return torch.mean(torch.abs(predictions - targets))


# Additional Helper Function for Batch Evaluation
def evaluate_metrics(predictions, targets):
    """
    Evaluate both RMSE and MAE for a batch of predictions and targets.
    :param predictions: Predicted values (NumPy array or list).
    :param targets: Ground truth values (NumPy array or list).
    :return: Dictionary containing RMSE and MAE values.
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    return {
        "RMSE": rmse(predictions, targets),
        "MAE": mae(predictions, targets)
    }


def torch_evaluate_metrics(predictions, targets):
    """
    Evaluate both RMSE and MAE for a batch of predictions and targets using PyTorch tensors.
    :param predictions: Predicted values (torch.Tensor).
    :param targets: Ground truth values (torch.Tensor).
    :return: Dictionary containing RMSE and MAE values as tensors.
    """
    return {
        "RMSE": torch_rmse(predictions, targets),
        "MAE": torch_mae(predictions, targets)
    }
